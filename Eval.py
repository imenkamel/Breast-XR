

import os
import json
import docx
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.modeling_outputs import BaseModelOutput


from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


TEST_JSON_PATH    = r"E:\MammoLLM\test_set_stratified.json"
SAVE_DIR          = "./save_breastxr_aligned"
CHECKPOINT_PATH   = os.path.join(SAVE_DIR, "best_model.pt")  

BATCH_SIZE        = 8
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T5_MODEL_NAME     = "t5-base"
BERT_MODEL_NAME   = "emilyalsentzer/Bio_ClinicalBERT"
IMAGE_SIZE        = 224
SEQ_LEN_ENCODER   = 16
MAX_REPORT_TOKENS = 256
BERT_MAX_LEN      = 128
DROPOUT_FUSION    = 0.1
RADIMAGENET_LOCAL = os.path.join(SAVE_DIR, "RadImageNet-ResNet50_notop.pth")


N_QUALITATIVE_EXAMPLES = 3


test_albumentations = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def read_docx_text(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    except Exception:
        return ""


class MammoReportDataset(Dataset):
    def __init__(self, json_path: str, tokenizer_t5: T5Tokenizer,
                 albumentations_transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.albu_transform = albumentations_transform
        self.t5_tokenizer   = tokenizer_t5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e           = self.data[idx]
        img_path    = e['image']
        complaint   = e.get('complaints', "")
        report_path = e.get('medical_report_path', None)
        patient_id  = e.get('patient_id', f'P{idx}')

        try:
            pil_img = Image.open(img_path).convert('RGB')
            np_img  = np.array(pil_img)
        except Exception:
            np_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        if self.albu_transform:
            img = self.albu_transform(image=np_img)['image'].float()
        else:
            img = TF.to_tensor(Image.fromarray(np_img))

        report_text = ""
        if report_path and Path(report_path).exists():
            report_text = read_docx_text(report_path)
        else:
            report_text = e.get('report', "")

        tokenized = self.t5_tokenizer(
            report_text, truncation=True,
            padding='max_length', max_length=MAX_REPORT_TOKENS,
            return_tensors='pt'
        )
        labels = tokenized.input_ids.squeeze(0)
        labels[labels == self.t5_tokenizer.pad_token_id] = -100

        return {
            'image':       img,
            'complaint':   complaint,
            'report_text': report_text,
            'labels':      labels,
            'patient_id':  patient_id,
        }

def collate_fn(batch):
    return {
        'images':       torch.stack([b['image'] for b in batch]),
        'complaints':   [b['complaint'] for b in batch],
        'report_texts': [b['report_text'] for b in batch],
        'labels':       torch.stack([b['labels'] for b in batch]),
        'patient_ids':  [b['patient_id'] for b in batch],
    }


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, radimagenet_path: str = None):
        super().__init__()
        rn = models.resnet50(weights=None)
        if radimagenet_path and os.path.exists(radimagenet_path):
            print(f"[ResNet50] Chargement RadImageNet : {radimagenet_path}")
            state = torch.load(radimagenet_path, map_location='cpu')
            rn.load_state_dict(state, strict=False)
        else:
            print("[ResNet50] Fallback ImageNet1K.")
            rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(rn.children())[:-1])

    def forward(self, x):
        return self.backbone(x).view(x.size(0), -1)


class MultiModalToT5(nn.Module):
    def __init__(self, image_model, text_model, t5_d_model=768,
                 bert_out_dim=768, img_out_dim=2048,
                 seq_len=SEQ_LEN_ENCODER, dropout=DROPOUT_FUSION):
        super().__init__()
        self.image_model = image_model
        self.text_model  = text_model
        self.seq_len     = seq_len
        self.t5_d_model  = t5_d_model
        self.img_proj    = nn.Linear(img_out_dim, t5_d_model)
        self.txt_proj    = nn.Linear(bert_out_dim, t5_d_model)
        self.fusion_transform = nn.TransformerEncoderLayer(
            d_model=t5_d_model, nhead=8,
            dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.fusion_transform, num_layers=2
        )

    def forward_backbone(self, images, complaints_inputs):
        img_feats = self.image_model(images)
        img_proj  = self.img_proj(img_feats).unsqueeze(1)
        with torch.no_grad():
            bert_out = self.text_model(**complaints_inputs)
        txt_proj = self.txt_proj(bert_out.last_hidden_state[:, 0, :]).unsqueeze(1)
        return self.transformer_encoder(torch.cat([img_proj, txt_proj], dim=1))

    def forward(self, fused_vector):
        return fused_vector


def compute_bleu_scores(generated, references):
    """BLEU-1 à BLEU-4 (Table 3 du papier)."""
    refs = [[r] for r in references]
    return {
        f'bleu_{n}': BLEU(max_ngram_order=n).corpus_score(generated, refs).score / 100
        for n in range(1, 5)
    }

def compute_rouge_scores(generated, references):
    """ROUGE-1, ROUGE-2, ROUGE-L (Table 3)."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    totals = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for gen, ref in zip(generated, references):
        scores = scorer.score(ref, gen)
        for k in totals:
            totals[k] += scores[k].fmeasure
    n = len(generated)
    return {k: v / n for k, v in totals.items()} if n > 0 else totals

def compute_meteor(generated, references):
    """METEOR (Table 3)."""
    scores = []
    for gen, ref in zip(generated, references):
        try:
            scores.append(meteor_score(
                [word_tokenize(ref.lower())],
                word_tokenize(gen.lower())
            ))
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0

def compute_bertscore(generated, references):
    """BERTScore F1 (Table 3) — lang='fr' pour les rapports en français."""
    _, _, F1 = bert_score.score(generated, references, lang="fr", verbose=True)
    return F1.mean().item()


def print_qualitative_examples(patient_ids, ground_truths, generated_reports, n=3):
    print(f"\n{'='*70}")
    print(f"  Analyse Qualitative — {n} exemples (cf. Figure 3 du papier)")
    print(f"{'='*70}")
    for i in range(min(n, len(patient_ids))):
        print(f"\n--- Patient : {patient_ids[i]} ---")
        print(f"[Vérité terrain]\n{ground_truths[i]}")
        print(f"\n[Rapport généré]\n{generated_reports[i]}")
        print("-" * 70)


def evaluate():
    print(f"\n{'='*60}")
    print("  Breast-XR — Évaluation (alignée avec Table 3, KES 2026)")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}\n")

  
    t5_tokenizer   = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    t5_model   = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    img_backbone = ResNetFeatureExtractor(
        radimagenet_path=RADIMAGENET_LOCAL
    ).to(DEVICE)

    multimodal = MultiModalToT5(
        image_model=img_backbone,
        text_model=bert_model,
        t5_d_model=t5_model.config.d_model,
        bert_out_dim=bert_model.config.hidden_size,
        img_out_dim=2048,
        seq_len=SEQ_LEN_ENCODER,
        dropout=DROPOUT_FUSION
    ).to(DEVICE)


    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint introuvable : {CHECKPOINT_PATH}")
    
    print(f"Chargement du checkpoint : {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    t5_model.load_state_dict(ckpt['t5_state_dict'])
    multimodal.load_state_dict(ckpt['multimodal_state_dict'])
    if 'bert_state_dict' in ckpt:
        bert_model.load_state_dict(ckpt['bert_state_dict'])
    print(f"  → Epoch : {ckpt.get('epoch', '?')} | Val Loss : {ckpt.get('val_loss', '?'):.4f}\n")

    t5_model.eval()
    multimodal.eval()
    bert_model.eval()

    test_dataset = MammoReportDataset(
        TEST_JSON_PATH, t5_tokenizer,
        albumentations_transform=test_albumentations
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )
    print(f"Test set : {len(test_dataset)} échantillons\n")

   
    total_test_loss = 0.0
    print("Calcul de la Test Loss...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Loss"):
            images     = batch['images'].to(DEVICE)
            complaints = batch['complaints']
            labels     = batch['labels'].to(DEVICE)

            bert_inputs = bert_tokenizer(
                complaints, return_tensors='pt',
                padding=True, truncation=True, max_length=BERT_MAX_LEN
            )
            bert_inputs = {k: v.to(DEVICE) for k, v in bert_inputs.items()}

            fused        = multimodal.forward_backbone(images, bert_inputs)
            enc_outputs  = (multimodal(fused),)
            loss         = t5_model(encoder_outputs=enc_outputs, labels=labels).loss
            total_test_loss += loss.item() * images.size(0)

    avg_test_loss = total_test_loss / len(test_dataset)

    
    all_generated    = []
    all_ground_truth = []
    all_patient_ids  = []

    print("\nGénération des rapports (beam search, num_beams=4)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Génération"):
            images     = batch['images'].to(DEVICE)
            complaints = batch['complaints']

            bert_inputs = bert_tokenizer(
                complaints, return_tensors='pt',
                padding=True, truncation=True, max_length=BERT_MAX_LEN
            )
            bert_inputs = {k: v.to(DEVICE) for k, v in bert_inputs.items()}

            fused   = multimodal.forward_backbone(images, bert_inputs)
            enc_out = BaseModelOutput(
                last_hidden_state=multimodal(fused),
                hidden_states=None, attentions=None
            )

            generated = t5_model.generate(
                encoder_outputs=enc_out,
                max_length=MAX_REPORT_TOKENS,
                num_beams=4,
                early_stopping=True,
                pad_token_id=t5_tokenizer.pad_token_id,
                eos_token_id=t5_tokenizer.eos_token_id
            )

            decoded = [
                t5_tokenizer.decode(g, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
                for g in generated
            ]
            all_generated.extend(decoded)
            all_ground_truth.extend(batch['report_texts'])
            all_patient_ids.extend(batch['patient_ids'])

    
    print("\nCalcul des métriques...")
    bleu   = compute_bleu_scores(all_generated, all_ground_truth)
    rouge  = compute_rouge_scores(all_generated, all_ground_truth)
    meteor = compute_meteor(all_generated, all_ground_truth)
    print("Calcul BERTScore (peut prendre quelques minutes)...")
    bert_f1 = compute_bertscore(all_generated, all_ground_truth)

    
    print(f"\n{'='*60}")
    print("  RÉSULTATS — Breast-XR (cf. Table 3, KES 2026)")
    print(f"{'='*60}")
    print(f"\n  Test Loss   : {avg_test_loss:.4f}")
    print(f"\n  {'Metric':<20} {'Score':>8}")
    print(f"  {'-'*28}")
    print(f"  {'BLEU-1':<20} {bleu['bleu_1']:>8.3f}")
    print(f"  {'BLEU-2':<20} {bleu['bleu_2']:>8.3f}")
    print(f"  {'BLEU-3':<20} {bleu['bleu_3']:>8.3f}")
    print(f"  {'BLEU-4':<20} {bleu['bleu_4']:>8.3f}")
    print(f"  {'ROUGE-1':<20} {rouge['rouge1']:>8.3f}")
    print(f"  {'ROUGE-2':<20} {rouge['rouge2']:>8.3f}")
    print(f"  {'ROUGE-L':<20} {rouge['rougeL']:>8.3f}")
    print(f"  {'METEOR':<20} {meteor:>8.3f}")
    print(f"  {'BERTScore':<20} {bert_f1:>8.3f}")
    print(f"{'='*60}\n")

    
    print_qualitative_examples(
        all_patient_ids, all_ground_truth, all_generated,
        n=N_QUALITATIVE_EXAMPLES
    )

    
    results = {
        'test_loss': avg_test_loss,
        'bleu':      bleu,
        'rouge':     rouge,
        'meteor':    meteor,
        'bertscore': bert_f1,
    }
    out_path = os.path.join(SAVE_DIR, "evaluation_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés : {out_path}")
    print("Évaluation terminée.")


if __name__ == '__main__':
    evaluate()
