

import os
import json
import docx
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import AdamW

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

# --- Albumentations (Section 3.3.2 du papier) ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)


TRAIN_JSON_PATH = r""
VAL_JSON_PATH   = r""    

BATCH_SIZE       = 8
EPOCHS           = 10
LR               = 3e-5       
WEIGHT_DECAY     = 0.01       
DROPOUT_FUSION   = 0.1        

DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T5_MODEL_NAME    = "t5-base"
BERT_MODEL_NAME  = "emilyalsentzer/Bio_ClinicalBERT"
IMAGE_SIZE       = 224        
SEQ_LEN_ENCODER  = 16
MAX_REPORT_TOKENS = 256       
BERT_MAX_LEN      = 128       

SAVE_DIR = "./save_breastxr_aligned"
os.makedirs(SAVE_DIR, exist_ok=True)


RADIMAGENET_WEIGHTS_URL = (
    "https://github.com/BMEII-AI/RadImageNet/releases/download/v1.0/"
    "RadImageNet-ResNet50_notop.pth"
)
RADIMAGENET_LOCAL_PATH = os.path.join(SAVE_DIR, "RadImageNet-ResNet50_notop.pth")

def download_radimagenet_weights(url: str, dest: str):
    """Télécharge les poids RadImageNet si absents localement."""
    if os.path.exists(dest):
        print(f"[RadImageNet] Poids trouvés localement : {dest}")
        return
    print(f"[RadImageNet] Téléchargement depuis {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[RadImageNet] Sauvegardé sous : {dest}")


train_albumentations = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),          
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


val_albumentations = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


def read_docx_text(path: str) -> str:
    """Lit et retourne le texte d'un fichier .docx."""
    try:
        doc = docx.Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception:
        return ""

def count_module_params(module: nn.Module) -> int:
    """Compte les paramètres entraînables d'un module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def detail_fusion_params(multimodal_model: nn.Module) -> int:
    """Affiche le détail des paramètres du module de fusion."""
    img_proj_count  = count_module_params(multimodal_model.img_proj)
    txt_proj_count  = count_module_params(multimodal_model.txt_proj)
    fusion_layer    = multimodal_model.fusion_transform
    fusion_layer_count = count_module_params(fusion_layer)
    num_layers      = multimodal_model.transformer_encoder.num_layers
    total_fusion    = img_proj_count + txt_proj_count + (fusion_layer_count * num_layers)

    t5_d    = multimodal_model.t5_d_model
    img_dim = multimodal_model.img_proj.in_features
    txt_dim = multimodal_model.txt_proj.in_features

    print("\n" + "=" * 70)
    print("Détail Paramètres Fusion/Projection :")
    print(f"  img_proj  : {img_proj_count:,}  ({img_dim} → {t5_d})")
    print(f"  txt_proj  : {txt_proj_count:,}  ({txt_dim} → {t5_d})")
    print(f"  Transformer ({num_layers} couches) : {fusion_layer_count * num_layers:,}")
    print(f"  TOTAL FUSION : {total_fusion:,}")
    print("=" * 70)
    return total_fusion


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

      
        try:
            pil_img = Image.open(img_path).convert('RGB')
            np_img  = np.array(pil_img)
        except Exception:
            np_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        
        if self.albu_transform:
            augmented = self.albu_transform(image=np_img)
            img       = augmented['image'].float()
        else:
            img = TF.to_tensor(Image.fromarray(np_img))

        
        report_text = ""
        if report_path and Path(report_path).exists():
            report_text = read_docx_text(report_path)
        else:
            report_text = e.get('report', "")

       
        tokenized = self.t5_tokenizer(
            report_text,
            truncation=True,
            padding='max_length',
            max_length=MAX_REPORT_TOKENS,
            return_tensors='pt'
        )
        labels = tokenized.input_ids.squeeze(0)
        labels[labels == self.t5_tokenizer.pad_token_id] = -100
        attention_mask_labels = tokenized.attention_mask.squeeze(0)

        return {
            'image':       img,
            'complaint':   complaint,
            'report_text': report_text,
            'labels':      labels,
            'labels_mask': attention_mask_labels,
        }

def collate_fn(batch):
    return {
        'images':       torch.stack([b['image'] for b in batch]),
        'complaints':   [b['complaint'] for b in batch],
        'labels':       torch.stack([b['labels'] for b in batch]),
        'labels_mask':  torch.stack([b['labels_mask'] for b in batch]),
    }


class ResNetFeatureExtractor(nn.Module):
    """
    Extracteur ResNet50 pré-entraîné sur RadImageNet (Section 3.4).
    Fallback vers ImageNet si les poids RadImageNet ne sont pas disponibles.
    """
    def __init__(self, radimagenet_path: str = None):
        super().__init__()
        
        rn = models.resnet50(weights=None)

        if radimagenet_path and os.path.exists(radimagenet_path):
            print(f"[ResNet50] Chargement des poids RadImageNet depuis : {radimagenet_path}")
            state = torch.load(radimagenet_path, map_location='cpu')
            
            missing, unexpected = rn.load_state_dict(state, strict=False)
            if missing:
                print(f"  [INFO] Clés manquantes (normal pour FC) : {missing}")
        else:
            print("[ResNet50] RadImageNet non trouvé → fallback sur ImageNet1K.")
            rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        
        modules = list(rn.children())[:-1]
        self.backbone = nn.Sequential(*modules)

    def forward(self, x):
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)   


class MultiModalToT5(nn.Module):
    
    def __init__(self,
                 image_model: nn.Module,
                 text_model:  nn.Module,
                 t5_d_model:  int = 768,
                 bert_out_dim: int = 768,
                 img_out_dim:  int = 2048,
                 seq_len:      int = SEQ_LEN_ENCODER,
                 dropout:      float = DROPOUT_FUSION):
        super().__init__()
        self.image_model = image_model
        self.text_model  = text_model
        self.seq_len     = seq_len
        self.t5_d_model  = t5_d_model

      
        self.img_proj = nn.Linear(img_out_dim, t5_d_model)   
        self.txt_proj = nn.Linear(bert_out_dim, t5_d_model)  

        
        self.fusion_transform = nn.TransformerEncoderLayer(
            d_model=t5_d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.fusion_transform, num_layers=2
        )

    def forward_backbone(self, images, complaints_inputs):
     
        img_feats = self.image_model(images)                
        img_proj  = self.img_proj(img_feats).unsqueeze(1)    

        
        bert_out  = self.text_model(**complaints_inputs)
        txt_cls   = bert_out.last_hidden_state[:, 0, :]     
        txt_proj  = self.txt_proj(txt_cls).unsqueeze(1)     

       
        fused        = torch.cat([img_proj, txt_proj], dim=1) 
        fused_output = self.transformer_encoder(fused)        
        return fused_output

    def forward(self, fused_vector):
        return fused_vector


def train():
    print(f"\n{'='*60}")
    print("  Breast-XR ")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}\n")

   
    download_radimagenet_weights(RADIMAGENET_WEIGHTS_URL, RADIMAGENET_LOCAL_PATH)

    
    t5_tokenizer  = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    t5_model      = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)

        for param in t5_model.get_encoder().parameters():
        param.requires_grad = False
    print("INFO : Encodeur T5 gelé (remplacé par le module de fusion multimodale).")

    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model     = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    print("INFO : BioClinicalBERT entièrement entraînable.")


    img_backbone = ResNetFeatureExtractor(
        radimagenet_path=RADIMAGENET_LOCAL_PATH
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

    
    t5_params      = count_module_params(t5_model.get_decoder())
    bert_params    = count_module_params(bert_model)
    resnet_params  = count_module_params(img_backbone)
    fusion_params  = detail_fusion_params(multimodal)
    total_params   = t5_params + bert_params + resnet_params + fusion_params

    print(f"\n{'='*60}")
    print("  Paramètres entraînables :")
    print(f"  T5 Decoder          : {t5_params:,}")
    print(f"  BioClinicalBERT     : {bert_params:,}")
    print(f"  ResNet50            : {resnet_params:,}")
    print(f"  Fusion/Projection   : {fusion_params:,}")
    print(f"  TOTAL               : {total_params:,}")
    print(f"{'='*60}\n")

    
    train_dataset = MammoReportDataset(
        TRAIN_JSON_PATH, t5_tokenizer,
        albumentations_transform=train_albumentations      
    val_dataset = MammoReportDataset(
        VAL_JSON_PATH, t5_tokenizer,
        albumentations_transform=val_albumentations        

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    print(f"Train : {len(train_dataset)} samples | Val : {len(val_dataset)} samples")

    
    all_params = (
        list(multimodal.parameters()) +
        list(t5_model.parameters()) +
        list(bert_model.parameters()) +
        list(img_backbone.parameters())
    )
    trainable_params = [p for p in all_params if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY   
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.05 * total_steps)      
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"Scheduler : {total_steps} steps totaux, {warmup_steps} steps warmup.\n")

    
    train_losses = []
    val_losses   = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        
        t5_model.train()
        multimodal.train()
        bert_model.train()

        epoch_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for batch in pbar:
            images     = batch['images'].to(DEVICE)
            complaints = batch['complaints']
            labels     = batch['labels'].to(DEVICE)

            bert_inputs = bert_tokenizer(
                complaints, return_tensors='pt',
                padding=True, truncation=True, max_length=BERT_MAX_LEN
            )
            bert_inputs = {k: v.to(DEVICE) for k, v in bert_inputs.items()}

            fused_output          = multimodal.forward_backbone(images, bert_inputs)
            encoder_hidden_states = multimodal(fused_output)
            encoder_outputs       = (encoder_hidden_states,)

            
            outputs = t5_model(encoder_outputs=encoder_outputs, labels=labels)
            loss    = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        
        t5_model.eval()
        multimodal.eval()
        bert_model.eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
                images     = batch['images'].to(DEVICE)
                complaints = batch['complaints']
                labels     = batch['labels'].to(DEVICE)

                bert_inputs = bert_tokenizer(
                    complaints, return_tensors='pt',
                    padding=True, truncation=True, max_length=BERT_MAX_LEN
                )
                bert_inputs = {k: v.to(DEVICE) for k, v in bert_inputs.items()}

                fused_output          = multimodal.forward_backbone(images, bert_inputs)
                encoder_hidden_states = multimodal(fused_output)
                encoder_outputs       = (encoder_hidden_states,)

                outputs = t5_model(encoder_outputs=encoder_outputs, labels=labels)
                epoch_val_loss += outputs.loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1:>2}/{EPOCHS} | "
              f"Train Loss : {avg_train_loss:.4f} | Val Loss : {avg_val_loss:.4f}")

       
        ckpt_path = os.path.join(SAVE_DIR, f"multimodal_t5_epoch{epoch+1}.pt")
        torch.save({
            't5_state_dict':        t5_model.state_dict(),
            'multimodal_state_dict': multimodal.state_dict(),
            'bert_state_dict':      bert_model.state_dict(),
            'optimizer_state':      optimizer.state_dict(),
            'epoch':                epoch + 1,
            'train_loss':           avg_train_loss,
            'val_loss':             avg_val_loss,
        }, ckpt_path)

       
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save({
                't5_state_dict':         t5_model.state_dict(),
                'multimodal_state_dict': multimodal.state_dict(),
                'bert_state_dict':       bert_model.state_dict(),
                'epoch':                 epoch + 1,
                'val_loss':              best_val_loss,
            }, best_path)
            print(f"  → Meilleur modèle sauvegardé (val loss = {best_val_loss:.4f})")

    print("\nEntraînement terminé.")

 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses,
             marker='o', linestyle='-',  color='blue',  label='Training curves')
    plt.plot(range(1, EPOCHS + 1), val_losses,
             marker='o', linestyle='--', color='green', label='Validation curves')

    plt.title('Courbes de Perte — Breast-XR (Figure 2)')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.xticks(range(1, EPOCHS + 1))
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(SAVE_DIR, "loss_curve_train_val.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbe sauvegardée : {curve_path}")

 
    losses_log = os.path.join(SAVE_DIR, "losses_log.json")
    with open(losses_log, 'w') as f:
        json.dump({'train': train_losses, 'val': val_losses}, f, indent=2)
    print(f"Logs des losses : {losses_log}")


if __name__ == '__main__':
    train()
