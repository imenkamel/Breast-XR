# Breast-XR: Toward Multimodal Explainable and Automated Breast Cancer Radiographic Reporting
## Dataset

The dataset used in this work is based on the publicly available [CDD-CESM](https://www.cancerimagingarchive.net/collection/cdd-cesm/) dataset, enriched with missing patient complaints generated via a Retrieval Augmented Generation (RAG) system to provide a more realistic clinical context.
## Descriptipn 
Breast-XR jointly leverages mammography images and key clinical information extracted from patient complaints to automatically generate structured, coherent, and clinically meaningful radiology reports.
Unlike prior unimodal approaches that rely exclusively on imaging data, Breast-XR integrates both visual and textual modalities through a transformer-based fusion strategy, achieving state-of-the-art performance on the CDD-CESM dataset.
A key novelty of the framework is its automated textual preprocessing pipeline: a RAG-based system synthesizes realistic patient complaints from medical reports, and Gemini Flash 2.0 then extracts the four clinically salient categories used as model input.
To enhance clinical trust, Breast-XR also provides visual and textual explainability via Grad-CAM and gradient-based saliency maps.
<img width="1725" height="1554" alt="metho (19)" src="https://github.com/user-attachments/assets/f2cf6d1f-0cbc-44fb-a350-3331cb95ea2a" />
## Training of Multimodal Automatic Medical Report Generation 
To train the multimodal AMRG model run Train.py
## Evaluation of Multimodal Automatic Medical Report Generation 
To evaluate the multimodal AMRG model run Eval.py
