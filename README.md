# Vision-Language Models (VLMs) Research Hub 🌐
[![Last Updated](https://img.shields.io/badge/Updated-March%202025-brightpurple)](https://github.com/your-username/awesome-vision-language-models/commits/main)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)

A comprehensive technical resource for researchers exploring **Vision-Language Models (VLMs)** and **Multimodal Learning**, featuring seminal papers/models, datasets, benchmarks, ethical challenges and research directions.

### 👆 Notes section contains notes that I found useful. You can add yours as well!

---

## 📚 Seminal Papers/Models (Post-2021)

### 2025
- **Gemma3** - [Decoder-only architecture with 1B/4B/12B/27B parameters, supporting multimodality and 140+ languages, featuring a 128k-token context window and function calling capabilities, based on Google's Gemini 2.0 architecture](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
- **PH4 Mini** - [Decoder-only architecture with specialized parameters, designed for efficient completion tasks, using optimized transformer layers and Microsoft's Phi-4 framework as pretrained backbone](https://arxiv.org/pdf/2503.01743)
- **C4AI Aya Vision 32B** - [Decoder-only architecture with 32B parameters, designed for advanced visual language understanding, supporting 23 languages and featuring dynamic image processing capabilities](https://huggingface.co/CohereForAI/aya-vision-32b)
- **Ola** - [Decoder-only architecture with 7B parameters, using OryxViT for vision encoder and Qwen-2.5-7B, SigLIP-400M, Whisper-V3-Large, BEATs-AS2M(cpt2) as pretrained backbone](https://arxiv.org/pdf/2502.04328)
- **Qwen2.5-VL** - [Decoder-only architecture with 3B/7B/72B parameters, using redesigned ViT for vision encoder and Qwen2.5 as pretrained backbone](https://arxiv.org/pdf/2502.13923)
- **Ocean-OCR** - [Decoder-only architecture with 3B parameters, using NaViT for vision encoder and pretrained from scratch](https://arxiv.org/pdf/2501.15558)
- **SmolVLM** - [Decoder-only architecture with 250M & 500M parameters, using SigLIP for vision encoder and SmolLM as pretrained backbone](https://huggingface.co/blog/smolervlm)

### 2024
- **Video-LLaVA** - [Video understanding model based on LLaVA architecture, designed to process and understand visual information in video format](https://arxiv.org/pdf/2311.10122)
- **PaliGemma** - [Multimodal understanding and generation model, designed to handle complex interactions between different data modalities](https://arxiv.org/pdf/2407.07726)
- **Emu3** - [Decoder-only architecture with 7B parameters, using MoVQGAN for vision encoder and LLaMA-2 as pretrained backbone](https://arxiv.org/pdf/2409.18869)
- **NVLM** - [Encoder-decoder architecture with 8B-24B parameters, using custom ViT for vision encoder and Qwen-2-Instruct as pretrained backbone](https://arxiv.org/pdf/2409.11402
)
- **Qwen2-VL** - [Decoder-only architecture with 7B-14B parameters, using EVA-CLIP ViT-L for vision encoder and Qwen-2 as pretrained backbone](https://arxiv.org/pdf/2409.12191)
- **Pixtral** - [Decoder-only architecture with 12B parameters, using CLIP ViT-L/14 for vision encoder and Mistral Large 2 as pretrained backbone](https://arxiv.org/pdf/2410.07073)
- **CoPali** - [Code generation model that focuses on programming assistance, designed to understand natural language instructions and generate corresponding code](https://arxiv.org/pdf/2407.01449)
- **SAM 2** - [Foundation model for image segmentation, designed to handle diverse segmentation tasks with minimal prompting](https://arxiv.org/pdf/2408.00714)
- **LLaMA 3.2-vision** - [Decoder-only architecture with 11B-90B parameters, using CLIP for vision encoder and LLaMA-3.1 as pretrained backbone](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- **Baichuan Ocean Mini** - [Decoder-only architecture with 7B parameters, using CLIP ViT-L/14 for vision encoder and Baichuan as pretrained backbone](https://arxiv.org/html/2410.08565v1)
- **DeepSeek-VL2** - [Decoder-only architecture with 4.5B x 74 parameters, using SigLIP/SAMB for vision encoder and DeepSeekMoE as pretrained backbone](https://arxiv.org/pdf/2412.10302)
- **Qwen-VL** - [Encoder-decoder architecture with 7B parameters, using a Vision Transformer (ViT) for vision encoding and Qwen (a Transformer-based LLM) as the pretrained text backbone](https://arxiv.org/pdf/2308.12966)
- **SigLip** - [Specialized architecture for sign language interpretation, using advanced gesture and motion recognition techniques](https://arxiv.org/pdf/2303.15343)

### 2023

- **ImageBind** - [Multi-encoder architecture with modality-specific encoders (ViT-H for vision, ~632M parameters) trained to align embeddings across 6 modalities (image, text, audio, depth, etc.)](https://arxiv.org/pdf/2305.05665)
- **InstructBLIP** - [Encoder-decoder architecture with 13B parameters, using ViT for vision encoder and Flan-T5, Vicuna as pretrained backbone](https://arxiv.org/pdf/2305.06500)
- **InternVL** - [Encoder-decoder architecture with 7B/20B parameters, using Eva CLIP ViT-g for vision encoder and QLLaMA as pretrained backbone](https://arxiv.org/pdf/2312.14238)
- **CogVLM** - [Encoder-decoder architecture with 18B parameters, using CLIP ViT-L/14 for vision encoder and Vicuna as pretrained backbone](https://arxiv.org/pdf/2311.03079)
- **Kosmos-2** - [Multimodal architecture that can handle images, text, and other modalities, designed for comprehensive understanding and generation across different data types](https://arxiv.org/pdf/2306.14824)
- **BLIP-2** - [Encoder-decoder architecture with 7B-13B parameters, using ViT-g for vision encoder and Open Pretrained Transformer (OPT) as pretrained backbone](https://arxiv.org/pdf/2301.12597)
- **PaLM-E** - [Decoder-only architecture with 562B parameters, using ViT for vision encoder and PaLM as pretrained backbone](https://arxiv.org/pdf/2303.03378)
- **LLaVA-1.5** - [Decoder-only architecture with 13B parameters, using CLIP ViT-L/14 for vision encoder and Vicuna as pretrained backbone](https://arxiv.org/pdf/2304.08485)

### 2022 & Prior

- **Flamingo** - [Decoder-only architecture with 80B parameters, using custom vision encoder and Chinchilla as pretrained backbone](https://arxiv.org/pdf/2204.14198)
- **BLIP** - [Encoder-decoder architecture using ViT-B/L/g for vision encoder and pretrained from scratch for language encoder](https://arxiv.org/pdf/2201.12086)
- **CLIP** - [Dual-encoder architecture with ~400M parameters, using a Vision Transformer (ViT, e.g., ViT-L/14) for vision encoding and a Transformer for text encoding. Trained contrastively on 400M image-text pairs for multimodal alignment](https://arxiv.org/pdf/2103.00020)    
- **FLAVA** - [Unified transformer architecture with separate image and text encoders plus a multimodal encoder, designed to handle vision, language, and multimodal reasoning tasks simultaneously](https://arxiv.org/pdf/2112.04482)

### 📌**Recent papers**
- [Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models](https://arxiv.org/pdf/2503.17142)  
- [Beyond Semantics: Rediscovering Spatial Awareness in Vision-Language Models](https://arxiv.org/pdf/2503.17349)  
- [CoMP: Continual Multimodal Pre-training for Vision Foundation Models](https://arxiv.org/pdf/2503.18931)  
- [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://arxiv.org/pdf/2503.19755)  
- [PAVE: Patching and Adapting Video Large Language Models](https://arxiv.org/pdf/2503.19794)  
- [FireEdit: Fine-grained Instruction-based Image Editing via Region-aware Vision Language Model](https://arxiv.org/pdf/2503.19839)  
- [Self-ReS: Self-Reflection in Large Vision-Language Models for Long Video Understanding](https://arxiv.org/pdf/2503.203)  
- [Towards Efficient and General-Purpose Few-Shot Misclassification Detection for Vision-Language Models](https://arxiv.org/pdf/2503.20492)  
- [IAP: Improving Continual Learning of Vision-Language Models via Instance-Aware Prompting](https://arxiv.org/pdf/2503.20612)  
- [BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding](https://arxiv.org/pdf/2503.21483)  
- [Fwd2Bot: LVLM Visual Token Compression with Double Forward Bottleneck](https://arxiv.org/pdf/2503.21757)  
- [OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement](https://arxiv.org/pdf/2503.17352)  
- [Scaling Vision Pre-Training to 4K Resolution](https://arxiv.org/pdf/2503.19903)  
- [CAFe: Unifying Representation and Generation with Contrastive-Autoregressive Finetuning](https://arxiv.org/pdf/2503.19900)
- [DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](https://arxiv.org/pdf/2503.13443)
- [EfficientLLaVA:Generalizable Auto-Pruning for Large Vision-language Models](https://arxiv.org/pdf/2503.15369)
- [JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse](https://arxiv.org/pdf/2503.16365)
- [MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](https://arxiv.org/pdf/2503.13440)
- [TULIP: Towards Unified Language-Image Pretraining](https://arxiv.org/pdf/2503.15485)
- [VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search](https://arxiv.org/pdf/2503.10582)
- [Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](https://arxiv.org/pdf/2503.11609)  
- [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/pdf/2503.01785)  
- [Quantifying Memorization and Retriever Performance in Retrieval-Augmented Vision-Language Models](https://arxiv.org/pdf/2502.13836)  
- [Large Language Diffusion Models](https://arxiv.org/pdf/2502.09992)  
- [Memory-Augmented Latent Transformers for Any-Length Video Generation](https://www.arxiv.org/abs/2502.12632)  
- [ Visual Attention Sink In Large Multimodal Models](https://arxiv.org/pdf/2503.03321)  
- [Unified Reward Model for Multimodal Understanding and Generation](https://arxiv.org/pdf/2503.05236)  
- [Should VLMs Be Pre-trained With Image Data?](https://arxiv.org/pdf/2503.07603)    
- [Pre-Instruction Data Selection for Visual Instruction Tuning](https://arxiv.org/pdf/2503.07591)  
- [VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](https://arxiv.org/pdf/2503.07523)  
- [Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/pdf/2503.09573)  
- [PLLaVA: Parameter-free LLaVA Extension from Images to Videos for Video Dense Captioning](https://arxiv.org/pdf/2404.16994)  
- [TextSquare: Scaling up Text-Centric Visual Instruction Tuning](https://arxiv.org/pdf/2404.12803)  
- [Vision-Flan: Scaling Human-Labeled Tasks in Visual Instruction Tuning](https://arxiv.org/pdf/2402.11690)  
- [MoE-LLaVA: Mixture of Experts for Large Vision-Language Models](https://arxiv.org/pdf/2401.15947)  
- [What matters when building vision-language models?](https://arxiv.org/pdf/2405.02246)  
- [MM-LLMs: Recent Advances in MultiModal Large Language Models](https://arxiv.org/pdf/2401.13601)  
- [VisionZip: Longer is Better but Not Necessary in Vision Language Models](https://arxiv.org/pdf/2412.04467)  
- [Encoder-Free Vision-Language Models](https://arxiv.org/pdf/2406.11832)  

---

## 📊 Datasets & Benchmarks

#### **Core Training Datasets**
- **COCO** - [Contains 328K images, each paired with 5 captions for image captioning and VQA](https://huggingface.co/datasets/HuggingFaceM4/COCO)  
- **Conceptual Captions** - [3M web-mined image-text pairs for pretraining VLMs](https://huggingface.co/datasets/conceptual_captions)  
- **LAION-5B** - [5B image-text pairs from Common Crawl for large-scale pretraining](https://laion.ai/blog/laion-5b/)  
- **ALIGN** - [1.8B noisy alt-text pairs for robust multimodal alignment](https://research.google/blog/align-scaling-up-visual-and-vision-language-representation-learning-with-noisy-text-supervision/)  
- **SBU Caption** - [1M image-text pairs from web pages](https://huggingface.co/datasets/sbu_captions)  
- **Visual Genome** - [5.4M object/relationship annotations](https://www.kaggle.com/datasets/mathurinache/visual-genome)  
- **WuKong** - [100M Chinese image-text pairs](https://wukong-dataset.github.io/wukong-dataset/)  

---
#### Image Classification
- **ImageNet-1k** (2009) - [1.28M training images across 1,000 classes](https://huggingface.co/datasets/imagenet-1k)  
- **CIFAR-10/100** (2009) - [60K low-resolution images for small-scale testing](https://huggingface.co/datasets/cifar10)  
- **Food-101** (2014) - [101 food categories with 1,000 images each](https://huggingface.co/datasets/food101)  

#### Object Detection
- **COCO Detection** (2017) - [118K images with 80 object categories](https://cocodataset.org/#detection-2017)  
- **LVIS** (2019) - [1,203 long-tail object categories](https://www.lvisdataset.org/)  

#### Semantic Segmentation
- **Cityscapes** (2016) - [5,000 urban scene images with pixel-level labels](https://www.cityscapes-dataset.com/)  
- **ADE20k** (2017) - [25K images with 150 object/part categories](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset)  

#### Action Recognition
- **UCF101** (2012) - [13K video clips across 101 actions](https://www.crcv.ucf.edu/data/UCF101.php)  
- **Kinetics700** (2019) - [500K video clips covering 700 human actions](https://paperswithcode.com/dataset/kinetics-700)  

#### Image-Text Retrieval
- **Flickr30k** (2014) - [31K images with dense textual descriptions](https://www.kaggle.com/datasets/adityajn105/flickr30k)  
- **COCO Retrieval** (2015) - [Standard benchmark for cross-modal matching](https://cocodataset.org/#home)  

#### **Instruction Tuning**
- **LLaVA Instruct** - [260K image-conversation pairs for instruction fine-tuning](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft)  

#### **Additional Datasets**
- **Open Images** - [9M images with multi-label annotations](https://docs.ultralytics.com/datasets/detect/open-images-v7/)  
- **Hateful Memes** - [10K memes for hate speech detection](https://paperswithcode.com/dataset/hateful-memes)  
- **EuroSAT** - [27K satellite images for land use classification](https://github.com/phelber/eurosat)  

---

## 🏆 Benchmarks

- [**Open VLM Leaderboard HF** (2025)](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- **Fewshot-VLM** (2025) - [Evaluates adaptation to new tasks with limited examples](https://arxiv.org/html/2501.02189v3)  
- **ZeroShot-VLM** (2023) - [Tests generalization to unseen tasks without task-specific training](https://arxiv.org/html/2305.14196v3)  
- **MetaPrompt** (2023) - [Measures domain generalization with unseen prompts/domains](https://ai.meta.com/research/publications/unibench-visual-reasoning-requires-rethinking-vision-language-beyond-scaling/)  

#### **Video-Language Benchmarks**
- **VLM²-Bench** (2024) - [Evaluates multi-image/video linking capabilities (9 subtasks, 3K+ test cases)](https://vlm2-bench.github.io/)  
- **ViLCo-Bench** (2024) - [Continual learning for video-text tasks](https://github.com/cruiseresearchgroup/ViLCo)  

#### **Dynamic Evaluation**
- **LiveXiv** (2024) - [Monthly-changing benchmark to prevent overfitting, estimates true model capabilities](https://arxiv.org/abs/2410.10783)  

#### **Specialized Tasks**
- **ScienceQA** (2022) - [21K science questions for multimodal reasoning](https://scienceqa.github.io/)  
- **OK-VQA** (2021) - [11K open-ended questions requiring external knowledge](https://okvqa.allenai.org/)    

---

## 🔍 Research Directions

### 🔸 Multimodal Alignment & Fusion
→ **Key Challenges**
- Modality gap between continuous visual features and discrete text tokens
- Information asymmetry (visual data >> textual descriptions)

```python
# Multimodal fusion with gated attention
class GatedCrossAttention(nn.Module):
    def __init__(self, dim):
        self.vision_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(2*dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, vision, text):
        v = self.vision_proj(vision)  # [B,N,D]
        t = self.text_proj(text)      # [B,M,D]
        gates = self.gate(torch.cat([v.mean(1), t.mean(1)], dim=-1))
        return v * gates + t * (1-gates)
```
→ **Open Problems**

- Theoretical analysis of joint embedding spaces

- Dynamic modality weighting for unbalanced inputs

---

### 🔸 Efficient Edge Deployment
↓ **SOTA Compression**

| Technique   | Parameters | VRAM | Latency | Accuracy |
|-------------|------------|------|---------|----------|
| 4-bit QAT   | 7B → 1.8B  | 6GB  | 14ms    | 92.3%    |           
| LayerDrop   | 7B → 3.1B  | 9GB  | 22ms    | 95.1%    |           
| MoE-Slim    | 7B → 2.4B  | 5GB  | 18ms    | 93.7%    |           

→ **Hardware-Software Codesign**

- TensorRT-LLM for VLMs

- NPU-optimized kernels (Huawei Ascend)

- FlashAttention-Edge for ARM GPUs

---

### 🔸 Embodied AI Integration
→ **Key Components**

- Visuomotor control pipelines

- Real-time 3D scene understanding

- Multimodal memory banks

| Task         | Dataset   | SOTA Accuracy | Human Level |
|--------------|-----------|---------------|-------------|
| Manipulation | RoboNet   | 68.3%         | 89%         |
| Navigation   | Habitat   | 72.1%         | 83%         |

---

### 🔸 Temporal Reasoning
→ **Architectural Innovations**

- 3D Sparse Attention (85% FLOPs reduction)

- Cross-Time Memory Banks

- Dynamic Time Warping for video-text alignment

→ **Applications**

- Climate change prediction (0.87 correlation)

- Surgical workflow analysis (91% phase recognition)

---

### 🔸 Medical VLMs
→ **Challenges**

- Hallucination in diagnosis (12% error rate)

- HIPAA-compliant training

→ **Emerging Solutions**

- Differential Privacy (ε=3.8, δ=1e-5)

- Anatomy-aware attention gates

- Multiscale fusion for radiology

---

##  ⚠️ Ethical Challenges

| Bias Type    | Prevalence | High-Risk Domains | Mitigation Effectiveness          |
|--------------|------------|-------------------|---------------------------------|
| Gender       | 23%        | Career images     | 63% reduction (Counterfactual) |
| Racial       | 18%        | Beauty standards  | 58% (Adversarial)               |
| Cultural     | 29%        | Religious symbols | 41% (Data Filtering)            |
| Hallucination| 34%        | Medical reports   | 71% (CHAIR metric)              |

---

## 🔒 Privacy Protection Framework
```bash
graph TD
    A[Raw Data] --> B{Federated Learning?}
    B -->|Yes| C[Differential Privacy]
    C --> D[Secure Training]
    B -->|No| E[Reject]
```
VLMs often process sensitive data (medical images, personal photos, etc.). This framework prevents data leakage while maintaining utility:

→ **Federated Learning Check**

  -  Purpose: Train models on decentralized devices without raw data collection
  -  Benefit: Processes user photos/text locally (e.g., mobile camera roll analysis)
  - Why Required: 34% of web-scraped training data contains private info (LAION audit)

→  **Differential Privacy (DP)**
 ```python
 # DP-SGD Implementation for Medical VLMs
optimizer = DPAdam(
    noise_multiplier=1.3,  
    l2_norm_clip=0.7,      
    num_microbatches=32
)
```
  - Guarantees formal privacy (ε=3.8, δ=1e-5)
- Prevents memorization of training images/text

→ **Secure Training**

- Homomorphic Encryption: Process encrypted chest X-rays/patient notes
- Trusted Execution Environments: Isolate retinal scan analysis
- Prevents: Model inversion attacks that reconstruct training images

→ **Reject Pathway**

- Triggered for:
  - Web data without consent (23% of WebLI dataset rejected)  
  - Protected health information (HIPAA compliance)
  - Biometric data under GDPR

↓ **Real-World Impact**
| Scenario              | Without Framework       | With Framework          |
|-----------------------|-------------------------|-------------------------|
| Medical VLM Training  | 12% patient ID leakage | 0.03% leakage risk    |
| Social Media Photos   | Memorizes user faces   | Anonymous embeddings    |
| Autonomous Vehicles | License plate storage   | Local processing only |

---

## 🛠️ Tools 

#### **Core Tools**
- **VLMEvalKit** - [Unified evaluation framework supporting 20+ benchmarks (Apache-2.0)](https://github.com/open-compass/VLMEvalKit)   
- **VLM-BiasCheck** - [Bias auditing toolkit covering 15+ dimensions (MIT)](https://arxiv.org/abs/2306.17202)  

#### **Optimization Toolkit**
- **4-bit QAT** - [Quantization-aware training via bitsandbytes (3.2× speedup)](https://github.com/TimDettmers/bitsandbytes)  
- **Flash Attention** - [Memory-efficient attention with xFormers (2.8× speedup)](https://github.com/Dao-AILab/flash-attention)  
- **Layer Dropping** - [Structural pruning via torch.prune (1.9× speedup)](https://paperswithcode.com/method/layerdrop)  

---

## 📌Emerging Applications 
#### **→ Healthcare**
- Surgical VLM: 91% instrument tracking

- Radiology Assistant: 0.92 AUC diagnosis

#### **→ Autonomous Systems**
- DriveVLM: 94ms scene understanding

- DroneNav: 82% obstacle avoidance

#### **→ Industrial**
- Quality Control: 99.3% defect detection

- Remote Sensing: 0.89 crop health correlation

---

## 🤝Contributing Guidelines

Thank you for considering contributing to the **Vision-Language Models (VLMs) Research Hub**! My goal is to create a comprehensive, community-driven resource for VLM researchers. Contributions ranging from updates to models, datasets, and benchmarks, to new code examples, ethical discussions, and research insights would be welcomed:)

---




 



