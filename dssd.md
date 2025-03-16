# Vision-Language Models (VLMs)👀

[![Last Updated](https://img.shields.io/badge/Updated-March%202025-brightpurple)](https://github.com/your-username/awesome-vision-language-models/commits/main)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)

A curated hub for researchers and developers exploring **Vision-Language Models (VLMs)** and **Multimodal Learning**. Includes seminal papers/models, benchmark, datasets, and research directions.

---

## 📚 Seminal Papers/Models (Post 2021)

- **Gemma3** (2025) - [Decoder-only architecture with 1B/4B/12B/27B parameters, supporting multimodality and 140+ languages, featuring a 128k-token context window and function calling capabilities, based on Google's Gemini 2.0 architecture](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
- **PH4 Mini** (2025) - [Decoder-only architecture with specialized parameters, designed for efficient completion tasks, using optimized transformer layers and Microsoft's Phi-4 framework as pretrained backbone](https://arxiv.org/pdf/2503.01743)
- **C4AI Aya Vision 32B** (2025) - [Decoder-only architecture with 32B parameters, designed for advanced visual language understanding, supporting 23 languages and featuring dynamic image processing capabilities](https://huggingface.co/CohereForAI/aya-vision-32b)
- **Ola** (2025) - [Decoder-only architecture with 7B parameters, using OryxViT for vision encoder and Qwen-2.5-7B, SigLIP-400M, Whisper-V3-Large, BEATs-AS2M(cpt2) as pretrained backbone](https://arxiv.org/pdf/2502.04328)
- **Qwen2.5-VL** (2025) - [Decoder-only architecture with 3B/7B/72B parameters, using redesigned ViT for vision encoder and Qwen2.5 as pretrained backbone](https://arxiv.org/pdf/2502.13923)
- **Ocean-OCR** (2025) - [Decoder-only architecture with 3B parameters, using NaViT for vision encoder and pretrained from scratch](https://arxiv.org/pdf/2501.15558)
- **SmolVLM** (2025) - [Decoder-only architecture with 250M & 500M parameters, using SigLIP for vision encoder and SmolLM as pretrained backbone](https://huggingface.co/blog/smolervlm)
- **Emu3** (2024) - [Decoder-only architecture with 7B parameters, using MoVQGAN for vision encoder and LLaMA-2 as pretrained backbone](https://arxiv.org/pdf/2409.18869)
- **NVLM** (2024) - [Encoder-decoder architecture with 8B-24B parameters, using custom ViT for vision encoder and Qwen-2-Instruct as pretrained backbone](https://arxiv.org/pdf/2409.11402
)
- **Qwen2-VL** (2024) - [Decoder-only architecture with 7B-14B parameters, using EVA-CLIP ViT-L for vision encoder and Qwen-2 as pretrained backbone](https://arxiv.org/pdf/2409.12191)
- **Pixtral** (2024) - [Decoder-only architecture with 12B parameters, using CLIP ViT-L/14 for vision encoder and Mistral Large 2 as pretrained backbone](https://arxiv.org/pdf/2410.07073)
- **LLaMA 3.2-vision** (2024) - [Decoder-only architecture with 11B-90B parameters, using CLIP for vision encoder and LLaMA-3.1 as pretrained backbone](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- **Baichuan Ocean Mini** (2024) - [Decoder-only architecture with 7B parameters, using CLIP ViT-L/14 for vision encoder and Baichuan as pretrained backbone](https://arxiv.org/html/2410.08565v1)
- **DeepSeek-VL2** (2024) - [Decoder-only architecture with 4.5B x 74 parameters, using SigLIP/SAMB for vision encoder and DeepSeekMoE as pretrained backbone](https://arxiv.org/pdf/2412.10302)
- **Qwen-VL** (2023) - [Encoder-decoder architecture with 7B parameters, using a Vision Transformer (ViT) for vision encoding and Qwen (a Transformer-based LLM) as the pretrained text backbone](https://arxiv.org/pdf/2308.12966)
- **ImageBind** (2023) - [Multi-encoder architecture with modality-specific encoders (ViT-H for vision, ~632M parameters) trained to align embeddings across 6 modalities (image, text, audio, depth, etc.)](https://arxiv.org/pdf/2305.05665)
- **InstructBLIP** (2023) - [Encoder-decoder architecture with 13B parameters, using ViT for vision encoder and Flan-T5, Vicuna as pretrained backbone](https://arxiv.org/pdf/2305.06500)
- **InternVL** (2023) - [Encoder-decoder architecture with 7B/20B parameters, using Eva CLIP ViT-g for vision encoder and QLLaMA as pretrained backbone](https://arxiv.org/pdf/2312.14238)
- **CogVLM** (2023) - [Encoder-decoder architecture with 18B parameters, using CLIP ViT-L/14 for vision encoder and Vicuna as pretrained backbone](https://arxiv.org/pdf/2311.03079)
- **BLIP-2** (2023) - [Encoder-decoder architecture with 7B-13B parameters, using ViT-g for vision encoder and Open Pretrained Transformer (OPT) as pretrained backbone](https://arxiv.org/pdf/2301.12597)
- **PaLM-E** (2023) - [Decoder-only architecture with 562B parameters, using ViT for vision encoder and PaLM as pretrained backbone](https://arxiv.org/pdf/2303.03378)
- **LLaVA-1.5** (2023) - [Decoder-only architecture with 13B parameters, using CLIP ViT-L/14 for vision encoder and Vicuna as pretrained backbone](https://arxiv.org/pdf/2304.08485)
- **Flamingo** (2022) - [Decoder-only architecture with 80B parameters, using custom vision encoder and Chinchilla as pretrained backbone](https://arxiv.org/pdf/2204.14198)
- **BLIP** (2022) - [Encoder-decoder architecture using ViT-B/L/g for vision encoder and pretrained from scratch for language encoder](https://arxiv.org/pdf/2201.12086)
- **CLIP** (2021) - [Dual-encoder architecture with ~400M parameters, using a Vision Transformer (ViT, e.g., ViT-L/14) for vision encoding and a Transformer for text encoding. Trained contrastively on 400M image-text pairs for multimodal alignment](https://arxiv.org/pdf/2103.00020)

---

## 📊 Benchmarks & Datasets

### General VLM Training Datasets
| Dataset         | Year | Description                                                                 |
|-----------------|------|-----------------------------------------------------------------------------|
| **LAION-5B**    | 2022 | 5.8B image-text pairs from Common Crawl, widely used for training VLMs.    |
| **ALIGN**       | 2021 | 1.8B image-text pairs with noisy alt-text, emphasizing cross-modal alignment. |
| **YFCC100M**    | 2015 | 99.2M Flickr images with metadata/tags (still used for diverse pretraining). |
| **CC3M/CC12M**  | 2021 | Conceptual Captions: 3M/12M image-caption pairs with web-derived captions. |

### Specialized VLM Datasets
### Image-Text Understanding
| Dataset               | Year | Description                                                                 |
|-----------------------|------|-----------------------------------------------------------------------------|
| **MSCOCO**            | 2014 | 328K images with 5 captions each (standard for captioning/VQA).            |
| **VQA v2**            | 2017 | Balanced QA pairs to reduce dataset bias (updated splits used post-2021). |
| **GQA**               | 2022 | 22M compositional questions requiring scene graph reasoning.              |
| **RefCOCO/+/g**       | 2016 | Referring expression comprehension datasets for object localization.      |
| **Visual Genome**     | 2017 | 108K images with dense annotations (objects, attributes, relationships).  |

### Multimodal Video-Text
| Dataset           | Year | Description                                                                 |
|-------------------|------|-----------------------------------------------------------------------------|
| **HowTo100M**     | 2019 | 100M video-text pairs from instructional videos.                           |
| **ActivityNet**   | 2021 | 20K videos with captions and QA pairs for video-language tasks.            |
| **MSRVTT**        | 2016 | 10K video clips with 200K descriptions (benchmark for retrieval/captioning). |

### Geospatial Datasets
| Dataset               | Year | Description                                                                 |
|-----------------------|------|-----------------------------------------------------------------------------|
| **AiRound**           | 2020 | Scene Understanding, Object Classification using RGB and Sentinel-2 imagery. |
| **FAIR1M**            | 2021 | Spatial Relation Classification, Referring Expression Detection, Captioning. |
| **xBD**               | 2019 | Bounding Box, Instance Mask, Class annotations for disaster damage assessment. |

### Benchmarks
### Standard Evaluation
| Benchmark         | Year | Description                                                                 |
|-------------------|------|-----------------------------------------------------------------------------|
| **COCO-Caption**  | 2021 | Evaluates caption quality with BLEU, METEOR, CIDEr.                        |
| **VQA Challenge** | 2022 | Focuses on balanced QA accuracy across question types.                     |
| **GLUE/SuperGLUE**| 2022 | NLP benchmarks adapted to test VLM linguistic transfer to visual tasks.    |

### Specialized Task Benchmarks
| Benchmark           | Year | Description                                                                 |
|---------------------|------|-----------------------------------------------------------------------------|
| **Winoground**      | 2022 | Tests subtle image-text compositional reasoning.                           |
| **Hateful Memes**   | 2021 | Detects hate speech in multimodal memes (image + text).                    |
| **ScienceQA**       | 2022 | 21K science questions requiring visual-textual reasoning.                  |
| **OK-VQA**          | 2021 | 11K open-ended questions needing external knowledge.                       |

### Cross-Modal Retrieval
| Benchmark                  | Year | Description                                                                 |
|----------------------------|------|-----------------------------------------------------------------------------|
| **Flickr30k Entities**     | 2021 | Localizes objects based on referring expressions.                          |
| **MS-COCO Retrieval**      | 2022 | Tests image-text matching (recall@1/5/10).                                 |
| **Conceptual Captions Retrieval** | 2021 | Retrieval tasks for image-text pairs from CC3M/CC12M.               |

### Few-Shot & Zero-Shot Learning
| Benchmark         | Year | Description                                                                 |
|-------------------|------|-----------------------------------------------------------------------------|
| **Fewshot-VLM**   | 2023 | Evaluates adaptation to new tasks with limited examples.                   |
| **ZeroShot-VLM**  | 2023 | Tests generalization to unseen tasks without task-specific training.       |
| **MetaPrompt**    | 2023 | Measures domain generalization with unseen prompts/domains.                |

### Video-Language Benchmarks
| Benchmark         | Year | Description                                                                 |
|-------------------|------|-----------------------------------------------------------------------------|
| **ViLCo-Bench**   | 2024 | Evaluates continual learning models across video-text tasks.               |
| **ReXTime**       | 2024 | Tests temporal reasoning across different video segments.                  |

## Summary Table
| **Category**               | **Key Datasets/Benchmarks**                     |
|----------------------------|-----------------------------------------------|
| **General Training**        | LAION-5B, ALIGN, CC3M/CC12M                   |
| **Specialized VLM**         | GQA, RefCOCO, Visual Genome                   |
| **Multimodal Video**        | HowTo100M, ActivityNet, MSRVTT                |
| **Geospatial**              | AiRound, FAIR1M, xBD                          |
| **Specialized Benchmarks**  | Winoground, Hateful Memes, ScienceQA          |
| **Few/Zero-Shot**           | Fewshot-VLM, ZeroShot-VLM, MetaPrompt         |
| **Video-Language**          | ViLCo-Bench, ReXTime                          |

## Notes
- **YFCC100M** and **Visual Genome** are pre-2021 but remain widely used.
- **VQA v2** and **MSCOCO** are frequently updated with new splits/tasks post-2021.
- For licenses: LAION-5B (CC-BY), CC3M/CC12M (Google’s terms), ALIGN (proprietary).

---

## 🔍 Research Directions

### 🎯 Multi-Modal Alignment Improvement
- Advanced cross-modal attention mechanisms
- Hierarchical fusion strategies
- Unified embedding space optimization
- Contrastive learning refinements

### 🏗 Architecture Optimization
- Vision encoder/LM parameter ratio studies
- Modular component integration
- Efficient transformer variants
- Hybrid CNN-Transformer designs

### 🎛 Few-Shot/Zero-Shot Learning
- Meta-learning approaches
- Prompt engineering techniques
- Synthetic data generation
- Cross-task generalization methods

### 🌍 Cross-Lingual Capabilities
- Multilingual pretraining strategies
- Code-switching handling
- Cultural visual concept adaptation
- Low-resource language support

### 📈 Evaluation Metric Development
- Multidimensional benchmark suites
- Human-aligned assessment frameworks
- Bias quantification metrics
- Ecological validity measures

---

## 🚧 Current Challenges

### 💻 Computational Requirements
- Massive GPU/TPU cluster dependencies
- Energy consumption concerns
- Scaling limitations
- Inference latency issues

### 🏷 Data Quality & Annotation
- Cost of multimodal labeling
- Subjective interpretation risks
- Dataset drift challenges
- Privacy-preserving data collection

### 🧩 Contextual Understanding
- Long-range dependency modeling
- Situational ambiguity resolution
- Commonsense reasoning gaps
- Temporal relationship processing

### ⚖ Bias & Fairness
- Demographic representation gaps
- Cultural stereotype propagation
- Output toxicity risks
- Value alignment difficulties

### 🔗 Multi-Modal Integration
- Modality imbalance issues
- Granularity mismatch problems
- Synchronization challenges
- Information redundancy handling

---

## 🚀 Emerging Areas

### 🎥 Video Understanding
- Temporal attention mechanisms
- Action recognition
- Scene transition analysis
- Audio-visual-text fusion

### ⏱ Real-Time Interaction
- Streaming data processing
- Latency-accuracy tradeoffs
- Edge deployment optimization
- Interactive feedback systems

### 🤖 Embodied AI
- Robotic visual grounding
- Manipulation task planning
- Spatial reasoning
- Human-robot communication

### 🏥 Medical Imaging
- Radiology report generation
- Cross-modal retrieval (images ↔ literature)
- Differential diagnosis support
- Ethical deployment frameworks

### 🎓 Education
- Adaptive learning materials
- Multimodal tutoring systems
- Accessibility enhancements
- Automated assessment tools

---

## ⚖ Ethics & Best Practices

### ⚠ Bias Mitigation
- Dataset diversity audits
- Debiasing loss functions
- Fairness-aware sampling
- Continuous monitoring pipelines

### 🔒 Privacy Protection
- Differential privacy methods
- Federated learning approaches
- Data anonymization protocols
- Secure multi-party computation

### 💡 Transparency
- Model documentation standards
- Attention visualization tools
- Failure case repositories
- Uncertainty quantification

### 🛡 Safety & Security
- Content moderation systems
- Adversarial robustness testing
- Output watermarking
- Access control mechanisms

### 📜 Responsible Frameworks
- Impact assessment templates
- Stakeholder engagement guides
- Regulatory compliance checklists
- Ethical review board protocols 

---

## 🤝Contributing 
Help us build the ultimate VLM resource!  
- Add new papers with: `Year | Title | Link | Brief Description`
- Suggest new research directions
- Update benchmark leaderboards
- Add dataset cards with: `Name | Modalities | Size | License`
