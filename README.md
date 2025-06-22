# Multimodal/VLMs Research Hub 🌐
[![Last Updated](https://img.shields.io/badge/Updated-May%202025-brightpurple)](https://github.com/your-username/awesome-vision-language-models/commits/main)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow)](https://github.com/your-username/awesome-vision-language-models/blob/main/LICENSE)

A comprehensive technical resource for researchers exploring **Vision-Language Models (VLMs)** and **Multimodal Learning**, featuring seminal papers/models, datasets, benchmarks, ethical challenges and research directions.

---

## 📚 Seminal Papers/Models (Post-2021)

### 2025
- **Eagle 2.5** - [Decoder-only architecture with 8B parameters, designed for long-context multimodal learning, featuring a context window that can be dynamically adjusted based on input length. Utilizes progressive post-training to gradually increase the model's context window. Achieves 72.4% on Video-MME with 512 input frames, matching the results of top-tier commercial models like GPT-4o and large-scale open-source models like Qwen2.5-VL-72B and InternVL2.5-78B](https://github.com/NVlabs/EAGLE?tab=readme-ov-file)
- **InternVL3** - [Follows the "ViT-MLP-LLM" paradigm, integrating InternViT with various pre-trained LLMs. Comprises seven models ranging from 1B to 78B parameters. Supports text, images, and videos simultaneously. Features Variable Visual Position Encoding (V2PE) for better long-context understanding. Utilizes pixel unshuffling and dynamic resolution strategy for efficient image processing. Training data includes tool usage, GUI operations, scientific diagrams, etc. Can be deployed as an OpenAI-compatible API via LMDeploy's api_server](https://huggingface.co/OpenGVLab/InternVL3-78B)
- **Llama 3.2 Vision** - [Decoder-only architecture with 11B and 90B parameters, supports 8 languages - English, German, Italian, Portuguese, Hindi, Spanish, and Thai. NOTE: for image+text applications, English is the only language supported. Features a 128k-token context window, optimized for multimodal complex reasoning](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- **VLM-R1** - [Based on DeepSeek's R1 method, combined with Qwen2.5-VL, excels in Representational Expression Comprehension (REC), such as locating specific targets in images. Uses RL and SFT to enhance performance. Supports joint image and text processing. Provides open-source training code and dataset support](https://github.com/om-ai-lab/VLM-R1)
- **Kimi-VL** - [An efficient open-source MoE based VLM. Activates only 2.8B parameters in its language decoder (Kimi-VL-A3B). Demonstrates strong performance in multi-turn agent interaction tasks and diverse vision language tasks. Features a 128K extended context window for processing long inputs. Utilizes native-resolution vision encoder (MoonViT) for ultra-high-resolution visual inputs. Introduces Kimi-VL-Thinking variant for strong long-horizon reasoning capabilities](https://github.com/MoonshotAI/Kimi-VL/blob/main/Kimi-VL.pdf)
- **Molmo Series** - [Four models by Ai2: MolmoE-1B, Molmo-7B-O, Molmo-7B-D, and Molmo-72B. MolmoE-1B based on OLMoE-1B-7B, Molmo-7B-O on OLMo-7B-1024, Molmo-7B-D on Qwen2 7B, and Molmo-72B on Qwen2 72B. Known for high performance in vision-language tasks, with Molmo-72B outperforming many proprietary systems]()
- **Skywork-R1V** - [Open-sourced multimodal reasoning model with visual chain-of-thought capabilities. Supports mathematical and scientific analysis, and cross-modal understanding. Released quantized version (AWQ) for efficient inference](https://huggingface.co/Skywork/Skywork-R1V-38B)
- **DeepSeek Janus Series** - [Unified transformer architecture with decoupled visual encoding pathway, available in Janus-Pro (7B and 1B), Janus (1.3B), and JanusFlow (1.3B) parameter sizes, supports English and Chinese languages, designed to handle sequence lengths of up to 4,096 tokens](https://github.com/deepseek-ai/Janus)
- **Pixtral 12B & Pixtral Large** - [Decoder-only architecture with 12B (Pixtral 12B) and 124B (Pixtral Large) parameters, combines a 12B multimodal decoder with a 400M vision encoder. Pixtral Large integrates a 1B visual encoder with Mistral Large 2. Supports dozens of languages and features a 128k-token context window. Developed by Mistral AI, known for its efficiency in handling interleaved image and text data](https://huggingface.co/mistralai/Pixtral-12B-2409)
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

### 📌**Recent papers (2025)**
- 
- [Dual-Stage Value-Guided Inference with Margin-Based Reward Adjustment for Fast and Faithful VLM Captioning](https://arxiv.org/pdf/2506.15649)
- [Demystifying the Visual Quality Paradox in Multimodal Large Language Models](https://arxiv.org/pdf/2506.15645)
- [OpenPath: Open-Set Active Learning for Pathology Image Classification via Pre-trained Vision-Language Models](https://arxiv.org/pdf/2506.15318)
- [Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models](https://arxiv.org/pdf/2506.15201)
- [PeRL: Permutation-Enhanced Reinforcement Learning for Interleaved Vision-Language Reasoning](https://arxiv.org/pdf/2506.14907)
- [ASCD: Attention-Steerable Contrastive Decoding for Reducing Hallucination in MLLM](https://arxiv.org/pdf/2506.14766)
- [Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](https://arxiv.org/pdf/2506.14674)
- [SIRI-Bench: Challenging VLMs' Spatial Intelligence through Complex Reasoning Tasks](https://arxiv.org/pdf/2506.14512)
- [Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](https://arxiv.org/pdf/2506.14473)
- [Adapting Lightweight Vision Language Models for Radiological Visual Question Answering](https://arxiv.org/pdf/2506.14451)
- [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](https://arxiv.org/pdf/2506.13757)
- [OTFusion: Bridging Vision-only and Vision-Language Models via Optimal Transport for Transductive Zero-Shot Learning](https://arxiv.org/pdf/2506.13723)
- [DualEdit: Dual Editing for Knowledge Updating in Vision-Language Models](https://arxiv.org/pdf/2506.13638)
- [MambaMia: A State-Space-Model-Based Compression for Efficient Video Understanding in Large Multimodal Models](https://arxiv.org/pdf/2506.13564)
- [Leveraging Vision-Language Pre-training for Human Activity Recognition in Still Images](https://arxiv.org/pdf/2506.13458)
- [Anomaly Object Segmentation with Vision-Language Models for Steel Scrap Recycling](https://arxiv.org/pdf/2506.13282)
- [VGR: Visual Grounded Reasoning](https://arxiv.org/pdf/2506.11991)
- [How Visual Representations Map to Language Feature Space in Multimodal LLMs](https://arxiv.org/pdf/2506.11976)
- [Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](https://arxiv.org/pdf/2506.11672)
- [EasyARC: Evaluating Vision Language Models on True Visual Reasoning](https://arxiv.org/pdf/2506.11595)
- [VFaith: Do Large Multimodal Models Really Reason on Seen Images Rather than Previous Memories?](https://arxiv.org/pdf/2506.11571)
- [Manager: Aggregating Insights from Unimodal Experts in Two-Tower VLMs and MLLMs](https://arxiv.org/pdf/2506.11515)
- [On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving](https://arxiv.org/pdf/2506.11472)
- [Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs](https://arxiv.org/pdf/2506.10967)
- [IQE-CLIP: Instance-aware Query Embedding for Zero-/Few-shot Anomaly Detection in Medical Domain](https://arxiv.org/pdf/2506.10730)
- [Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing](https://arxiv.org/pdf/2506.09965)
- [Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better](https://arxiv.org/pdf/2506.09040)
- [Vision Transformers Don't Need Trained Registers](https://arxiv.org/pdf/2506.08010)
- [CoMemo: LVLMs Need Image Context with Image Memory](https://arxiv.org/pdf/2506.06279)
- [Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models](https://arxiv.org/pdf/2506.06242)
- [Challenging Vision-Language Models with Surgical Data: A New Dataset and Broad Benchmarking Study](https://arxiv.org/pdf/2506.06232)
- [Full Conformal Adaptation of Medical Vision-Language Models](https://arxiv.org/pdf/2506.06076)
- [SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs](https://arxiv.org/pdf/2506.05344)
- [Refer to Anything with Vision-Language Prompts](https://arxiv.org/pdf/2506.05342)
- [VideoMolmo: Spatio-Temporal Grounding Meets Pointing](https://arxiv.org/pdf/2506.05336)
- [MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning]()
- [Does Your 3D Encoder Really Work? When Pretrain-SFT from 2D VLMs Meets 3D VLMs](https://arxiv.org/pdf/2506.05318)
- [MARBLE: Material Recomposition and Blending in CLIP-Space](https://arxiv.org/pdf/2506.05313)
- [Perceive Anything: Recognize, Explain, Caption, and Segment Anything in Images and Videos](https://arxiv.org/pdf/2506.05302)
- [Struct2D: A Perception-Guided Framework for Spatial Reasoning in Large Multimodal Models](https://arxiv.org/pdf/2506.04220)
- [Language-Image Alignment with Fixed Text Encoders](https://arxiv.org/pdf/2506.04209)
- [Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization](https://arxiv.org/pdf/2506.04039)
- [Vocabulary-free few-shot learning for Vision-Language Models](https://arxiv.org/pdf/2506.04005)
- [UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation](https://arxiv.org/pdf/2506.03147)
- [Targeted Forgetting of Image Subgroups in CLIP Models](https://arxiv.org/pdf/2506.03117)
- [Revisiting Continuity of Image Tokens for Cross-domain Few-shot Learning](https://arxiv.org/pdf/2506.03110)
- [EgoVLM: Policy Optimization for Egocentric Video Understanding](https://arxiv.org/pdf/2506.03097)
- [FuseLIP: Multimodal Embeddings via Early Fusion of Discrete Tokens](https://arxiv.org/pdf/2506.03096)
- [FuseLIP: Multimodal Embeddings via Early Fusion of Discrete Tokens](https://arxiv.org/pdf/2506.03096)
- [Dual-Process Image Generation](https://arxiv.org/pdf/2506.01955)
- [MLLMs Need 3D-Aware Representation Supervision for Scene Understanding](https://arxiv.org/pdf/2506.01946)
- [MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs](https://arxiv.org/pdf/2506.01850)
- [R2SM: Referring and Reasoning for Selective Masks](https://arxiv.org/pdf/2506.01795)
- [Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks](https://arxiv.org/pdf/2505.24876)
- [ProxyThinker: Test-Time Guidance through Small Visual Reasoners](https://arxiv.org/pdf/2505.24872)
- [Time Blindness: Why Video-Language Models Can't See What Humans Can?](https://arxiv.org/pdf/2505.24867)
- [Vision LLMs Are Bad at Hierarchical Visual Understanding, and LLMs Are the Bottleneck](https://arxiv.org/pdf/2505.24840)
- [CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](https://arxiv.org/pdf/2505.24816)
- [Reinforcing Video Reasoning with Focused Thinking](https://arxiv.org/pdf/2505.24718)
- [Conformal Prediction for Zero-Shot Models](https://arxiv.org/pdf/2505.24693)
- [Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](https://arxiv.org/pdf/2505.23766)
- [Impromptu VLA: Open Weights and Open Data for Driving Vision-Language-Action Models](https://arxiv.org/pdf/2505.23757)
- [Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence](https://arxiv.org/pdf/2505.23747)
- [To Trust Or Not To Trust Your Vision-Language Model's Prediction](https://arxiv.org/pdf/2505.23745)
- [PIXELTHINK:Towards Efficient Chain-of-Pixel Reasoning](https://arxiv.org/pdf/2505.23727)
- [DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers](https://arxiv.org/pdf/2505.23694)
- [VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](https://arxiv.org/pdf/2505.23693)
- [Grounded Reinforcement Learning for Visual Reasoning](https://arxiv.org/pdf/2505.23678)
- [Zero-Shot Vision Encoder Grafting via LLM Surrogates](https://arxiv.org/pdf/2505.22664)
- [VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models](https://arxiv.org/pdf/2505.22654)
- [Sherlock: Self-Correcting Reasoning in Vision-Language Models](https://arxiv.org/pdf/2505.22651)
- [SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning](https://arxiv.org/pdf/2505.22596)
- [Thinking with Generated Images](https://arxiv.org/pdf/2505.22525)
- [ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models](https://arxiv.org/pdf/2505.21500)
- [Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment](https://arxiv.org/pdf/2505.21494)
- [Mitigating Hallucination in Large Vision-Language Models via Adaptive Attention Calibration](https://arxiv.org/pdf/2505.21472)
- [ID-Align: RoPE-Conscious Position Remapping for Dynamic High-Resolution Adaptation in Vision-Language Models](https://arxiv.org/pdf/2505.21465)
- [Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO](https://arxiv.org/pdf/2505.21457)
- [GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution](https://arxiv.org/pdf/2505.21375)
- [Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?](https://arxiv.org/pdf/2505.21374)
- [VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection](https://arxiv.org/pdf/2505.20289)
- [VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction](https://arxiv.org/pdf/2505.20279)
- [Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning](https://arxiv.org/pdf/2505.20272)
- [Seeing is Believing, but How Much? A Comprehensive Analysis of Verbalized Calibration in Vision-Language Models](https://arxiv.org/pdf/2505.20236)
- [Hard Negative Contrastive Learning for Fine-Grained Geometric Understanding in Large Multimodal Models](https://arxiv.org/pdf/2505.20152)
- [TUNA: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](https://arxiv.org/pdf/2505.20124)
- [TokBench: Evaluating Your Visual Tokenizer before Visual Generation](https://arxiv.org/pdf/2505.18142)
- [Adapting SAM 2 for Visual Object Tracking: 1st Place Solution for MMVPR Challenge Multi-Modal Tracking](https://arxiv.org/pdf/2505.18111)
- [TokBench: Evaluating Your Visual Tokenizer before Visual Generation](https://arxiv.org/pdf/2505.18142)
- [One RL to See Them All: Visual Triple Unified Reinforcement Learning](https://arxiv.org/pdf/2505.18129)
- [Adapting SAM 2 for Visual Object Tracking: 1st Place Solution for MMVPR Challenge Multi-Modal Tracking](https://arxiv.org/pdf/2505.18111)
- [FDBPL: Faster Distillation-Based Prompt Learning for Region-Aware Vision-Language Models Adaptation](https://arxiv.org/pdf/2505.18053)
- [LookWhere? Efficient Visual Recognition by Learning Where to Look and What to See from Self-Supervision](https://arxiv.org/pdf/2505.18051)
- [Clip4Retrofit: Enabling Real-Time Image Labeling on Edge Devices via Cross-Architecture CLIP Distillation](https://arxiv.org/pdf/2505.18039)
- [Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](https://arxiv.org/pdf/2505.17982)
- [GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning](https://arxiv.org/pdf/2505.17022)
- [SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](https://arxiv.org/pdf/2505.17018)
- [Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models](https://arxiv.org/pdf/2505.17015)
- [SpatialScore: Towards Unified Evaluation for Multimodal Spatial Understanding](https://arxiv.org/pdf/2505.17012)
- [SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](https://arxiv.org/pdf/2505.16805)
- [Self-Rewarding Large Vision-Language Models for Optimizing Prompts in Text-to-Image Generation](https://arxiv.org/pdf/2505.16763)
- [STAR-R1: Spacial TrAnsformation Reasoning by Reinforcing Multimodal LLMs](https://arxiv.org/pdf/2505.15804)
- [Visual Perturbation and Adaptive Hard Negative Contrastive Learning for Compositional Reasoning in Vision-Language Models](https://arxiv.org/pdf/2505.15576)
- [Clapper: Compact Learning and Video Representation in VLMs](https://arxiv.org/pdf/2505.15529)
- [Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](https://arxiv.org/pdf/2505.15510)
- [Prompt Tuning Vision Language Models with Margin Regularizer for Few-Shot Learning under Distribution Shifts](https://arxiv.org/pdf/2505.15506)
- [Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models](https://arxiv.org/pdf/2505.15489)
- [ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning](https://arxiv.org/pdf/2505.15447)
- [Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL](https://arxiv.org/pdf/2505.15436)
- [TimeCausality: Evaluating the Causal Ability in Time Dimension for Vision Language Models](https://arxiv.org/pdf/2505.15435)
- [Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks](https://arxiv.org/pdf/2505.15414)
- [Visual Question Answering on Multiple Remote Sensing Image Modalities](https://arxiv.org/pdf/2505.15401)
- [Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/pdf/2505.14683)
- [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.14677)
- [Beyond Words: Multimodal LLM Knows When to Speak](https://arxiv.org/pdf/2505.14654)
- [Investigating and Enhancing the Robustness of Large Multimodal Models Against Temporal Inconsistency](https://arxiv.org/pdf/2505.14405)
- [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/pdf/2505.14362)
- [Vision-Language Modeling Meets Remote Sensing: Models, Datasets and Perspectives](https://arxiv.org/pdf/2505.14361)
- [Speculative Decoding Reimagined for Multimodal Large Language Models](https://arxiv.org/pdf/2505.14260)
- [Aligning Attention Distribution to Information Flow for Hallucination Mitigation in Large Vision-Language Models](https://arxiv.org/pdf/2505.14257)
- [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/pdf/2505.14246)
- [UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning](https://arxiv.org/pdf/2505.14231)
- [G1: Bootstrapping Perception and Reasoning Abilities of Vision-Language Model via Reinforcement Learning](https://arxiv.org/pdf/2505.13426)
- [From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection](https://arxiv.org/pdf/2505.13233)
- [Uniformity First: Uniformity-aware Test-time Adaptation of Vision-language Models against Image Corruption](https://arxiv.org/pdf/2505.12912)
- [End-to-End Vision Tokenizer Tuning](https://arxiv.org/pdf/2505.10562)
- [MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](https://arxiv.org/pdf/2505.10557)
- [Exploring Implicit Visual Misunderstandings in Multimodal Large Language Models through Attention Analysis](https://arxiv.org/pdf/2505.10541)
- [UniEval: Unified Holistic Evaluation for Unified Multimodal Understanding and Generation](https://arxiv.org/pdf/2505.10483)
- [Vision language models have difficulty recognizing virtual objects](https://arxiv.org/pdf/2505.10453)
- [MSCI: Addressing CLIP's Inherent Limitations for Compositional Zero-Shot Learning](https://arxiv.org/pdf/2505.10289)
- [Variational Visual Question Answering](https://arxiv.org/pdf/2505.09591)
- [Flash-VL 2B: Optimizing Vision-Language Model Performance for Ultra-Low Latency and High Throughput](https://arxiv.org/pdf/2505.09498)
- [A 2D Semantic-Aware Position Encoding for Vision Transformers](https://arxiv.org/pdf/2505.09466)
- [Endo-CLIP: Progressive Self-Supervised Pre-training on Raw Colonoscopy Records](https://arxiv.org/pdf/2505.09435)
- [MAKE: Multi-Aspect Knowledge-Enhanced Vision-Language Pretraining for Zero-shot Dermatological Assessment](https://arxiv.org/pdf/2505.09372)
- [Extending Large Vision-Language Model for Diverse Interactive Tasks in Autonomous Driving](https://arxiv.org/pdf/2505.08725)
- [TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series](https://arxiv.org/pdf/2505.08723)
- [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/pdf/2505.08617)
- [Beyond CLIP Generalization: Against Forward&Backward Forgetting Adapter for Continual Learning of Vision-Language Models](https://arxiv.org/pdf/2505.07690)
- [MAIS: Memory-Attention for Interactive Segmentation](https://arxiv.org/pdf/2505.07511)
- [Register and CLS tokens yield a decoupling of local and global features in large ViTs](https://arxiv.org/pdf/2505.05892)
- [VLM Q-Learning: Aligning Vision-Language Models for Interactive Decision-Making](https://arxiv.org/pdf/2505.03181)
- [MM-Skin: Enhancing Dermatology Vision-Language Model with an Image-Text Dataset Derived from Textbooks](https://arxiv.org/pdf/2505.06152)
- [Does CLIP perceive art the same way we do?](https://arxiv.org/pdf/2505.05229)
- [Mapping User Trust in Vision Language Models: Research Landscape, Challenges, and Prospects](https://arxiv.org/pdf/2505.05318)
- [TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation](https://arxiv.org/pdf/2505.05422)
- [Adaptive Markup Language Generation for Contextually-Grounded Visual Document Understanding](https://arxiv.org/pdf/2505.05446)
- [DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](https://arxiv.org/pdf/2505.04410)
- [OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning](https://arxiv.org/pdf/2505.04601)
- [ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant](https://arxiv.org/pdf/2505.03654)
- [Investigating Zero-Shot Diagnostic Pathology in Vision-Language Models with Efficient Prompt Design](https://arxiv.org/pdf/2505.00134)
- [Detecting and Mitigating Hateful Content in Multimodal Memes with Vision-Language Models](https://arxiv.org/pdf/2505.00150)
- [V3LMA: Visual 3D-enhanced Language Model for Autonomous Driving](https://arxiv.org/pdf/2505.00156)
- [AdCare-VLM: Leveraging Large Vision Language Model (LVLM) to Monitor Long-Term Medication Adherence and Care](https://arxiv.org/pdf/2505.00275)
- [Visual Test-time Scaling for GUI Agent Grounding](https://arxiv.org/pdf/2505.00684)
- [Vision-Language Model-Based Semantic-Guided Imaging Biomarker for Early Lung Cancer Detection](https://arxiv.org/pdf/2504.21344)
- [Rethinking Visual Layer Selection in Multimodal LLMs](https://arxiv.org/pdf/2504.21447)
- [Black-Box Visual Prompt Engineering for Mitigating Object Hallucination in Large Vision Language Models](https://arxiv.org/pdf/2504.21559)
- [Early Exit and Multi Stage Knowledge Distillation in VLMs for Video Summarization](https://arxiv.org/pdf/2504.21831)
- [DeepAndes: A Self-Supervised Vision Foundation Model for Multi-Spectral Remote Sensing Imagery of the Andes](https://arxiv.org/pdf/2504.20303)
- [MicarVLMoE: A Modern Gated Cross-Aligned Vision-Language Mixture of Experts Model for Medical Image Captioning and Report Generation](https://arxiv.org/pdf/2504.20343)
- [Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception](https://arxiv.org/pdf/2504.20468)
- [SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data](https://arxiv.org/pdf/2504.20648)
- [FedMVP: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models](https://arxiv.org/pdf/2504.20860)
- [YoChameleon: Personalized Vision and Language Generation](https://arxiv.org/pdf/2504.20998)
- [Contrastive Language-Image Learning with Augmented Textual Prompts for 3D/4D FER Using Vision-Language Model](https://arxiv.org/pdf/2504.19739)
- [Back to Fundamentals: Low-Level Visual Features Guided Progressive Token Pruning](https://arxiv.org/pdf/2504.17996)
- [A Large Vision-Language Model based Environment Perception System for Visually Impaired People](https://arxiv.org/pdf/2504.18027)
- [ActionArt: Advancing Multimodal Large Models for Fine-Grained Human-Centric Video Understanding](https://arxiv.org/pdf/2504.18152)
- [E-InMeMo: Enhanced Prompting for Visual In-Context Learning](https://arxiv.org/pdf/2504.18158)
- [Revisiting Data Auditing in Large Vision-Language Models](https://arxiv.org/pdf/2504.18349)
- [Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization](https://arxiv.org/pdf/2504.18397)
- [Vision language models are unreliable at trivial spatial cognition](https://arxiv.org/pdf/2504.16061)
- [MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](https://arxiv.org/pdf/2504.16083)
- [Object-Level Verbalized Confidence Calibration in Vision-Language Models via Semantic Perturbation](https://arxiv.org/pdf/2504.14848)
- [Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks: A Comprehensive Evaluation](https://arxiv.org/pdf/2504.14988)
- [Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs](https://arxiv.org/pdf/2504.15280)
- [Perception Encoder: The best visual embeddings are not at the output of the network](https://arxiv.org/pdf/2504.13181)
- [Vision and Language Integration for Domain Generalization](https://arxiv.org/pdf/2504.12966)
- [Low-hallucination Synthetic Captions for Large-Scale Vision-Language Model Pre-training](https://arxiv.org/pdf/2504.13123)
- [Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling](https://arxiv.org/pdf/2504.13169)
- [Logits DeConfusion with CLIP for Few-Shot Learning](https://arxiv.org/pdf/2504.12104)
- [Efficient Contrastive Decoding with Probabilistic Hallucination Detection - Mitigating Hallucinations in Large Vision Language Models](https://arxiv.org/pdf/2504.12137)
- [Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR](https://arxiv.org/pdf/2504.11101)
- [SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL](https://arxiv.org/pdf/2504.11455)
- [The Scalability of Simplicity: Empirical Analysis of Vision-Language Learning with a Single Transformer](https://arxiv.org/pdf/2504.10462)
- [SlowFastVAD: Video Anomaly Detection via Integrating Simple Detector and RAG-Enhanced Vision-Language Model](https://arxiv.org/pdf/2504.10320)
- [Investigating Vision-Language Model for Point Cloud-based Vehicle Classification](https://arxiv.org/pdf/2504.08154)
- [EO-VLM: VLM-Guided Energy Overload Attacks on Vision Models](https://arxiv.org/pdf/2504.08205)
- [VL-UR: Vision-Language-guided Universal Restoration of Images Degraded by Adverse Weather Conditions](https://arxiv.org/pdf/2504.08219)
- [VLMT: Vision-Language Multimodal Transformer for Multimodal Multi-hop Question Answering](https://arxiv.org/pdf/2504.08269)
- [Steering CLIP's vision transformer with sparse autoencoders](https://arxiv.org/pdf/2504.08729)
- [Taxonomy-Aware Evaluation of Vision-Language Models](https://arxiv.org/pdf/2504.05457)
- [A Lightweight Large Vision-language Model for Multimodal Medical Images](https://arxiv.org/pdf/2504.05575)
- [Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis](https://arxiv.org/pdf/2503.22420)
- [Breaking Language Barriers in Visual Language Models via Multilingual Textual Regularization](https://arxiv.org/pdf/2503.22577)
- [Adapting Vision Foundation Models for Real-time Ultrasound Image Segmentation](https://arxiv.org/pdf/2503.24368)
- [POPEN: Preference-Based Optimization and Ensemble for LVLM-Based Reasoning Segmentation](https://arxiv.org/pdf/2504.00640)
- [QG-VTC: Question-Guided Visual Token Compression in MLLMs for Efficient VQA](https://arxiv.org/pdf/2504.00654)
- [AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization](https://arxiv.org/pdf/2504.01735)
- [Prompting Medical Vision-Language Models to Mitigate Diagnosis Bias by Generating Realistic Dermoscopic Images](https://arxiv.org/pdf/2504.01838)
- [FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](https://arxiv.org/pdf/2504.01916)
- [Systematic Evaluation of Large Vision-Language Models for Surgical Artificial Intelligence](https://arxiv.org/pdf/2504.02799)
- [Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](https://arxiv.org/pdf/2504.02821)
- [STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection](https://arxiv.org/pdf/2504.02823)
- [It’s a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](https://arxiv.org/pdf/2503.24129)
- [Show or Tell? Effectively prompting Vision-Language Models for semantic segmentation](https://arxiv.org/pdf/2503.19647v1)  
- [ToVE: Efficient Vision-Language Learning via Knowledge Transfer from Vision Experts](https://arxiv.org/pdf/2504.00691)
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
- **Conceptual Captions 3M** - [3M web-mined image-text pairs for pretraining VLMs](https://huggingface.co/datasets/conceptual_captions)
- **Conceptual Captions 12M** - [12M web-mined image-text pairs for pretraining VLMs](https://github.com/google-research-datasets/conceptual-12m)
- **LAION400M** - [400 million image-text pairs](https://laion.ai/blog/laion-400-open-dataset/)  
- **LAION-5B** - [5B image-text pairs from Common Crawl for large-scale pretraining](https://laion.ai/blog/laion-5b/)  
- **ALIGN** - [1.8B noisy alt-text pairs for robust multimodal alignment](https://research.google/blog/align-scaling-up-visual-and-vision-language-representation-learning-with-noisy-text-supervision/)  
- **SBU Caption** - [1M image-text pairs from web pages](https://huggingface.co/datasets/sbu_captions)  
- **Visual Genome** - [5.4M object/relationship annotations](https://www.kaggle.com/datasets/mathurinache/visual-genome)  
- **WuKong** - [100M Chinese image-text pairs](https://wukong-dataset.github.io/wukong-dataset/) 
- **Localized Narratives** - [0.87 million image-text pairs](https://paperswithcode.com/dataset/localized-narratives)
- **Wikipedia-based Image Text** - [37.6 million image-text pairs, 108 languages](https://github.com/google-research-datasets/wit)
- **Red Caps** - [12 million image-text pairs](https://huggingface.co/datasets/kdexd/red_caps)
- **FILIP300M** - [300 million image-text pairs](https://openreview.net/forum?id=cpDhcsEDC2)
- **WebLI** - [12 billion image-text pairs](https://paperswithcode.com/dataset/webli)

---
#### Image Classification
- **ImageNet-1k** - [1.28M training images across 1,000 classes](https://huggingface.co/datasets/imagenet-1k)  
- **CIFAR-10/100** - [60K low-resolution images for small-scale testing](https://huggingface.co/datasets/cifar10)  
- **Food-101** - [101 food categories with 1,000 images each](https://huggingface.co/datasets/food101)
- **MNIST** - [Handwritten digit database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- **Caltech-101** - [101 object categories](https://paperswithcode.com/dataset/caltech-101)

#### Object Detection
- **COCO Detection** - [118K images with 80 object categories](https://cocodataset.org/#detection-2017)  
- **LVIS** - [1,203 long-tail object categories](https://www.lvisdataset.org/)
- **ODinW** - [Object detection in the wild](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw) 

#### Semantic Segmentation
- **Cityscapes** - [5,000 urban scene images with pixel-level labels](https://www.cityscapes-dataset.com/)  
- **ADE20k** - [25K images with 150 object/part categories](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset)
- **PASCAL VOC** - [Visual Object Classes dataset](https://paperswithcode.com/dataset/pascal-voc)  

#### Action Recognition
- **UCF101** - [13K video clips across 101 actions](https://www.crcv.ucf.edu/data/UCF101.php)  
- **Kinetics700** - [500K video clips covering 700 human actions](https://paperswithcode.com/dataset/kinetics-700)
- **RareAct** - [Dataset for recognizing rare actions](https://paperswithcode.com/dataset/rareact) 

#### Image-Text Retrieval
- **Flickr30k** (2014) - [31K images with dense textual descriptions](https://www.kaggle.com/datasets/adityajn105/flickr30k)  
- **COCO Retrieval** (2015) - [Standard benchmark for cross-modal matching](https://cocodataset.org/#home) 

#### Visual Question Answering (VQA)
- **MedFrameQA** - [A Multi-Image Medical VQA Benchmark for Clinical Reasoning](https://ucsc-vlaa.github.io/MedFrameQA/)
- **VQA** - [Visual Question Answering dataset](https://visualqa.org/)
- **GQA** - [Dataset for compositional question answering](https://paperswithcode.com/dataset/gqa)
- **OK-VQA** - [11K open-ended questions requiring external knowledge](https://okvqa.allenai.org/)
- **ScienceQA** - [21K science questions for multimodal reasoning](https://scienceqa.github.io/)
- **TextVQA** - [Dataset to read and reason about text in images](https://textvqa.org/)
 
#### **Instruction Tuning**
- **LLaVA Instruct** - [260K image-conversation pairs for instruction fine-tuning](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft)

#### **Bias**
- **MM-RLHF** - [Contains 120,000 finely annotated preference comparison pairs, including scores, rankings, textual descriptions of reasons, and tie annotations across three dimensions](https://mm-rlhf.github.io/)
- **Chicago Face Dataset (CFD)** - [Provides facial images of different races and genders, helps analyzing how models classify and identify faces of various demographic groups and uncover potential biases](https://www.chicagofaces.org/)
- **SocialCounterfactuals** - [Dataset containing 171,000 image-text pairs generated using an over-generate-then-filter method](https://huggingface.co/datasets/Intel/SocialCounterfactuals)
- **StereoSet** - [Dataset targeting stereotypical bias in multimodal models](https://huggingface.co/datasets/McGill-NLP/stereoset)

#### **Additional Datasets**
- **VideoMathQA** - [Benchmarking Mathematical Reasoning via Multimodal Understanding in Videos](https://mbzuai-oryx.github.io/VideoMathQA/#leaderboard-2)
- **FragFake** - [A Dataset for Fine-Grained Detection of Edited Images with Vision Language Models](https://huggingface.co/datasets/Vincent-HKUSTGZ/FragFake)
- **The P3 dataset** - [Pixels, Points and Polygons for Multimodal Building Vectorization](https://github.com/raphaelsulzer/PixelsPointsPolygons.git)
- **ReXGradient-160K** - [ A Large-Scale Publicly Available Dataset of Chest Radiographs with Free-text Reports](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K)
- **Open Images** - [9M images with multi-label annotations](https://docs.ultralytics.com/datasets/detect/open-images-v7/)  
- **Hateful Memes** - [10K memes for hate speech detection](https://paperswithcode.com/dataset/hateful-memes)  
- **EuroSAT** - [27K satellite images for land use classification](https://github.com/phelber/eurosat)
- **MathVista** - [Evaluating Math Reasoning in Visual Contexts](https://mathvista.github.io/)

---

## 🏆 Benchmarks

- **AV-Reasoner** (2025) - [Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs](https://av-reasoner.github.io/)
- **EOC-Bench** (2025) - [Can MLLMs Identify, Recall, and Forecast Objects in an Egocentric World?](https://circleradon.github.io/EOCBench/)
- **OmniSpatial** (2025) - [Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](https://qizekun.github.io/omnispatial/)
- **MMSI-Bench** (2025) - [A Benchmark for Multi-Image Spatial Intelligence](https://runsenxu.com/projects/MMSI_Bench/)
- **RBench-V** (2025) - [A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs](https://evalmodels.github.io/rbenchv/)
- **LENS** (2025) - [A multi-level benchmark explicitly designed to assess MLLMs across three hierarchical tiers—Perception, Understanding, and Reasoning—encompassing 8 tasks and 12 real-world scenarios.](https://github.com/Lens4MLLMs/LENS.git)
- **On Path to Multimodal Generalist: General-Level and General-Bench** (2025) - [Does higher performance across tasks indicate a stronger capability of MLLM, and closer to AGI?](https://arxiv.org/pdf/2505.04620)
- **VisuLogic** (2025) - [Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models](https://visulogic-benchmark.github.io/VisuLogic/) 
- **V2R-Bench** (2025) - [Holistically Evaluating LVLM Robustness to Fundamental Visual Variations](https://arxiv.org/pdf/2504.16727)
- **Open VLM Leaderboard** [HF(2025)](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- **SVLTA** (2025) - [Benchmarking Vision-Language temporal alignment via synthetic video situation](https://svlta-ai.github.io/SVLTA/)
- **Fewshot-VLM** (2025) - [Evaluates adaptation to new tasks with limited examples](https://arxiv.org/html/2501.02189v3) 
- **OCRBench & OCRBench v2** (2024) - [OCR capabilities of Large Multimodal Models](https://github.com/Yuliang-Liu/MultimodalOCR)
- **ZeroShot-VLM** (2023) - [Tests generalization to unseen tasks without task-specific training](https://arxiv.org/html/2305.14196v3)  
- **MetaPrompt** (2023) - [Measures domain generalization with unseen prompts/domains](https://ai.meta.com/research/publications/unibench-visual-reasoning-requires-rethinking-vision-language-beyond-scaling/)
- **MMBench** (2023) - [Multimodal benchmark](https://github.com/open-compass/MMBench) 
- **MMMU** (2023) - [Massive Multi-discipline Multimodal Understanding benchmark](https://mmmu-benchmark.github.io/)

#### **Video-Language Benchmarks**
- **Video-MMLU** (2025) - [A Massive Multi-Discipline Lecture Understanding Benchmark](https://enxinsong.com/Video-MMLU-web/)
- **VLM²-Bench** (2024) - [Evaluates multi-image/video linking capabilities (9 subtasks, 3K+ test cases)](https://vlm2-bench.github.io/)  
- **ViLCo-Bench** (2024) - [Continual learning for video-text tasks](https://github.com/cruiseresearchgroup/ViLCo)  

#### **Dynamic Evaluation**
- **LiveXiv** (2024) - [Monthly-changing benchmark to prevent overfitting, estimates true model capabilities](https://arxiv.org/abs/2410.10783)
- **MM-Vet** (2023) - [Evaluating Large Multimodal Models for Integrated Capabilities](https://github.com/yuweihao/MM-Vet)  

#### **Specialized Tasks**
- **ScienceQA** (2022) - [21K science questions for multimodal reasoning](https://scienceqa.github.io/)  
- **OK-VQA** (2021) - [11K open-ended questions requiring external knowledge](https://okvqa.allenai.org/)    

---

## 🔍 Research Directions

### 🔸 Regression Tasks
→ **Key Challenges**
- Effectively capturing numerical relationships
- Lack of tailored tokenizers for numerical values

→ **Open Problems**  
- Development of task-specific regression heads for VLMs

---

### 🔸 Diverse Visual Data
→ **Key Challenges**
- Adapting VLMs for multispectral and SAR data (remote sensing)  
- Processing multidimensional inputs  
- Integrating textual patient records with medical images    

→ **Applications**  
- Remote sensing analysis  
- Medical image analysis for diagnosis and treatment planning

---

### 🔸 Multimodal Output Beyond Text
→ **Key Challenges**
- Generating images, videos, and 3D data
- Adapting VLMs for dense prediction tasks

→ **Emerging Solutions**
- Adding text branches to computer vision models
- Using VLMs as agents interfacing with specialized output heads

---

### 🔸 Multitemporal Data Analysis
→ **Key Challenges**
- Analyzing sequences of visual and textual information over time  
- Identifying temporal trends and dependencies  

→ **Applications**
- Climate change monitoring
- Land-use change prediction

---

### 🔸 Efficient Edge Deployment
↓ **SOTA Compression**
| Technique | Parameters | VRAM | Latency | Accuracy |
|-----------|------------|------|---------|----------|
| 4-bit QAT | 7B → 1.8B  | 6GB  | 14ms    | 92.3%    |
| LayerDrop | 7B → 3.1B  | 9GB  | 22ms    | 95.1%    |
| MoE-Slim  | 7B → 2.4B  | 5GB  | 18ms    | 93.7%    |

→ **Hardware-Software Codesign**
- TensorRT-LLM for VLMs  
- NPU-optimized kernels (Huawei Ascend)  
- FlashAttention-Edge for ARM GPUs

---

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

| Bias Type          | Prevalence | High-Risk Domains     | Mitigation Effectiveness                     |
|--------------------|------------|-----------------------|----------------------------------------------|
| Gender             | 23%        | Career images         | 63% reduction (Counterfactual)               |
| Racial             | 18%        | Beauty standards      | 58% (Adversarial)                            |
| Cultural           | 29%        | Religious symbols     | 41% (Data Filtering)                         |
| Hallucination      | 34%        | Medical reports       | 71% (CHAIR metric)                           |
| Spatial Reasoning  | High       | Scene understanding   | Requires further research                    |
| Counting           | Moderate   | Object detection      | Requires specialized techniques              |
| Attribute Recognition| Moderate   | Detailed descriptions | Needs improved mechanisms                    |
| Prompt Ignoring    | Moderate   | Task-specific prompts | Requires better understanding of intent      |

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

#### **→ Automotive**
- ADAS (Advanced Driver-Assistance Systems)
- Autonomous Driving
- Driver Warning Systems

#### **→ Robotics**
- Enhanced environmental understanding
- Improved interaction through visual and linguistic cues  

#### **→ Augmented and Virtual Reality (AR/VR)**
- Real-world visual pattern analysis
- Overlaying relevant digital information  

#### **→ Real-time Processing and Edge Deployment**
- Real-time image captioning for social media
- Autonomous robot operation based on visual and textual instructions

#### **→ Industrial**
- Quality Control: 99.3% defect detection
- Remote Sensing: 0.89 crop health correlation

---

## 🤝Contributing Guidelines

Thank you for considering contributing to the **Vision-Language Models (VLMs) Research Hub**! My goal is to create a comprehensive, community-driven resource for Multimodal and VLM researchers. Contributions ranging from updates to models, datasets, and benchmarks, to new code examples, ethical discussions, and research insights would be welcomed:)

---




 



