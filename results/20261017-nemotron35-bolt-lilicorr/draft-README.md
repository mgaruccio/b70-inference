---
pipeline_tag: text-generation
base_model:
- nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
library_name: gguf
tags:
- nvidia
- ModelOpt
- Nemotron-3.5-Bolt
- latent-moe
- LiLiCorr
- gguf
---

# NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF

## Description

NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF is the LiLiCorr speculative-decoding checkpoint for [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF).
The compatible target is a hybrid LatentMoE vision-language model for reasoning, chat, and agent workflows.
For more information, see the [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) target card.

This checkpoint is not a standalone language model. It proposes draft token candidates that the compatible [24GB target](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) verifies; the target produces the user-visible text.

This model is ready for commercial or non-commercial use.

## License/Terms of Use

**GOVERNING DOWNLOAD TERMS:** Use of this model is governed by the [OpenMDW-1.1 model license](https://openmdw.ai/license/1-1/).

## Deployment Geography

Global

## Use Case

Developers can pair this draft with Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF for text generation.
The target supports reasoning, chat, retrieval-augmented generation, visual understanding, and agent workflows.
The draft accelerates generation through speculative decoding.

## Release Date

Hugging Face 10/15/2026 via https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF

## References

- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [NVIDIA Nemotron-3.5-Bolt-74B-A9B-BF16 reasoning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)
- [NVIDIA Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF reasoning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)
- [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/)
- [LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding](https://arxiv.org/abs/2608.20530)

## Model Architecture

**Architecture Type:** Dense GQA (Dense MLP + GQA Attention)

**Network Architecture:** Five dense feed-forward layers use grouped-query attention (GQA). Draft attention is bidirectional within the draft block. All five layers use a 1024-token per-head sliding window and attention sinks, with a grouped dynamic convolution around each attention and MLP sublayer. A separate two-layer LiLiCorr head correlates the drafter's top-8 candidates at each draft slot.

**Number of Model Parameters:** 1.113B draft parameters, excluding shared target embeddings and the LM head. The two-layer LiLiCorr head accounts for 37M of them.

**Target Model Architecture:** Hybrid LatentMoE vision-language model with 74B total parameters and 9B active parameters.

## Input

The inputs below describe the target-model serving interface. LiLiCorr drafts tokens using representations from the target model.

**Input Type(s):** Text
**Input Format(s):** Text: String
**Input Parameters:** Text: One-Dimensional (1D)
**Other Properties Related to Input:** The target model supports up to 256K tokens of context. Supported languages include English, French, Spanish, Italian, German, Japanese, Hindi, Korean, Brazilian Portuguese, and Chinese.

## Output

- **Output Type(s):** Text
- **Output Format:** String
- **Output Parameters:** One-Dimensional (1D): Sequences
- **Other Properties Related to Output:** Maximum context length up to 256K tokens. Supports JSON output format, reasoning output with chain-of-thought, and tool calling.

**Other Properties Related to Output:** Outputs can include responses, reasoning traces, tool-use content, and structured data.
The chat template and application configuration determine the output form.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

**Supported Runtime Engine(s):**

- llama.cpp

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA Blackwell, including DGX Spark (GB10)

**Preferred Operating System(s):**

- Linux
- Windows

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s)

The model is a NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF speculative decoding version quantized with Model Optimizer @ b5f10f8

## Training and Evaluation Datasets

## Training Dataset

For target training details, see the [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) target card.

**Link:** [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2), [Nemotron-RL-Super-Training-Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends), [Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3), [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1), and [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1).
Only dataset prompts and conversation context were used for data synthesis.
The original responses were not training targets.
Every training response was generated by Nemotron-3.5-Bolt at temperature 1.0.
The draft was first trained on responses from a Nemotron-3.5-Bolt post-training checkpoint, then fine-tuned on responses from the released checkpoint.

**Data Modality:** Text

**Training Data Size:** 1 Billion to 10 Trillion Tokens

**Data Collection Method by dataset:** Hybrid: Automated, manually-collected, synthetic.

**Labeling Method by dataset:** Hybrid: Automated, manually-labelled, synthetic.

**Properties:** This model was trained exclusively on a post-training corpus includes synthetic data for reasoning, code, science, tool use, instruction following, structured outputs, and multilingual tasks.

For more information about the target model's training data, see the [Public Summary of Training Content](https://developer.download.nvidia.com/assets/nemo/docs/public-summary-of-training-content-for-nvidia-nemotron.pdf).

## Evaluation Dataset

The model is evaluated on [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) for speculative decoding.

**Data Collection Method by dataset:** Hybrid: Automated, manually-collected, synthetic.

**Labeling Method by dataset:** Hybrid: Automated, manually-labelled, synthetic.

**Properties:** This corpus comprises a mix of high-quality standard benchmarks and test suites for LLMs. These benchmarks prompt the underlying model on a diverse set of tasks such as coding, math, writing, translation, etc., and the acceptance rates achieved from speculative decoding over the responses are measured in order to evaluate the quality of the speculation model.

## Inference

**Acceleration Engine:** llama.cpp

**Test Hardware:** NVIDIA RTX Spark & DGX Spark (GB10)

## LiLiCorr Speculative Decoding

This release provides the LiLiCorr draft for the compatible [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) target.
LiLiCorr proposes token blocks that the target verifies during generation.

This model supports a speculative decoding path with LiLiCorr for RTX Spark & DGX Spark and low-concurrency data center workflows. This release is the LiLiCorr speculative decoding checkpoint intended to be paired with the [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) target model. LiLiCorr is designed to improve accepted length while preserving the latency benefits of block drafting on compact Blackwell systems such as RTX Spark & DGX Spark.

## Usage

For serving with [llama.cpp](https://github.com/ggml-org/llama.cpp), see the [compatible NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF target card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF).

## Evaluation

Acceptance length on [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) with draft length 7:

| Category | SPEED-Bench Acceptance Length |
| --- | ---: |
| coding | **5.26** |
| humanities | **3.54** |
| math | **4.59** |
| multilingual | **5.06** |
| qa | **3.91** |
| rag | **5.13** |
| reasoning | **4.47** |
| roleplay | **3.45** |
| stem | **3.73** |
| summarization | **4.80** |
| writing | **3.41** |
| **Overall Average** | **4.30** |

> Baseline: [Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
> Benchmarked with `temperature=1.0`, `top_p=0.95`.

These are checkpoint-level acceptance results and are not attributed to a specific serving runtime.

## Model Limitations

This checkpoint is not a standalone language model and cannot produce user-visible output without the compatible 24GB target.
The draft checkpoint provides no independent safety controls; retain the target model's safeguards and application guardrails.

The target model may generate inaccurate, incomplete, or otherwise undesirable responses, even when prompts are benign. It may reflect biases or content artifacts present in its training data, and output quality can vary by domain, language, reasoning depth, and context length. Developers should add application-specific safeguards and evaluate the system in the intended deployment environment before production use.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, see the Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards.

Please report model quality, risk, security vulnerabilities, or NVIDIA AI concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

SUBCARDS:

# **Explainability**

| Field | Response |
| --- | --- |
| Intended Task/Domain | Text generation from text, image, and video inputs; reasoning; tool use; and agentic workflows accelerated with speculative decoding |
| Model Type | Hybrid LatentMoE vision-language model used with a LiLiCorr draft checkpoint |
| Intended Users | Developers who use the compatible 24GB target for reasoning, chat, retrieval, visual understanding, and agent workflows |
| Output | Draft token candidates; the paired target model produces user-visible text |
| Describe how the model works | The target uses Mamba-2, MoE, and attention layers. LiLiCorr proposes token blocks. The target verifies each block. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of | Not Applicable |
| Technical Limitations & Mitigation | With correct target-model verification, speculative decoding preserves the target model's output distribution. Acceptance length and latency depend on the workload, sampling settings, hardware, and serving configuration. |
| Verified to have met prescribed quality standards? | Acceptance was evaluated on SPEED-Bench. Deployment quality requires separate evaluation. |
| Performance Metrics | Accuracy, throughput, latency, and speculative-decoding acceptance length |
| Potential Known Risk | The target model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. Therefore, before deploying any applications of this model, developers should perform safety testing and tuning tailored to their specific applications of the model. Developers should validate task accuracy, latency, and safety in their own deployment environment and add application-specific safeguards before production use. |
| Licensing | **Governing Terms:** Use of this model is governed by the [OpenMDW-1.1 model license](https://openmdw.ai/license/1-1/) |

# **Bias**

| Field | Response |
| --- | --- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing | None |
| Measures taken to mitigate against unwanted bias | None |
| Bias Metric | None |

# **Safety & Security**

| Field | Response |
| --- | --- |
| Model Application(s) | Chat, instruction following, RAG, reasoning, visual understanding, and agentic AI workflows |
| Describe life critical application (if present) | Not Applicable |
| Use Case Restrictions | Use must comply with the [OpenMDW-1.1 model license](https://openmdw.ai/license/1-1/) and applicable laws and regulations |
| Model and Dataset Restrictions | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |

# **Privacy**

| Field | Response |
| --- | --- |
| Generatable or Reverse engineerable personal data? | No |
| Personal data used to create this model? | No |
| Was consent obtained for any personal data used? | Not Applicable |
| How often is dataset reviewed? | Before Release |
| Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? | No |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | Not Applicable |
| Applicable NVIDIA Privacy Policy | [https://www.nvidia.com/en-us/about-nvidia/privacy-policy/](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |
