# Fine-Tune Qwen3.5-VL

## Introduction

[Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) is the latest vision-language model in the Qwen series developed by Alibaba. It's a 397B-parameter (17B active) hybrid MoE model that uses a repeated layout of Gated DeltaNet and Gated Attention blocks, each paired with sparse MoE (512 experts; 10 routed + 1 shared active). Qwen3.5 is a major upgrade that unifies vision+language, boosts efficiency and multilingual coverage, delivering higher performance at lower latency/cost for developers and enterprises. Qwen3.5-397B-A17B shows competitive benchmark performance across knowledge, reasoning, coding, and agent tasks.
<p align="center">
  <img src="qwen3_5scores.png" alt="Qwen3.5 benchmark" width="500">
</p>

This guide walks you through fine-tuning Qwen3.5 on a medical Visual Question Answering task using NVIDIA NeMo Automodel. You will learn how to prepare the dataset, launch training on a Slurm cluster, and inspect the results.

To set up your environment to run NeMo Automodel, follow the [installation guide](https://github.com/NVIDIA-NeMo/Automodel#-install-nemo-automodel).

## Data

### MedPix-VQA Dataset

We use the [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) dataset, a comprehensive medical Visual Question Answering dataset containing radiological images paired with question-answer pairs for medical image interpretation.

- **20,500 total examples** (85% train / 15% validation)
- **Columns**: `image_id`, `mode`, `case_id`, `question`, `answer`

For a full walkthrough of how MedPix-VQA is preprocessed and integrated into NeMo Automodel—including the chat-template conversion and collate functions—see the [Multi-Modal Dataset Guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/vlm/dataset.md#multi-modal-datasets).

## Launch Training

We provide a ready-to-use recipe at [`examples/vlm_finetune/qwen3_5_moe/qwen3_5_moe_medpix.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/qwen3_5_moe/qwen3_5_moe_medpix.yaml). This recipe is configured to run on 32 x 8 H100 nodes.

NeMo Automodel supports several ways to launch training—via the Automodel CLI with Slurm, interactive sessions, `torchrun`, and more. For full details on all launch options (Slurm batch jobs, multi-node configuration, environment variables, etc.), see the [Run on a Cluster](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/launcher/cluster.md) guide.

### Standalone Slurm Script

We also provide a standalone Slurm script example for Qwen3.5. Before running it, ensure your cluster environment is configured following the [Standalone Slurm Script (Advanced)](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/launcher/cluster.md#standalone-slurm-script-advanced) guide. Then submit the job with the following command:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HOME=your/path/to/hf_cache
export HF_DATASETS_OFFLINE=1
export WANDB_API_KEY=your_wandb_key

srun --output=output.out \
     --error=output.err \
     --container-image /your/path/to/automodel26.02.image.sqsh --no-container-mount-home bash -c "
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 examples/vlm_finetune/finetune.py \
  -c examples/vlm_finetune/qwen3_5_moe/qwen3_5_moe_medpix.yaml \
  --model.pretrained_model_name_or_path=/your/local/qwen3.5weights \
  --processor.pretrained_model_name_or_path=/your/local/qwen3.5weights "
```

**Before you start**:
- Hugging Face applies rate limits on downloads. We recommend cloning the model repository to your local filesystem beforehand.
- Ensure your Hugging Face cache (`HF_HOME`) is configured and that the dataset is already cached locally.
- To enable Weights & Biases logging, set your `WANDB_API_KEY` and configure the `wandb` section in the YAML file.

## Training Results

The training loss curves for Qwen3.5-VL fine-tuned on MedPix-VQA are shown below.

<p align="center">
  <img src="qwen3_5.png" alt="Qwen3.5-VL Training Loss Curve" width="500">
</p>

## Deploy the Fine-Tuned Model

After fine-tuning your Qwen3.5-VL model, you can deploy it for production use or share it with the community. This section covers loading checkpoints for inference, deploying with vLLM, and publishing to the Hugging Face Hub.

### Load the Checkpoint for Inference

The inference functionality is provided through [`examples/vlm_generate/generate.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_generate/generate.py), which supports loading fine-tuned checkpoints and performing image-text generation.

#### Basic Usage

```bash
python examples/vlm_generate/generate.py \
    --checkpoint-path /path/to/checkpoint/epoch_X_step_Y \
    --base-model-path Qwen/Qwen3.5-397B-A17B \
    --prompt "What medical condition is shown in this image?" \
    --image /path/to/medical_image.jpg \
    --max-new-tokens 100
```

The script supports loading from different checkpoint formats:
- **Consolidated safetensors checkpoint**: Load directly from `checkpoint/model/consolidated`
- **Distributed checkpoint**: Load from the checkpoint directory and specify the base model

Example output:

```
The image shows a subdural hematoma in the left hemisphere, visible as a hyperdense crescentic collection along the inner surface of the skull.
```

#### Output Options

You can customize the output format and destination:

```bash
# JSON output
python examples/vlm_generate/generate.py \
    --checkpoint-path /path/to/checkpoint \
    --base-model-path Qwen/Qwen3.5-397B-A17B \
    --prompt "Describe this medical finding." \
    --image medical_scan.jpg \
    --output-format json

# Write to file
python examples/vlm_generate/generate.py \
    --checkpoint-path /path/to/checkpoint \
    --base-model-path Qwen/Qwen3.5-397B-A17B \
    --prompt "Describe this medical finding." \
    --image medical_scan.jpg \
    --output-file results.txt
```

### Deploy with vLLM

[vLLM](https://github.com/vllm-project/vllm) is an efficient inference engine for large language models. While vLLM primarily focuses on text-only models, support for vision-language models is expanding.

:::{note}
vLLM support for vision-language models varies by model architecture. Check the [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for current VLM support status.
:::

If your Qwen3.5-VL checkpoint is saved in a consolidated Hugging Face format, you can attempt to use it with vLLM:

```python
from vllm import LLM, SamplingParams

# Load the fine-tuned VLM checkpoint
llm = LLM(
    model="/path/to/checkpoint/epoch_X_step_Y/model/consolidated/",
    model_impl="transformers",
    trust_remote_code=True
)

# Configure generation parameters
params = SamplingParams(max_tokens=100, temperature=0.0)

# Generate response
outputs = llm.generate(
    "What condition is visible in this CT scan?",
    sampling_params=params
)
print(f"Generated text: {outputs[0].outputs[0].text}")
```

For production deployment, consider using vLLM's OpenAI-compatible server:

```bash
vllm serve /path/to/checkpoint/model/consolidated/ \
    --trust-remote-code \
    --max-model-len 4096 \
    --dtype bfloat16
```

### Publish to Hugging Face Hub

After fine-tuning, you can share your model with the community by publishing it to the Hugging Face Hub. This enables easy sharing, reproducibility, and integration with the Hugging Face ecosystem.

#### Prerequisites

1. Install the Hugging Face Hub library:

```bash
pip install huggingface_hub
```

2. Log in to Hugging Face:

```bash
huggingface-cli login
```

#### Upload the Checkpoint

Use the Hugging Face Hub API to upload your fine-tuned checkpoint:

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload consolidated checkpoint
api.upload_folder(
    folder_path="/path/to/checkpoint/epoch_X_step_Y/model/consolidated",
    repo_id="your-username/qwen3.5-vl-medpix-finetuned",
    repo_type="model"
)
```

#### Load from the Hub

Once uploaded, others can load your fine-tuned model directly:

```python
from transformers import AutoProcessor
from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

# Load the fine-tuned model
model = NeMoAutoModelForImageTextToText.from_pretrained(
    "your-username/qwen3.5-vl-medpix-finetuned"
)
processor = AutoProcessor.from_pretrained(
    "your-username/qwen3.5-vl-medpix-finetuned"
)

# Use for inference
# ... (inference code)
```

#### Model Card

When publishing your model, include a model card with:
- Training dataset information (MedPix-VQA)
- Fine-tuning configuration details
- Evaluation metrics
- Intended use cases and limitations
- Example usage code

This helps the community understand your model and use it appropriately.