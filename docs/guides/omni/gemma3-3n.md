# Fine-Tune Gemma 3 and Gemma 3n

This document explains how to fine-tune Gemma 3 and Gemma 3n using NeMo Automodel. It outlines key operations, including initiating SFT and PEFT-LoRA runs and managing experiment configurations using YAML.

To set up your environment to run NeMo Automodel, follow the [installation guide](https://github.com/NVIDIA-NeMo/Automodel#-install-nemo-automodel).

## Data

### MedPix-VQA Dataset

The [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) dataset is a comprehensive medical Visual Question-Answering dataset designed for training and evaluating VQA models in the medical domain. It contains medical images from MedPix, a well-known medical image database, paired with questions and answers that focus on medical image interpretation.

The dataset consists of 20,500 examples with the following structure:
- **Training Set**: 17,420 examples (85%)
- **Validation Set**: 3,080 examples (15%)
- **Columns**: `image_id`, `mode`, `case_id`, `question`, `answer`

### Preprocess the Dataset

NeMo Automodel provides built-in preprocessing for the MedPix-VQA dataset through the `make_medpix_vqa_dataset` function. Here's how the preprocessing works:

```python
from nemo_automodel.datasets.vlm.datasets import make_medpix_vqa_dataset

# Load and preprocess the dataset
dataset = make_medpix_vqa_dataset(
    path_or_dataset="mmoukouba/MedPix-VQA",
    split="train"
)
```

The preprocessing pipeline performs the following steps:

1. **Loads the dataset** using the Hugging Face `datasets` library.
2. **Extracts question-answer pairs** by processing the `question` and `answer` fields from the dataset.
3. **Converts to the Hugging Face message list format** to restructure the data into a chat-style format compatible with the Autoprocessor's `apply_chat_template` function.

```python
# Example of the conversation format created
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": example["image_id"]},
            {"type": "text", "text": example["question"]},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": example["answer"]}]
    },
]
```

### Use the Collate Functions

NeMo Automodel provides specialized collate functions for different VLM processors. The collate function is responsible for batching examples and preparing them for model input.

Both Gemma 3 and Gemma 3n models work seamlessly with the Hugging Face `AutoProcessor` and use the default collate function:

```python
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
# For Gemma 3n, get processor:
# processor = AutoProcessor.from_pretrained("google/gemma-3n-e4b-it")

# For Gemma 3 and Gemma 3n, use the default collate function
def default_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )

    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    batch["labels"] = labels
    loss_mask = create_batch_loss_masks(
        batch["input_ids"], processor, start_of_response_token=start_of_response_token
    )
    batch["loss_mask"] = loss_mask

    return batch
```

The default collate function:
- Applies the processor's chat template to convert message lists into model-ready inputs.
- Creates labels for training to guide supervised learning.
- Masks prompts and special tokens so that only answer tokens are considered during loss calculation.

### Preprocess Custom Datasets

When using a custom dataset with a model whose Hugging Face `AutoProcessor` supports the `apply_chat_template` method, you'll need to convert your data into the Hugging Face message list format expected by the `apply_chat_template`.
We provide [examples](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/datasets/vlm/datasets.py) demonstrating how to perform this conversion.

Some models, such as [Qwen2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), have specific preprocessing requirements and require custom collate functions. For instance, Qwen2.5-VL uses the `qwen_vl_utils.process_vision_info` function to process images:

```python

texts = [processor.apply_chat_template(example["conversation"], tokenize=False) for example in examples]
image_inputs = [process_vision_info(example["conversation"])[0] for example in examples]

batch = processor(
    text=texts,
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)

```
If your dataset requires custom preprocessing logic, you can define a custom collate function. To use it, specify the function in your YAML configuration:

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  collate_fn:
    _target_: nemo_automodel.datasets.vlm.collate_fns.qwen2_5_collate_fn
```

We provide [example custom collate functions](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/datasets/vlm/collate_fns.py) that you can use as references for your implementation.

## Run the Fine-Tune Script

The VLM fine-tuning functionality is provided through [`examples/vlm_finetune/finetune.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/finetune.py).

### Apply YAML-Based Configuration

NeMo Automodel uses a flexible configuration system that combines YAML configuration files with command-line overrides. This allows you to maintain base configurations while easily experimenting with different parameters.

The simplest way to run fine-tuning is with a YAML configuration file. We provide configs for both Gemma 3 and Gemma 3n.

#### Run Gemma 3 Fine-Tuning

* **Single GPU**

```bash
uv run examples/vlm_finetune/finetune.py --config examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix.yaml
```

* **Multi GPU**

```bash
uv run torchrun --nproc-per-node=2 examples/vlm_finetune/finetune.py \
    --config examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix.yaml
```

#### Run Gemma 3n Fine-Tuning

* **Single GPU**

```bash
uv run examples/vlm_finetune/finetune.py --config examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix.yaml
```

* **Multi-GPU**

```bash
uv run torchrun --nproc-per-node=2 examples/vlm_finetune/finetune.py \
    --config examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix.yaml
```

#### Override Configuration Parameters

You can override any configuration parameter using dot-notation without modifying the YAML file:

```bash
uv run examples/vlm_finetune/finetune.py \
    --config examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix.yaml \
    --step_scheduler.ckpt_every_steps 100 \
    --step_scheduler.max_steps 1000 \
    --optimizer.lr 2e-5 \
    --rng.seed 1234
```

### Configure Model Freezing

NeMo Automodel supports parameter freezing, allowing you to control which parts of a model remain trainable during fine-tuning. This is especially useful for VLMs, where you may want to preserve the pre-trained visual and audio encoders while adapting only the language model components.

With the freezing configuration, you can selectively freeze specific parts of the model to suit your training objectives:

```yaml
freeze_config:
  freeze_embeddings: true        # Freeze embeddings
  freeze_vision_tower: true      # Freeze vision encoder (recommended for VLMs)
  freeze_audio_tower: true       # Freeze audio encoder (for multimodal models)
  freeze_language_model: false   # Allow language model adaptation
```

### Run Parameter Efficient Fine-Tuning

For memory-efficient training, you can use Low-Rank Adaptation (LoRA) instead of full fine-tuning. NeMo Automodel provides a dedicated PEFT recipe for Gemma 3:

To run PEFT with Gemma 3:

```bash
uv run examples/vlm_finetune/finetune.py --config examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix_peft.yaml
```

The LoRA configuration excludes vision and audio components from adaptation to preserve pre-trained visual representations:

```yaml
peft:
  peft_fn: nemo_automodel._peft.lora.apply_lora_to_linear_modules
  match_all_linear: False
  exclude_modules:  # exclude all vision and audio modules and lm_head
    - "*vision_tower*"
    - "*vision*"
    - "*visual*"
    - "*audio*"
    - "*image_encoder*"
    - "*lm_head*"
  dim: 8
  alpha: 32
  use_triton: True
```

The training loss should look similar to the example below:

```{image} medpix_peft.jpg
:alt: Training Loss Curve
:class: bg-primary
:width: 400px
:align: center
```

### Checkpointing

We support training state checkpointing in either [Safetensors](https://huggingface.co/docs/safetensors/en/index) or [PyTorch DCP](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) format.

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: vlm_checkpoints/
  model_save_format: torch_save  # or "safetensors"
  save_consolidated: false
```

#### Integrate Weights & Biases

You can enable W&B logging by setting your API key and configuring the logger:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

Then, add the W&B configuration to your YAML file:

```yaml
wandb:
  project: nemo_automodel_vlm
  entity: your_entity
  name: gemma3_medpix_vqa_experiment
  save_dir: ./wandb_logs
```

## Deploy the Fine-Tuned Model

After fine-tuning your Gemma 3 or Gemma 3n model, you can deploy it for production use or share it with the community. This section covers loading checkpoints for inference, deploying with vLLM, and publishing to the Hugging Face Hub.

### Load the Checkpoint for Inference

The inference functionality is provided through [`examples/vlm_generate/generate.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_generate/generate.py), which supports loading fine-tuned checkpoints and performing image-text generation.

#### Basic Usage

```bash
python examples/vlm_generate/generate.py \
    --checkpoint-path vlm_checkpoints/epoch_0_step_200 \
    --base-model-path google/gemma-3-4b-it \
    --prompt "What medical condition is shown in this image?" \
    --image medical_image.jpg
```

The output can be either `text` (default) or `json`, with an optional write file.

For models trained on MedPix-VQA, load the trained checkpoint and generate outputs using the following command. Be sure to specify the same base model used during training:

```bash
python examples/vlm_generate/generate.py \
    --checkpoint-path vlm_checkpoints/epoch_0_step_200 \
    --base-model-path google/gemma-3-4b-it \
    --prompt "What medical condition is shown in this image?" \
    --image medical_image.jpg \
    --max-new-tokens 100
```

#### PEFT Checkpoint Inference

When checkpoints are saved from PEFT training, they contain only the adapter weights. To use them for generation, you need to specify the base model:

```bash
python examples/vlm_generate/generate.py \
    --checkpoint-path peft_vlm_checkpoints/epoch_0_step_200/ \
    --base-model-path google/gemma-3-4b-it \
    --prompt "What medical condition is shown in this image?" \
    --image medical_image.jpg
```

#### Example Output

Given the following image:

```{image} medpix.jpg
:alt: Sample image from the MedPix dataset
:class: bg-primary
:width: 200px
:align: center
```

And the prompt:

```
How does the interhemispheric fissure appear in this image?
```

Example Gemma 3 response:

```
The interhemispheric fissure appears as a dark streak, indicating significant tissue loss.
```

Example Gemma 3n response:

```
The interhemispheric fissure appears somewhat obscured by the fluid-filled mass.
```

#### Output Formats

Customize the output format and destination:

```bash
# JSON output
python examples/vlm_generate/generate.py \
    --checkpoint-path vlm_checkpoints/epoch_0_step_200 \
    --base-model-path google/gemma-3-4b-it \
    --prompt "Describe this medical finding." \
    --image medical_scan.jpg \
    --output-format json

# Write to file
python examples/vlm_generate/generate.py \
    --checkpoint-path vlm_checkpoints/epoch_0_step_200 \
    --base-model-path google/gemma-3-4b-it \
    --prompt "Describe this medical finding." \
    --image medical_scan.jpg \
    --output-file results.txt
```

### Deploy with vLLM

[vLLM](https://github.com/vllm-project/vllm) is an efficient inference engine for large language models. For production deployment with optimized inference, you can use vLLM with your fine-tuned checkpoints.

:::{note}
vLLM support for vision-language models varies by model architecture. Check the [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for current VLM support status.
:::

If your Gemma 3 or Gemma 3n checkpoint is saved in a consolidated Hugging Face format, you can use it with vLLM:

```python
from vllm import LLM, SamplingParams

# Load the fine-tuned VLM checkpoint
llm = LLM(
    model="vlm_checkpoints/epoch_0_step_200/model/consolidated/",
    model_impl="transformers",
    trust_remote_code=True
)

# Configure generation parameters
params = SamplingParams(max_tokens=100, temperature=0.0)

# Generate response
outputs = llm.generate(
    "What condition is visible in this medical image?",
    sampling_params=params
)
print(f"Generated text: {outputs[0].outputs[0].text}")
```

For production deployment, consider using vLLM's OpenAI-compatible server:

```bash
vllm serve vlm_checkpoints/epoch_0_step_200/model/consolidated/ \
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

For SFT checkpoints, upload the consolidated model directory:

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload consolidated checkpoint
api.upload_folder(
    folder_path="vlm_checkpoints/epoch_0_step_200/model/consolidated",
    repo_id="your-username/gemma-3-4b-medpix-finetuned",
    repo_type="model"
)
```

For PEFT adapters, upload the adapter directory:

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload PEFT adapter
api.upload_folder(
    folder_path="peft_vlm_checkpoints/epoch_0_step_200/model",
    repo_id="your-username/gemma-3-4b-medpix-lora",
    repo_type="model"
)
```

#### Load from the Hub

Once uploaded, others can load your fine-tuned model directly:

```python
from transformers import AutoProcessor
from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

# Load SFT checkpoint
model = NeMoAutoModelForImageTextToText.from_pretrained(
    "your-username/gemma-3-4b-medpix-finetuned"
)
processor = AutoProcessor.from_pretrained(
    "your-username/gemma-3-4b-medpix-finetuned"
)
```

For PEFT adapters:

```python
from transformers import AutoProcessor
from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
from peft import PeftModel

# Load base model
base_model = NeMoAutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it")
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

# Load adapter
model = PeftModel.from_pretrained(base_model, "your-username/gemma-3-4b-medpix-lora")
```

#### Model Card

When publishing your model, include a model card with:
- Training dataset information (MedPix-VQA)
- Fine-tuning configuration details
- Evaluation metrics and results
- Intended use cases and limitations
- Example usage code
- Hardware requirements

This helps the community understand your model and use it appropriately.