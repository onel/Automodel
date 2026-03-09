# LLM Fine-Tuning Examples

This directory contains ready-to-run fine-tuning configurations for large language models across multiple model families and tasks. Each configuration demonstrates how to fine-tune models using NeMo Automodel with Hugging Face-compatible checkpoints.

## Available Model Configurations

NeMo Automodel supports fine-tuning for the following model families:

| Model Family | Example Configurations | Notable Features |
|--------------|----------------------|------------------|
| **Llama 3.1/3.2/3.3** | SQuAD, HellaSwag, PEFT, HSDP, QAT | Multiple sizes (1B, 3B, 70B+) |
| **Gemma 2/3** | SQuAD, FP8, PEFT | Google's efficient models |
| **Mistral/Mixtral** | SQuAD, HellaSwag, FP8, PEFT | Base and MoE variants |
| **Qwen** | SQuAD, HellaSwag, PEFT | Alibaba's multilingual models |
| **Nemotron** | SQuAD, PEFT | NVIDIA's optimized models |
| **Falcon 3** | SQuAD, PEFT | TII's large-scale models |
| **Granite** | SQuAD, PEFT | IBM's enterprise models |
| **Phi** | SQuAD, PEFT | Microsoft's small language models |
| **DeepSeek V3.2** | SQuAD, PEFT | MoE architecture |
| **GLM** | SQuAD, PEFT | Bilingual models |
| **Starcoder** | SQuAD, PEFT | Code generation models |

For a complete list of supported models, see the [model coverage documentation](../../docs/model-coverage/llm.md).

## Quick Start

### Run a Basic Fine-Tuning Job

Fine-tune Llama 3.2 1B on SQuAD using a single GPU:

```bash
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

### Run Multi-GPU Fine-Tuning

Fine-tune on multiple GPUs (single node):

```bash
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node=4
```

### Run PEFT Fine-Tuning

Fine-tune using parameter-efficient methods (LoRA):

```bash
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml
```

## Configuration Examples

### Full Fine-Tuning

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

checkpoint:
  model_save_format: safetensors
  save_consolidated: true  # Enables deployment-ready checkpoints

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train
```

### PEFT Fine-Tuning with LoRA

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true
  dim: 16
  alpha: 32
  use_triton: true
```

### FP8 Training

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: mistralai/Mistral-7B-v0.1

quantization:
  enabled: true
  precision: fp8
```

For detailed configuration options, see the [fine-tuning guide](../../docs/guides/llm/finetune.md).

## Training Options

### Datasets

Examples demonstrate fine-tuning on popular datasets:

- **SQuAD**: Question answering
- **HellaSwag**: Commonsense reasoning
- **Custom datasets**: See [dataset integration guide](../../docs/guides/llm/dataset.md)

### Distributed Training

- **FSDP2**: Default distributed strategy
- **HSDP**: Hybrid sharding for multi-node setups
- **Megatron FSDP**: Advanced parallelism for large models
- **Pipeline Parallelism**: For models that don't fit on a single GPU

See examples with `_hsdp.yaml` or `_megatron_fsdp.yaml` suffixes.

### Optimization Techniques

- **PEFT (LoRA/QLoRA)**: Train only a small subset of parameters
- **FP8 Training**: Reduce memory and increase throughput
- **QAT**: Quantization-aware training for efficient deployment
- **Gradient Checkpointing**: Trade compute for memory

## Deploying Fine-Tuned Models

After fine-tuning, NeMo Automodel produces Hugging Face-compatible checkpoints that can be deployed using multiple inference frameworks.

### Checkpoint Location

Checkpoints are saved to the `checkpoint_dir` specified in your configuration:

```
checkpoints/
├── LATEST -> epoch_0_step_1000
├── LOWEST_VAL -> epoch_0_step_1000
└── epoch_0_step_1000/
    └── model/
        └── consolidated/  # Deployment-ready checkpoint
```

### Deployment with Hugging Face Transformers

Load and run inference locally:

```python
from transformers import pipeline
import torch

model_path = "checkpoints/LATEST/model/consolidated/"
pipe = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

result = pipe("Answer this question: What is machine learning?")
print(result[0]["generated_text"])
```

### Deployment with vLLM

Deploy for high-throughput production serving:

```bash
# Start vLLM server
vllm serve checkpoints/LATEST/model/consolidated/ \
    --dtype bfloat16 \
    --max-model-len 4096
```

Query the server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

response = client.chat.completions.create(
    model="checkpoints/LATEST/model/consolidated/",
    messages=[{"role": "user", "content": "What is supervised fine-tuning?"}],
)

print(response.choices[0].message.content)
```

### Deployment with SGLang

Deploy using SGLang for efficient structured generation:

```bash
python -m sglang.launch_server \
    --model-path checkpoints/LATEST/model/consolidated/ \
    --dtype bfloat16
```

### PEFT Adapter Deployment

For PEFT-trained models, deploy the lightweight adapters:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

checkpoint_path = "checkpoints/LATEST/model/"
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model = model.to("cuda")
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Or merge adapters for deployment without PEFT overhead:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("checkpoints/LATEST/model/")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model/")

# Deploy merged_model/ using any standard method
```

For comprehensive deployment guidance, including cloud deployment, quantization, and production best practices, see the [deployment guide](../../docs/guides/deployment.md).

## Example Configurations by Model Family

### Llama 3.2

- `llama3_2/llama3_2_1b_squad.yaml` - Basic fine-tuning on SQuAD
- `llama3_2/llama3_2_1b_squad_peft.yaml` - LoRA fine-tuning
- `llama3_2/llama3_2_1b_hellaswag.yaml` - Commonsense reasoning
- `llama3_2/llama3_2_1b_squad_qat.yaml` - Quantization-aware training
- `llama3_2/llama3_2_1b_hellaswag_hsdp.yaml` - Hybrid sharding

### Gemma

- `gemma/gemma_2_9b_it_squad.yaml` - Instruction-tuned variant
- `gemma/gemma_2_9b_it_squad_peft.yaml` - PEFT on instruction model
- `gemma/gemma_2_9b_it_hellaswag_fp8.yaml` - FP8 training
- `gemma/gemma_3_270m_squad.yaml` - Compact model fine-tuning

### Mistral/Mixtral

- `mistral/mistral_7b_squad.yaml` - Base Mistral fine-tuning
- `mistral/mistral_7b_squad_peft.yaml` - PEFT fine-tuning
- `mistral/mixtral-8x7b-v0-1_squad.yaml` - MoE model fine-tuning
- `mistral/mistral_nemo_2407_squad.yaml` - Latest Mistral variant

### Qwen

- `qwen/qwen_0.6b_squad.yaml` - Compact model
- `qwen/qwen_7b_squad.yaml` - Standard fine-tuning
- `qwen/qwen_7b_squad_peft.yaml` - PEFT fine-tuning

## Advanced Features

### Customizing Configurations

Override configuration values from the command line:

```bash
automodel finetune llm \
    -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --model.pretrained_model_name_or_path Qwen/Qwen2.5-1.5B \
    --step_scheduler.num_epochs 3 \
    --optimizer.lr 2.0e-5
```

### Multi-Node Training

For cluster deployment using Slurm:

```bash
automodel finetune llm \
    -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --slurm.nodes 4 \
    --slurm.ntasks_per_node 8
```

See the [cluster guide](../../docs/launcher/cluster.md) for details.

### Experiment Tracking

Enable Weights & Biases logging:

```yaml
wandb:
  project: my-finetune-project
  entity: my-team
  name: llama3-2-1b-squad-run1
```

Enable MLflow logging:

```yaml
mlflow:
  experiment_name: llm-finetune
  tracking_uri: http://localhost:5000
```

## Best Practices

1. **Start small**: Test configurations on a single GPU before scaling
2. **Use PEFT for exploration**: LoRA enables fast iteration with minimal resources
3. **Enable consolidated checkpoints**: Set `save_consolidated: true` for deployment
4. **Monitor validation loss**: Configure `val_every_steps` to track overfitting
5. **Leverage FP8**: Use FP8 training for larger models to reduce memory usage
6. **Version your experiments**: Use meaningful checkpoint directory names

## Troubleshooting

### Out of Memory

- Reduce `local_batch_size` or `global_batch_size`
- Enable gradient checkpointing
- Use PEFT instead of full fine-tuning
- Switch to FP8 training
- Increase the number of GPUs

### Slow Training

- Increase `local_batch_size` if memory permits
- Disable `torch.compile` if it's causing slowdowns
- Check that CUDA is properly configured
- Ensure distributed strategy matches your hardware

### Checkpoint Loading Errors

- Verify model ID is accessible from Hugging Face Hub
- Check Hugging Face authentication for gated models
- Ensure sufficient disk space for checkpoint downloads

## See Also

- [Fine-Tuning Guide](../../docs/guides/llm/finetune.md) - Detailed fine-tuning instructions
- [Deployment Guide](../../docs/guides/deployment.md) - Comprehensive deployment options
- [Checkpointing Guide](../../docs/guides/checkpointing.md) - Checkpoint formats and management
- [Dataset Integration](../../docs/guides/llm/dataset.md) - Using custom datasets
- [Configuration Guide](../../docs/guides/configuration.md) - Configuration system details
