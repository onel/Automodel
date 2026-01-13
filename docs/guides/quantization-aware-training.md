# Quantization-Aware Training (QAT) for Supervised Fine-Tuning

NeMo Automodel supports Quantization-Aware Training (QAT) for Supervised Fine-Tuning (SFT) using [TorchAO](https://github.com/pytorch/ao). QAT simulates quantization during training, allowing models to adapt to lower precision and maintain better accuracy compared to post-training quantization.

QAT is particularly valuable for deploying models on resource-constrained devices or optimizing inference performance while preserving model quality.

## What is Quantization-Aware Training?

Quantization-Aware Training simulates the effects of quantization during the training process. Unlike post-training quantization (PTQ), which quantizes a trained model, QAT trains the model with quantization in mind from the start. This allows the model to learn to compensate for quantization errors, resulting in better accuracy at lower precision.

### Benefits of QAT

- **Better accuracy**: Models adapt to quantization during training, maintaining higher accuracy than PTQ
- **Efficient deployment**: Quantized models require less memory and compute resources
- **Faster inference**: Lower precision operations execute faster on compatible hardware
- **Edge deployment**: Enables deployment on resource-constrained devices

### QAT vs. Post-Training Quantization

| Aspect | QAT | Post-Training Quantization |
|--------|-----|---------------------------|
| Accuracy | Higher - model adapts during training | Lower - no adaptation |
| Training time | Longer - requires retraining | None - applied after training |
| Use case | When accuracy is critical | Quick deployment, less critical accuracy |

## Requirements

To use QAT in NeMo Automodel, you need:

- **Software**: TorchAO library must be installed
- **Hardware**: Compatible NVIDIA GPU (recommended: A100 or newer)
- **Model**: Any supported model architecture for SFT

## Install TorchAO

TorchAO is included as a dependency in NeMo Automodel. If you need to install it separately, follow the [TorchAO installation guide](https://github.com/pytorch/ao?tab=readme-ov-file#-installation).

## Supported Quantization Schemes

NeMo Automodel supports two QAT quantization schemes through TorchAO:

### 1. Int8 Dynamic Activation + Int4 Weight (8da4w-qat)

- **Quantizer**: `Int8DynActInt4WeightQATQuantizer`
- **Precision**: 8-bit dynamic activations, 4-bit weights
- **Use case**: Balanced accuracy and efficiency
- **Recommended for**: Most fine-tuning scenarios

### 2. Int4 Weight-Only (4w-qat)

- **Quantizer**: `Int4WeightOnlyQATQuantizer`
- **Precision**: 4-bit weights only, full precision activations
- **Use case**: Maximum memory savings with minimal accuracy loss
- **Recommended for**: Memory-constrained deployments

## Configuration

To enable QAT in your training configuration, add a `qat` section to your YAML file:

```yaml
# Enable Quantization-Aware Training
qat:
  enabled: true
  quantizer_type: "8da4w"  # Options: "8da4w" or "4w"
  delay_fake_quant_steps: 0  # Number of steps before enabling fake quantization
```

### QAT Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable QAT for training |
| `quantizer_type` | str | "8da4w" | Quantization scheme: "8da4w" or "4w" |
| `delay_fake_quant_steps` | int | 0 | Delay fake quantization for initial training steps |

### Delayed Fake Quantization

You can delay the activation of fake quantization to allow the model to stabilize during initial training:

```yaml
qat:
  enabled: true
  quantizer_type: "8da4w"
  delay_fake_quant_steps: 1000  # Enable fake quant after 1000 steps
```

This can help with training stability, especially for aggressive quantization schemes.

## Complete Training Example

Here's a complete configuration example for fine-tuning Llama 3.2 1B with QAT:

```yaml
# Model configuration
model:
  model_name: "meta-llama/Llama-3.2-1B"
  trust_remote_code: true

# Dataset configuration
dataset:
  type: "ColumnMappedTextInstructionDataset"
  train:
    path: "squad"
    split: "train"
    column_map:
      input: "question"
      context: "context"
      output: "answers.text"
  validation:
    path: "squad"
    split: "validation[:1000]"
    column_map:
      input: "question"
      context: "context"
      output: "answers.text"

# Training configuration
training:
  max_steps: 5000
  val_check_interval: 500
  gradient_accumulation_steps: 1
  micro_batch_size: 2

# Enable QAT
qat:
  enabled: true
  quantizer_type: "8da4w"
  delay_fake_quant_steps: 0

# Optimizer
optimizer:
  lr: 1e-5
  weight_decay: 0.01

# Checkpointing
checkpoint:
  save_interval: 500
  save_top_k: 3
```

## Training Workflow

### 1. Prepare Your Configuration

Create a YAML configuration file with QAT enabled as shown above.

### 2. Run Training

Use the standard training command:

```bash
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config your_qat_config.yaml
```

### 3. Monitor Training

QAT training will show similar metrics to standard training. Monitor:
- Training loss convergence
- Validation accuracy
- Fake quantization activation (if delayed)

### 4. Export for Deployment

After training, the model checkpoints can be used for inference. The quantization-aware weights will perform better when deployed with actual quantization.

## Performance Considerations

### Training Overhead

- QAT adds computational overhead during training due to fake quantization operations
- Training time may increase by 10-30% compared to full-precision training
- The overhead is worthwhile for the improved quantized model accuracy

### Memory Usage

- QAT training memory usage is similar to full-precision training
- Memory savings are realized during inference with the quantized model

### Accuracy vs. Efficiency Trade-offs

| Quantization Scheme | Accuracy | Memory Savings | Inference Speed |
|---------------------|----------|----------------|----------------|
| Full Precision (BF16) | Baseline | Baseline | Baseline |
| 8da4w QAT | ~98-99% of baseline | ~50% reduction | ~2x faster |
| 4w QAT | ~95-98% of baseline | ~60% reduction | ~2.5x faster |

*Note: Actual results vary by model and task*

## Best Practices

### When to Use QAT

- **Use QAT when**:
  - Deploying to resource-constrained environments
  - Accuracy is critical and PTQ results are insufficient
  - You have time and resources for retraining
  - Target deployment uses quantized inference

- **Consider PTQ when**:
  - Quick deployment is needed
  - Slight accuracy loss is acceptable
  - Training resources are limited

### Choosing Quantization Scheme

- **8da4w**: Best balance for most use cases
- **4w**: Maximum memory savings, use when memory is the primary constraint

### Training Tips

1. **Start with delayed fake quantization**: Use `delay_fake_quant_steps` to stabilize initial training
2. **Monitor validation metrics**: Ensure the model converges properly with quantization
3. **Use appropriate learning rates**: QAT may require slightly different learning rates than full-precision training
4. **Validate on target hardware**: Test the quantized model on your deployment hardware

## Deployment

After QAT training:

1. **Save the checkpoint**: Use standard checkpointing mechanisms
2. **Convert for inference**: Apply actual quantization using TorchAO or your deployment framework
3. **Validate accuracy**: Verify the quantized model meets your accuracy requirements
4. **Deploy**: Use the quantized model for efficient inference

## Limitations

- QAT requires retraining, which takes additional time and compute resources
- Not all model architectures may benefit equally from QAT
- Quantization schemes are limited to those supported by TorchAO
- Deployment infrastructure must support the chosen quantization format

## References

- [TorchAO Documentation](https://github.com/pytorch/ao)
- [TorchAO QAT Guide](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat)
- [Quantization Overview](https://pytorch.org/docs/stable/quantization.html)
