# Deploying Models Trained with NeMo Automodel

NeMo Automodel produces Hugging Face-compatible checkpoints that can be deployed using industry-standard inference tools. This guide covers deployment options for models trained with NeMo Automodel, from local testing to production serving at scale.

## Checkpoint Format Overview

NeMo Automodel saves checkpoints in two formats optimized for different use cases:

- **Consolidated checkpoints**: Complete model saved as Hugging Face-compatible bundle (recommended for deployment)
- **Sharded checkpoints**: Model weights split across files for distributed training

For deployment, use consolidated checkpoints with `save_consolidated: true` in your training configuration. See the [checkpointing guide](checkpointing.md) for details on checkpoint formats.

### Recommended Checkpoint Configuration

```yaml
checkpoint:
  model_save_format: safetensors
  save_consolidated: true
```

This produces a `consolidated/` directory containing your model in Hugging Face-compatible format:

```
checkpoints/epoch_0_step_1000/model/consolidated/
├── config.json
├── generation_config.json
├── model-00001-of-00001.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

## Deployment Options

### Hugging Face Transformers

The simplest deployment option is using Hugging Face Transformers directly. Load your consolidated checkpoint for local inference or testing.

#### Basic Inference

```python
import torch
from transformers import pipeline

model_path = "checkpoints/epoch_0_step_1000/model/consolidated/"
pipe = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

result = pipe("The key to successful deployment is")
print(result[0]["generated_text"])
```

#### Lower-Level API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "checkpoints/epoch_0_step_1000/model/consolidated/"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = tokenizer("The key to successful deployment is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### vLLM

vLLM is a high-throughput inference engine optimized for serving large language models at scale. It provides:

- State-of-the-art serving throughput with PagedAttention
- Continuous batching for efficient request handling
- OpenAI-compatible API server
- Support for quantization (FP8, INT4, INT8)
- Multi-LoRA batching

#### Installation

```bash
pip install vllm
```

#### Launch Inference Server

```bash
vllm serve checkpoints/epoch_0_step_1000/model/consolidated/ \
    --dtype bfloat16 \
    --max-model-len 4096
```

This starts an OpenAI-compatible API server on `http://localhost:8000`.

#### Query the Server

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="checkpoints/epoch_0_step_1000/model/consolidated/",
    messages=[
        {"role": "user", "content": "Explain model deployment in one sentence."}
    ],
)

print(completion.choices[0].message.content)
```

#### Offline Batch Inference

For batch processing without a server:

```python
from vllm import LLM, SamplingParams

model_path = "checkpoints/epoch_0_step_1000/model/consolidated/"
llm = LLM(model=model_path, dtype="bfloat16")

prompts = [
    "Explain deployment.",
    "What is inference?",
    "How do I serve models?",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

### SGLang

SGLang is a high-performance serving framework designed for low-latency and high-throughput inference. It excels at:

- RadixAttention for efficient prefix caching
- Structured outputs with constrained generation
- Prefill-decode disaggregation for large-scale deployments
- Support for language models, vision-language models, and diffusion models

#### Installation

```bash
pip install "sglang[all]"
```

#### Launch Inference Server

```bash
python -m sglang.launch_server \
    --model-path checkpoints/epoch_0_step_1000/model/consolidated/ \
    --dtype bfloat16 \
    --port 30000
```

#### Query the Server

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Explain model deployment in one sentence.",
        "sampling_params": {
            "temperature": 0.8,
            "max_new_tokens": 100,
        },
    },
)

print(response.json()["text"])
```

#### Offline Inference

```python
import sglang as sgl

@sgl.function
def deployment_qa(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))

model_path = "checkpoints/epoch_0_step_1000/model/consolidated/"
runtime = sgl.Runtime(model_path=model_path)
sgl.set_default_backend(runtime)

state = deployment_qa.run(question="What is the best way to deploy LLMs?")
print(state["answer"])

runtime.shutdown()
```

## Deploying PEFT Adapters

When using parameter-efficient fine-tuning (PEFT), only adapter weights are saved, dramatically reducing checkpoint size. NeMo Automodel produces PEFT checkpoints compatible with the Hugging Face PEFT library.

### Load PEFT Model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

checkpoint_path = "checkpoints/epoch_0_step_1000/model/"
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model = model.to("cuda")
model.eval()

inputs = tokenizer("Explain PEFT deployment.", return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

### Merge Adapters for Production

For production deployment, merge adapters into the base model to eliminate overhead:

```python
from peft import AutoPeftModelForCausalLM

checkpoint_path = "checkpoints/epoch_0_step_1000/model/"
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)

# Merge adapters into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("merged_model/")
```

Then deploy `merged_model/` using any of the methods above (Transformers, vLLM, SGLang).

## Performance Optimization

### Quantization

Reduce memory footprint and increase throughput with quantization:

#### vLLM with FP8

```bash
vllm serve checkpoints/epoch_0_step_1000/model/consolidated/ \
    --quantization fp8 \
    --dtype bfloat16
```

#### vLLM with INT4/AWQ

```bash
# Requires pre-quantized model
vllm serve quantized_model_path/ \
    --quantization awq \
    --dtype half
```

### Batching

Both vLLM and SGLang support continuous batching, which automatically groups incoming requests for better GPU utilization. Configure batch size limits based on your hardware:

```bash
# vLLM
vllm serve model_path/ \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256

# SGLang
python -m sglang.launch_server \
    --model-path model_path/ \
    --max-running-requests 256
```

### Multi-GPU Deployment

#### vLLM Tensor Parallelism

```bash
vllm serve model_path/ \
    --tensor-parallel-size 4 \
    --dtype bfloat16
```

#### SGLang Data Parallelism

```bash
python -m sglang.launch_server \
    --model-path model_path/ \
    --dp-size 4
```

## Cloud Deployment

### Docker Container

Create a `Dockerfile` for your deployment:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip install vllm

COPY checkpoints/epoch_0_step_1000/model/consolidated/ /model/

EXPOSE 8000

CMD ["vllm", "serve", "/model/", "--dtype", "bfloat16", "--host", "0.0.0.0"]
```

Build and run:

```bash
docker build -t my-model:latest .
docker run --gpus all -p 8000:8000 my-model:latest
```

### Kubernetes

Deploy using a Kubernetes manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm
  template:
    metadata:
      labels:
        app: llm
    spec:
      containers:
      - name: vllm
        image: my-model:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Cloud Platforms

#### AWS SageMaker

Deploy using the SageMaker SDK:

```python
from sagemaker.huggingface import HuggingFaceModel

# Upload model to S3 first
model_path = "s3://my-bucket/models/my-model/"

huggingface_model = HuggingFaceModel(
    model_data=model_path,
    role="SageMakerRole",
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
)
```

#### Google Cloud Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

model = aiplatform.Model.upload(
    display_name="my-llm",
    artifact_uri="gs://my-bucket/models/my-model/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest",
)

endpoint = model.deploy(
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

#### Azure Machine Learning

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment

ml_client = MLClient.from_config()

model = Model(
    path="checkpoints/epoch_0_step_1000/model/consolidated/",
    name="my-llm",
    description="Fine-tuned LLM",
)
registered_model = ml_client.models.create_or_update(model)

endpoint = ManagedOnlineEndpoint(
    name="llm-endpoint",
    description="LLM inference endpoint",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="llm-deployment",
    endpoint_name="llm-endpoint",
    model=registered_model,
    instance_type="Standard_NC6s_v3",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
```

## Monitoring and Observability

### Metrics

Both vLLM and SGLang expose Prometheus metrics for monitoring:

- Request throughput and latency
- Token generation rate
- GPU utilization
- Queue depth
- Error rates

#### vLLM Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

#### SGLang Metrics

SGLang provides built-in monitoring through its runtime API.

### Logging

Configure structured logging for production deployments:

```python
# vLLM with custom logging
vllm serve model_path/ \
    --disable-log-requests false \
    --log-level info
```

## Best Practices

1. **Use consolidated checkpoints**: Set `save_consolidated: true` during training for deployment-ready checkpoints
2. **Enable bfloat16**: Use `torch_dtype=torch.bfloat16` for better performance with minimal accuracy loss
3. **Tune batch size**: Adjust `max-num-batched-tokens` based on GPU memory and latency requirements
4. **Monitor resource usage**: Track GPU memory, throughput, and latency in production
5. **Test locally first**: Validate deployments with Transformers before scaling to vLLM or SGLang
6. **Version your models**: Use clear checkpoint naming and versioning for rollback capability
7. **Implement health checks**: Add readiness and liveness probes for container deployments

## Troubleshooting

### Out of Memory Errors

- Reduce `max-model-len` or `max-num-batched-tokens`
- Enable quantization (FP8, INT4)
- Use tensor parallelism to split model across GPUs

### Slow Inference

- Increase batch size for better throughput
- Enable continuous batching
- Use quantization
- Check for CPU bottlenecks in preprocessing

### Compatibility Issues

- Verify checkpoint was saved with `save_consolidated: true`
- Check Hugging Face Transformers version compatibility
- Ensure all required model files are present (config.json, tokenizer files)

## See Also

- [Checkpointing Guide](checkpointing.md) - Checkpoint formats and configuration
- [Hugging Face API Compatibility](huggingface-api-compatibility.md) - Model loading and inference
- [vLLM Documentation](https://docs.vllm.ai/) - vLLM deployment guide
- [SGLang Documentation](https://sgl-project.github.io/) - SGLang deployment guide
