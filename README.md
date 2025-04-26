# benchmark-contamination
Analysis of benchmark contamination in open-source LLMs, evaluating reasoning performance on AIME-2024 & MMLU while implementing contamination detection techniques.

## Evaluation Script Usage

Run the evaluation script with various options to test different models and tasks:

```bash
# Basic usage with defaults (DeepSeek 1.5B model on aime24)
./eval.sh

# Evaluate DeepSeek-R1-Distill-Qwen-14B on aime24 with 4 GPUs
./eval.sh --model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --task=aime24 --devices=0,1,2,3

# Evaluate Qwen2.5-Math-7B on MMLU with 2 GPUs (note the max-model-length for Qwen models)
./eval.sh --model=Qwen/Qwen2.5-Math-7B-Instruct --task=mmlu:high_school_mathematics --devices=0,1 --max-model-length=4096 --temp=1.0 --top-p=0.9
```

### Available Models

#### DeepSeek Models
* deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
* deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
* deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
* deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
* deepseek-ai/DeepSeek-R1-Distill-Llama-8B
* deepseek-ai/DeepSeek-R1-Distill-Llama-70B

#### Qwen Models
* Qwen/Qwen2.5-Math-1.5B
* Qwen/Qwen2.5-Math-1.5B-Instruct
* Qwen/Qwen2.5-Math-7B
* Qwen/Qwen2.5-Math-72B
* Qwen/Qwen2.5-Math-72B-Instruct
* Qwen/Qwen2.5-Math-RM-72B
* Qwen/Qwen2.5-Math-7B-Instruct
* Qwen/Qwen2.5-Math-PRM-7B
* Qwen/Qwen2.5-Math-PRM-72B

### Available Options
* `--model`: Model to evaluate
* `--task`: Evaluation task (aime24, aime25, math_500, gpqa:diamond, mmlu:high_school_mathematics)
* `--devices`: Comma-separated list of GPU device IDs
* `--max-samples`: Maximum number of samples to evaluate (defaults to 100 for MMLU, unlimited for other tasks)
* `--max-model-length`: Maximum model context length (defaults to 32768 for DeepSeek, use 4096 for Qwen models)
* `--temp`: Temperature parameter for generation (default: 0.6)
* `--top-p`: Top-p sampling parameter (default: 0.95)

## Testing for Contamination

Refer to link [here](https://github.com/nate-daba/detect-benchmark-contamination) for details on how to test for contamination in the models.
