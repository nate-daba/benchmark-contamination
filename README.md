# benchmark-contamination
Analysis of benchmark contamination in open-source LLMs, evaluating reasoning performance on AIME-2024 &amp; MMLU while implementing contamination detection techniques.

## Evaluation Script Usage

Run the evaluation script with various options to test different models and tasks:

```bash
# Basic usage with defaults (1.5B model on aime24)
./eval.sh

# Evaluate Qwen-14B on mmlu with 2 GPUs
./eval.sh --model=Qwen-14B --task=mmlu:high_school_mathematics --ngpus=2 --devices=2,3

# Evaluate Llama-70B on math_500 with 8 GPUs (max-samples=100 by default for MMLU)
./eval.sh --model=Llama-70B --task=math_500 --ngpus=8 --devices=0,1,2,3,4,5,6,7

# Override max samples for any task
./eval.sh --task=aime24 --max-samples=10
