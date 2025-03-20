#!/bin/bash

# Default values
MODEL_BASE="deepseek-ai/DeepSeek-R1-Distill"
MODEL_VARIANT="Qwen-1.5B"
TASK="aime24"
NUM_GPUS=1
DEVICE_IDS="0"
MAX_SAMPLES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model=*)
      MODEL_VARIANT="${1#*=}"
      shift
      ;;
    --task=*)
      TASK="${1#*=}"
      shift
      ;;
    --ngpus=*)
      NUM_GPUS="${1#*=}"
      shift
      ;;
    --devices=*)
      DEVICE_IDS="${1#*=}"
      shift
      ;;
    --max-samples=*)
      MAX_SAMPLES="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 --model=MODEL_VARIANT --task=TASK --ngpus=NUM_GPUS --devices=DEVICE_IDS [--max-samples=MAX_SAMPLES]"
      echo "  MODEL_VARIANT: Qwen-1.5B, Qwen-7B, Qwen-14B, Qwen-32B, Llama-8B, Llama-70B"
      echo "  TASK: aime24, aime25, math_500, gpqa:diamond, mmlu:high_school_mathematics"
      exit 1
      ;;
  esac
done

# Validate model
case $MODEL_VARIANT in
  "Qwen-1.5B"|"Qwen-7B"|"Qwen-14B"|"Qwen-32B"|"Llama-8B"|"Llama-70B")
    # Valid model
    ;;
  *)
    echo "Invalid model: $MODEL_VARIANT"
    echo "Valid models are: Qwen-1.5B, Qwen-7B, Qwen-14B, Qwen-32B, Llama-8B, Llama-70B"
    exit 1
    ;;
esac

# Validate task
case $TASK in
  "aime24"|"aime25"|"math_500"|"gpqa:diamond"|"mmlu:high_school_mathematics")
    # Valid task
    ;;
  *)
    echo "Invalid task: $TASK"
    echo "Valid tasks are: aime24, aime25, math_500, gpqa:diamond, mmlu:high_school_mathematics"
    exit 1
    ;;
esac

# Set the full model name
MODEL="${MODEL_BASE}-${MODEL_VARIANT}"
OUTPUT_DIR="../data/evals/$MODEL"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"

# Construct model arguments
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8"

# Add tensor parallelism if using multiple GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
    MODEL_ARGS="$MODEL_ARGS,tensor_parallel_size=$NUM_GPUS"
fi

MODEL_ARGS="$MODEL_ARGS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Set VLLM worker method for all runs
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Construct max-samples argument if specified
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
elif [ "$TASK" == "mmlu:high_school_mathematics" ]; then
    # Default to 100 samples for MMLU
    MAX_SAMPLES_ARG="--max-samples 100"
fi

# Print configuration
echo "Running evaluation with:"
echo "  Model: $MODEL"
echo "  Task: $TASK"
echo "  GPUs: $NUM_GPUS (devices: $DEVICE_IDS)"
echo "  Output directory: $OUTPUT_DIR"
echo "  Max samples: ${MAX_SAMPLES:-'not specified'}"

# Run the evaluation
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks ../open-r1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR" \
    $MAX_SAMPLES_ARG

echo "Evaluation complete."