#!/bin/bash

# Default values
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TASK="aime24"
DEVICE_IDS="0"
MAX_SAMPLES=""
MAX_MODEL_LENGTH="32768"  # 32768 (DeepSeek-R1) or 4096 (Qwen)
TEMP="0.6" # 0.6 (DeepSeek-R1) or 1.0 (Qwen)
TOP_P="0.95" # 0.95 (DeepSeek-R1) or 0.9 (Qwen)
# Add cleanup function for graceful termination
cleanup_gpu_processes() {
  echo "Cleaning up GPU processes for devices: $DEVICE_IDS"
  GPU_PIDS=$(nvidia-smi -i "$DEVICE_IDS" --query-compute-apps=pid --format=csv,noheader)
  if [ -n "$GPU_PIDS" ]; then
    for pid in $GPU_PIDS; do
      if ! ps -p $pid -o comm= | grep -q "Xorg"; then
        echo "Terminating process $pid"
        kill -9 $pid 2>/dev/null || true
      fi
    done
    echo "GPU cleanup complete"
  else
    echo "No processes to clean up"
  fi
}
trap cleanup_gpu_processes EXIT INT TERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --task=*)
      TASK="${1#*=}"
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
    --max-model-length=*)
      MAX_MODEL_LENGTH="${1#*=}"
      shift
      ;;
    --temp=*)
      TEMP="${1#*=}"
      shift
      ;;
    --top-p=*)
      TOP_P="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 --model=MODEL --task=TASK --devices=DEVICE_IDS [--max-samples=MAX_SAMPLES]"
      echo "  MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, \
                     deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, \
                     deepseek-ai/DeepSeek-R1-Distill-Qwen-14B, \
                     deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, \
                     deepseek-ai/DeepSeek-R1-Distill-Llama-8B, \
                     deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
      echo "  TASK: aime24, aime25, math_500, gpqa:diamond, mmlu:high_school_mathematics"
      exit 1
      ;;
  esac
done

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

# Use parameter expansion to replace slashes with hyphens for directory name
MODEL_NAME=${MODEL//\//-}
OUTPUT_DIR="../data/evals/$MODEL_NAME"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"

# Count the number of GPUs 
NUM_GPUS=$(echo "$DEVICE_IDS" | awk -F, '{print NF}')

# Construct model arguments
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.9"

# Add tensor parallelism if using multiple GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
    MODEL_ARGS="$MODEL_ARGS,tensor_parallel_size=$NUM_GPUS"
fi

MAX_NEW_TOKENS=$((MAX_MODEL_LENGTH / 2))  # Use half of the model's context for generation
MODEL_ARGS="$MODEL_ARGS,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMP,top_p:$TOP_P}"

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
echo "  Max model length: $MAX_MODEL_LENGTH"
echo "  Temperature: $TEMP"
echo "  Top-p: $TOP_P"

# Run the evaluation
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks ../open-r1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR" \
    $MAX_SAMPLES_ARG

echo "Evaluation complete."