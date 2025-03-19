#! /bin/bash

NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=2,3
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=../data/evals/$MODEL

# AIME 2024
TASK=aime24
export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks ../open-r1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

