#!/bin/bash

# GRPO训练脚本 - 三个模型独立训练

echo "=== GRPO Three Models Independent Training ==="
export VLLM_USE_V1=0
PYTHON=${PYTHON:-python3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export TORCH_SHOW_CPP_STACKTRACES=1 VLLM_LOGGING_LEVEL=DEBUG
# 创建必要的目录
mkdir -p logs checkpoints
mkdir -p checkpoints/qwen_lora checkpoints/phi_lora checkpoints/smollm3_lora

# Qwen模型训练（lora微调）
echo "Training Qwen with LoRA finetuning..."
# 使用两张卡：第1张给训练(HF)，第2张给vLLM，vLLM本地索引为1
CUDA_VISIBLE_DEVICES=6,7 python train_new.py \
  --models "Qwen/Qwen2.5-3B-Instruct" \
  --adapter lora \
  --dtype bfloat16 \
  --lr 5e-6 \
  --epochs 1 \
  --batch-size 128 \
  --num-answers 8 \
  --task-type math \
  --data-root "/workspace/data" \
  --max-steps 100 \
  --ckpt-dir "/workspace/checkpoints/qwen_lora" \
  --ckpt-interval 0 \
  --val-interval 20 \
  --use-ref-model \
  --ppo-epochs 1 \
  --beta 0.2 \
  --lr 5e-6 \
  --loss-aggregation token-mean \
  --advantage-clip 0.6 \
  --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.6 --vllm-max-model-len 32768 \
  --rollout-batch-size 128 \
  --run-name "GRPO_Qwen2.5-3B-Instruct_lora_$(date +%Y%m%d_%H%M%S)" \
  > logs/qwen_lora.log 2>&1 &

QWEN_PID=$!

# # Phi模型训练（lora微调）
# echo "Training Phi with LoRA finetuning..."
# CUDA_VISIBLE_DEVICES=6,2 python train_new.py \
#   --models "microsoft/Phi-3-mini-128k-instruct" \
#   --adapter lora \
#   --dtype bfloat16 \
#   --lr 1e-5 \
#   --epochs 1 \
#   --batch-size 128 \
#   --num-answers 8 \
#   --task-type math \
#   --data-root "/workspace/data" \
#   --max-steps 100 \
#   --ckpt-dir "/workspace/checkpoints/phi_lora" \
#   --ckpt-interval 0 \
#   --use-ref-model \
#   --ppo-epochs 2 \
#   --loss-aggregation token-mean \
#   --advantage-clip 2.0 \
#   --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.6 --vllm-max-model-len 32768 \
#   --rollout-batch-size 128 \
#   --run-name "GRPO_Phi-3-mini-128k-instruct_lora_$(date +%Y%m%d_%H%M%S)" \
#   > logs/phi_lora.log 2>&1 &

# PHI_PID=$!


# Llama模型训练（lora微调）
echo "Training SmolLM3-3B-Instruct with LoRA finetuning..."
CUDA_VISIBLE_DEVICES=5,4 python train_new.py \
  --models "HuggingFaceTB/SmolLM3-3B" \
  --adapter lora \
  --dtype bfloat16 \
  --lr 1e-5 \
  --epochs 1 \
  --batch-size 128 \
  --num-answers 8 \
  --task-type math \
  --data-root "/workspace/data" \
  --max-steps 100 \
  --ckpt-dir "/workspace/checkpoints/smollm3_lora" \
  --ckpt-interval 0 \
  --val-interval 20 \
  --use-ref-model \
  --ppo-epochs 1 \
  --beta 0.2 \
  --lr 5e-6 \
  --loss-aggregation token-mean \
  --advantage-clip 0.6 \
  --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.6 --vllm-max-model-len 32768 \
  --rollout-batch-size 128 \
  --run-name "GRPO_SmolLM3-3B_lora_$(date +%Y%m%d_%H%M%S)" \
  > logs/smollm3_lora.log 2>&1 &

SMOLLM3_PID=$!

# # Llama模型训练（lora微调）
# echo "Training Llama with LoRA finetuning..."
# CUDA_VISIBLE_DEVICES=5,4 python train_new.py \
#   --models "meta-llama/Llama-3.2-3B-Instruct" \
#   --adapter lora \
#   --dtype bfloat16 \
#   --lr 1e-5 \
#   --epochs 1 \
#   --batch-size 128 \
#   --num-answers 8 \
#   --task-type math \
#   --data-root "/workspace/data" \
#   --max-steps 500 \
#   --ckpt-dir "/workspace/checkpoints/llama_lora" \
#   --ckpt-interval 0 \
#   --val-interval 50 \
#   --use-ref-model \
#   --ppo-epochs 1 \
#   --beta 0.2 \
#   --lr 5e-6 \
#   --loss-aggregation token-mean \
#   --advantage-clip 2.0 \
#   --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.6 --vllm-max-model-len 32768 \
#   --rollout-batch-size 128 \
#   --run-name "GRPO_Llama-3.2-3B-Instruct_lora_$(date +%Y%m%d_%H%M%S)" \
#   > logs/llama_lora.log 2>&1 &

# LLAMA_PID=$!

echo "All training started in parallel!"
echo "PIDs: Qwen=$QWEN_PID, SmolLM3=$SMOLLM3_PID"
echo "Check logs: tail -f logs/qwen_lora.log"
# echo "Check logs: tail -f logs/phi_lora.log" 
echo "Check logs: tail -f logs/smollm3_lora.log"

# 等待所有任务完成
wait $QWEN_PID
echo "Qwen training completed!"

# wait $PHI_PID
# echo "Phi training completed!"

wait $SMOLLM3_PID
echo "SmolLM3 training completed!"

echo "All parallel training completed!"
