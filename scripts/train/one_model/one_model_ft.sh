#!/bin/bash

# GRPO训练脚本 - 三个模型独立训练

echo "=== GRPO Three Models Independent Training ==="
export VLLM_USE_V1=0
PYTHON=${PYTHON:-python3}

export TOKENIZERS_PARALLELISM=true

export TORCH_SHOW_CPP_STACKTRACES=1 VLLM_LOGGING_LEVEL=DEBUG

# 创建必要的目录
mkdir -p logs checkpoints
mkdir -p checkpoints/qwen_ft checkpoints/phi_ft checkpoints/llama_ft

# # Qwen模型训练（全量微调 Dense）
# echo "Training Qwen with full finetuning (dense)..."
# CUDA_VISIBLE_DEVICES=2,3 ${PYTHON} train_new.py \
#   --models "Qwen/Qwen2.5-3B-Instruct" \
#   --adapter none \
#   --dtype bfloat16 \
#   --lr 1e-5 \
#   --epochs 1 \
#   --batch-size 128 \
#   --num-answers 8 \
#   --task-type math \
#   --data-root "/workspace/data" \
#   --max-steps 100 \
#   --ckpt-dir "/workspace/checkpoints/qwen_ft" \
#   --ckpt-interval 0 \
#   --no-ref-model \
#   --use-vllm \
#   --vllm-gpu 1 \
#   --vllm-gpu-mem 0.6 \
#   --vllm-max-model-len 32768 \
#   --ppo-epochs 1 \
#   --loss-aggregation token-mean \
#   --advantage-clip 2.0 \
#   --val-interval 4 \
#   --rollout-batch-size 128 \
#   --run-name "GRPO_Qwen2.5-3B-Instruct_ft_$(date +%Y%m%d_%H%M%S)" \
#   > logs/qwen_ft.log 2>&1 &

# QWEN_PID=$!

# # Phi模型训练（已注释）
# echo "Training Phi with FacT finetuning..."
# CUDA_VISIBLE_DEVICES=6,2 ${PYTHON} train_new.py \
#   --models "microsoft/Phi-3-mini-128k-instruct" \
#   --adapter fact \
#   --dtype bfloat16 \
#   --lr 1e-5 \
#   --epochs 1 \
#   --batch-size 128 \
#   --num-answers 8 \
#   --task-type math \
#   --data-root "/workspace/data" \
#   --max-steps 100 \
#   --ckpt-dir "/workspace/checkpoints/phi_fact" \
#   --ckpt-interval 0 \
#   --use-ref-model \
#   --ppo-epochs 2 \
#   --loss-aggregation token-mean \
#   --advantage-clip 2.0 \
#   --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.6 --vllm-max-model-len 32768 \
#   --rollout-batch-size 128 \
#   --run-name "GRPO_Phi-3-mini-128k-instruct_fact_$(date +%Y%m%d_%H%M%S)" \
#   > logs/phi_fact.log 2>&1 &

# PHI_PID=$!

Llama模型训练（full 适配器）
echo "Training Llama with FacT finetuning..."
CUDA_VISIBLE_DEVICES=4,5 ${PYTHON} train_new.py \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --adapter none \
  --dtype bfloat16 \
  --lr 1e-5 \
  --epochs 1 \
  --batch-size 128 \
  --num-answers 8 \
  --task-type math \
  --data-root "/workspace/data" \
  --max-steps 100 \
  --ckpt-dir "/workspace/checkpoints/llama_ft" \
  --ckpt-interval 0 \
  --no-ref-model \
  --use-vllm \
  --vllm-gpu 1 \
  --vllm-gpu-mem 0.3 \
  --vllm-max-model-len 32768 \
  --ppo-epochs 1 \
  --loss-aggregation token-mean \
  --advantage-clip 2.0 \
  --rollout-batch-size 128 \
  --run-name "GRPO_Llama-3.2-3B-Instruct_fact_$(date +%Y%m%d_%H%M%S)" \
  > logs/llama_ft.log 2>&1 &

LLAMA_PID=$!

echo "All training started in parallel!"
# echo "PIDs: Qwen=$QWEN_PID"
# echo "Check logs: tail -f logs/qwen_ft.log"
# echo "Check logs: tail -f logs/phi_fact.log" 
echo "Check logs: tail -f logs/llama_ft.log"

# 等待所有任务完成
# wait $QWEN_PID
# echo "Qwen training completed!"

# wait $PHI_PID
# echo "Phi training completed!"

wait $LLAMA_PID
echo "Llama training completed!"

echo "All parallel training completed!"