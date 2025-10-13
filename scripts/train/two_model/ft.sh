#!/bin/bash

# GRPO训练脚本 - 三个模型并行全量微调（每模型两张卡：一张HF更新 + 一张vLLM推理）

echo "=== GRPO Three Models Full Finetuning (HF+vLLM) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${PYTHON:-python3}

# 创建必要的目录
mkdir -p logs checkpoints
mkdir -p checkpoints/qwen_full checkpoints/phi_full checkpoints/llama_full

# GPU 配置（每模型两张卡，前者用于HF训练/更新，后者用于vLLM rollout）
# Qwen  使用: 2(训练), 3(vLLM)
# Phi   使用: 4(训练), 5(vLLM)
# Llama 使用: 6(训练), 7(vLLM)

# Qwen 模型训练（全量微调）
echo "Training Qwen with full finetuning..."
CUDA_VISIBLE_DEVICES=2,3 ${PYTHON} train_new.py \
  --models "Qwen/Qwen2.5-3B-Instruct" \
  --dtype bfloat16 \
  --adapter none \
  --lr 2e-5 \
  --epochs 1 \
  --batch-size 128 \
  --num-answers 8 \
  --task-type math \
  --data-root "/workspace/data" \
  --max-steps 100 \
  --val-interval 2 \
  --val-batch-size 64 \
  --ckpt-dir "checkpoints/qwen_full" \
  --ckpt-interval 0 \
  --ppo-epochs 2 \
  --loss-aggregation token-mean \
  --advantage-clip 2.0 \
  --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.5 --vllm-max-model-len 32768 \
  --rollout-batch-size 128 \
  --use-ref-model \
  --run-name "GRPO_Qwen2.5-3B-Instruct_full_$(date +%Y%m%d_%H%M%S)" \
  > logs/qwen_full.log 2>&1 &

QWEN_PID=$!

# Phi 模型训练（全量微调）
echo "Training Phi with full finetuning..."
CUDA_VISIBLE_DEVICES=4,5 ${PYTHON} train_new.py \
  --models "microsoft/Phi-3-mini-128k-instruct" \
  --dtype bfloat16 \
  --adapter none \
  --lr 3e-5 \
  --epochs 1 \
  --batch-size 128 \
  --num-answers 8 \
  --task-type math \
  --data-root "/workspace/data" \
  --max-steps 100 \
  --val-interval 2 \
  --val-batch-size 64 \
  --ckpt-dir "checkpoints/phi_full" \
  --ckpt-interval 0 \
  --ppo-epochs 2 \
  --loss-aggregation token-mean \
  --advantage-clip 2.0 \
  --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.5 --vllm-max-model-len 32768 \
  --rollout-batch-size 128 \
  --use-ref-model \
  --run-name "GRPO_Phi-3-mini-128k-instruct_full_$(date +%Y%m%d_%H%M%S)" \
  > logs/phi_full.log 2>&1 &

PHI_PID=$!

# Llama 模型训练（全量微调）
echo "Training Llama with full finetuning..."
CUDA_VISIBLE_DEVICES=6,7 ${PYTHON} train_new.py \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --dtype bfloat16 \
  --adapter none \
  --lr 1e-6 \
  --epochs 1 \
  --batch-size 128 \
  --num-answers 8 \
  --task-type math \
  --data-root "/workspace/data" \
  --max-steps 100 \
  --val-interval 2 \
  --val-batch-size 64 \
  --ckpt-dir "checkpoints/llama_full" \
  --ckpt-interval 0 \
  --ppo-epochs 2 \
  --loss-aggregation token-mean \
  --advantage-clip 2.0 \
  --use-vllm --vllm-gpu 1 --vllm-gpu-mem 0.5 --vllm-max-model-len 32768 \
  --rollout-batch-size 128 \
  --use-ref-model \
  --run-name "GRPO_Llama-3.2-3B-Instruct_full_$(date +%Y%m%d_%H%M%S)" \
  > logs/llama_full.log 2>&1 &

LLAMA_PID=$!

echo "All training started in parallel!"
echo "PIDs: Qwen=$QWEN_PID, Phi=$PHI_PID, Llama=$LLAMA_PID"
echo "Check logs: tail -f logs/qwen_full.log"
echo "Check logs: tail -f logs/phi_full.log"
echo "Check logs: tail -f logs/llama_full.log"

# 等待所有任务完成
wait $QWEN_PID
echo "Qwen training completed!"

wait $PHI_PID
echo "Phi training completed!"

wait $LLAMA_PID
echo "Llama training completed!"

echo "All parallel training completed!"
