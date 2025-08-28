#!/bin/bash

# GRPO训练脚本 - 三个模型独立训练

echo "=== GRPO Three Models Independent Training ==="

# 创建必要的目录
mkdir -p logs checkpoints
mkdir -p checkpoints/qwen_full checkpoints/phi_full checkpoints/llama_full

# Qwen模型训练（全量微调）
echo "Training Qwen with full finetuning..."
CUDA_VISIBLE_DEVICES=7 python train_new.py \
  --models "Qwen/Qwen2.5-3B-Instruct" \
  --adapter none \
  --lr 2e-5 \
  --epochs 2 \
  --batch-size 16 \
  --num-answers 8 \
  --max-steps 2000 \
  --ckpt-dir "checkpoints/qwen_full" \
  --ckpt-interval 0 

QWEN_PID=$!

# Phi模型训练（全量微调）
echo "Training Phi with full finetuning..."
CUDA_VISIBLE_DEVICES=6 python train_new.py \
  --models "microsoft/Phi-3-mini-128k-instruct" \
  --adapter none \
  --lr 3e-5 \
  --epochs 2 \
  --batch-size 16 \
  --num-answers 8 \
  --max-steps 2000 \
  --ckpt-dir "checkpoints/phi_full" \
  --ckpt-interval 0 \
  > logs/phi_full.log 2>&1 &

PHI_PID=$!

# Llama模型训练（全量微调）
echo "Training Llama with full finetuning..."
CUDA_VISIBLE_DEVICES=5 python train_new.py \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --adapter none \
  --lr 1e-6 \
  --epochs 2 \
  --batch-size 16 \
  --num-answers 8 \
  --max-steps 2000 \
  --ckpt-dir "checkpoints/llama_full" \
  --ckpt-interval 0 \
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

# 显示结果
echo "=== Training Results ==="
echo "Qwen Full checkpoints: $(ls checkpoints/qwen_full/ | wc -l) saved"
echo "Phi Full checkpoints: $(ls checkpoints/phi_full/ | wc -l) saved"
echo "Llama Full checkpoints: $(ls checkpoints/llama_full/ | wc -l) saved"