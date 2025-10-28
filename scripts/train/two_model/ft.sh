#!/bin/bash

# GRPO 训练脚本 - 双模型协同训练 LoRA（单进程同时训练并共同更新）

echo "=== GRPO Two Models FT (Collaborative Training) ==="
export VLLM_USE_V1=0
PYTHON=${PYTHON:-python3}
export TOKENIZERS_PARALLELISM=true

export TORCH_SHOW_CPP_STACKTRACES=1 VLLM_LOGGING_LEVEL=DEBUG

# 创建必要的目录
mkdir -p logs checkpoints
mkdir -p checkpoints/two_model_ft

# 单进程 + 双模型：将两种模型一起传入 --models，开启协同训练路径
# 显卡分配：为每个模型分配两张卡（左=HF更新，右=vLLM rollout）。
# 使用顺序：CUDA_VISIBLE_DEVICES=2,3,4,5 →
#   模型0（Qwen）：HF=索引0(2)，vLLM=索引2(4)
#   模型1（Llama）：HF=索引1(3)，vLLM=索引3(5)

CUDA_VISIBLE_DEVICES=4,5,6,7 ${PYTHON} train_new.py \
  --models "Qwen/Qwen2.5-3B-Instruct" "HuggingFaceTB/SmolLM3-3B" \
  --adapter none \
  --dtype bfloat16 \
  --epochs 1 \
  --batch-size 64 \
  --num-answers 8 \
  --task-type math \
  --data-root "/workspace/data" \
  --max-steps 100 \
  --ckpt-dir "checkpoints/two_model_ft" \
  --ckpt-interval 0 \
  --ppo-epochs 1 \
  --beta 0.2 \
  --lr 5e-6 \
  --use-ref-model \
  --ppo-micro-batch-size 2 \
  --val-interval 25 \
  --loss-aggregation token-mean \
  --advantage-clip 0.6 \
  --use-vllm --vllm-gpu-mem 0.6 --vllm-max-model-len 32768 --vllm-gpus 2 -1 \
  --rollout-batch-size 32 \
  --run-name "GRPO_TwoModels_FT_$(date +%Y%m%d_%H%M%S)" \
  > logs/two_model_ft.log 2>&1 &

PID=$!

echo "Two-model FT collaborative training started! PID=$PID"
echo "Check logs: tail -f logs/two_model_ft.log"

wait $PID
echo "Two-model FT collaborative training completed!"
