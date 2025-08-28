# FACT模型评估 - Qwen在GSM8K上的表现
python evaluate.py \
  --model /home/local/PARTNERS/yz646/grpo_revision/grpo-s1/checkpoints/grpo_fact_qwen3b/Qwen2.5-3B-Instruct_step_6000 \
  --model-type fact \
  --task gsm8k \
  --device cuda:1 \
  --output qwen_fact_gsm8k_results.json

# FACT模型评估 - Llama在GSM8K上的表现
python evaluate.py \
  --model /home/local/PARTNERS/yz646/grpo_revision/grpo-s1/checkpoints/grpo_fact_llama/Llama-3.2-3B-Instruct_step_6000 \
  --model-type fact \
  --task gsm8k \
  --device cuda:1 \
  --output llama_fact_gsm8k_results.json

# FACT模型评估 - Phi在GSM8K上的表现
python evaluate.py \
  --model /home/local/PARTNERS/yz646/grpo_revision/grpo-s1/checkpoints/grpo_fact_phi/Phi-3-mini-128k-instruct_step_6000 \
  --model-type fact \
  --task gsm8k \
  --device cuda:1 \
  --output phi_fact_gsm8k_results.json

# FACT模型评估 - Qwen在GSM8K上的表现
python evaluate.py \
  --model /home/local/PARTNERS/yz646/grpo_revision/grpo-s1/checkpoints/grpo_lora_qwen3b/Qwen2.5-3B-Instruct_step_6000 \
  --model-type lora \
  --task gsm8k \
  --device cuda:1 \
  --output qwen_lora_gsm8k_results.json

# FACT模型评估 - Llama在GSM8K上的表现
python evaluate.py \
  --model /home/local/PARTNERS/yz646/grpo_revision/grpo-s1/checkpoints/grpo_lora_llama/Llama-3.2-3B-Instruct_step_6000 \
  --model-type lora \
  --task gsm8k \
  --device cuda:1 \
  --output llama_lora_gsm8k_results.json

# FACT模型评估 - Phi在GSM8K上的表现
python evaluate.py \
  --model /home/local/PARTNERS/yz646/grpo_revision/grpo-s1/checkpoints/grpo_lora_phi/Phi-3-mini-128k-instruct_step_6000 \
  --model-type lora \
  --task gsm8k \
  --device cuda:1 \
  --output phi_lora_gsm8k_results.json