#Training Scrpt for LoRA
CUDA_VISIBLE_DEVICES=6 python -u train_new.py --config config/grpo_s1_lora.yaml > result_lora.txt

#Training Scrpt for FACT
CUDA_VISIBLE_DEVICES=7 python -u train_new.py --config config/grpo_s1_fact.yaml > result_fact.txt

#Evaluation Scrpt for LoRA
CUDA_VISIBLE_DEVICES=7 python evaluate.py --config config/test/lora_qwen_gsm8k.yaml 

#Evaluation Scrpt for FACT
CUDA_VISIBLE_DEVICES=6 python evaluate.py --config config/test/fact_qwen_gsm8k.yaml 