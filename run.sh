#Training Scrpt for LoRA
CUDA_VISIBLE_DEVICES=6 python -u train_new.py --config config/grpo_s1_lora.yaml > result_lora.txt

#Training Scrpt for FACT
CUDA_VISIBLE_DEVICES=7 python -u train_new.py --config config/grpo_s1_fact.yaml > result_fact.txt

#Evaluation Scrpt for LoRA
CUDA_VISIBLE_DEVICES=7 python evaluate.py --config config/test/lora_qwen_gsm8k.yaml 

#Evaluation Scrpt for FACT
CUDA_VISIBLE_DEVICES=6 python evaluate.py --config config/test/fact_qwen_gsm8k.yaml 

#尝试权重实验
python train_weight.py --config config/train/grpo_ranking_weights.yaml > logs/grpo_ranking_weights.txt

#尝试threshold实验
python train_threshold.py --config config/train/grpo_threshold.yaml > logs/grpo_threshold.txt

#尝试grpo_threshold实验
python evaluate.py --config config/test/gsm8k_eval.yaml









#ablation study
python evaluate.py --config config/test/llama_trl_gsm8k.yaml

python evaluate.py --config config/test/min_trl_gsm8k.yaml

#尝试phil实验
python evaluate.py --config config/test/gsm8l_phil.yaml




