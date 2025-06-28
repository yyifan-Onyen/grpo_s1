# GRPO
NEED Three GPU to run.

CUDA_VISIBLE_DEVICES=0,1,2 python train.py

## STEP1
QWEN-2.5-3B-Insturct
LLAMA-3.2-3B-Insturct
MINISTRAL-3B-Insturct

| Model      | Task   | Performance (Success Rate) | Method    |
|------------|--------|----------------------------|-----------|
| QWEN       | gsm8k  | 79.38%                     | zero-shot |
| QWEN       | gsm8k  | 81.96%                     | trl       |
| MINISTRAL  | gsm8k  | 1.14%                      | zero-shot |
| gemma      | gsm8k  | 8.64%                      | zero-shot |
| QWEN       | gsm8k  | 75.43%                     | costum_trl|



```bash
CUDA_VISIBLE_DEVICES=7 python train_new.py --config config/single_qwen_grpo.yaml 
```

