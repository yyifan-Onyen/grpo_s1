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
| QWEN       | gsm8k  | 83.85%                     | costum_trl|
| QWEN       | gsm8k  | 82.79%                     | FACT 4000 |
| QWEN       | gsm8k  | 82.03%                     | FACT 6000 |
| QWEN       | gsm8k  | 79.83%                     | LoRA 4000 |
| QWEN       | gsm8k  | 82.18%                     | FACT 2000 |





# This is how to run the train
```bash
CUDA_VISIBLE_DEVICES=5 python train_new.py --config config/single_qwen_grpo.yaml 
```

# This is How to run the test
```bash
CUDA_VISIBLE_DEVICES=7 python evaluate.py --config config/single_qwen_grpo.yaml 
```
