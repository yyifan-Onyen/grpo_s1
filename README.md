# GRPO: SAIYAN


## STEP1
QWEN-2.5-3B-Insturct
LLAMA-3.2-3B-Insturct
MINISTRAL-3B-Insturct


WORKFLOW
1.同一个问题：三个模型都对相同batch进行sampling
2.汇总所有答案：把三个模型的答案混合成一个大的经验池
3.共享学习：每个模型都从整个经验池中学习，不只是自己的答案
4.分别更新：虽然学习材料相同，但每个模型独立更新自己的参数