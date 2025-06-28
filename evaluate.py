import argparse
import torch
import yaml
from pathlib import Path
from utils.model import LanguageModel
from utils.gsm8k import create_gsm8k_dataset, gsm8k_reward_function
from utils.mbpp import create_mbpp_dataset, verify_answer
import numpy as np
from tqdm import tqdm


def evaluate_model(model, tokenizer, dataset, reward_function, config, task_name="gsm8k"):
    model.model.eval()
    total_rewards = []
    total_samples = len(dataset) if config.get("eval_samples", -1) == -1 else min(len(dataset), config.get("eval_samples", -1))
    detailed_results = []
    
    with torch.no_grad():
        for i in tqdm(range(total_samples), desc="Evaluating"):
            example = dataset[i]
            if task_name == "gsm8k":
                prompt = example['prompt']
                target_solution = example['solution']
            elif task_name == "mbpp":
                prompt = f"{example['instruction']}\n\n{example['problem']}"
                target_solution = example['solution']
            else:
                raise ValueError(f"Unsupported task: {task_name}")
            response = model.generate(
                prompts=[prompt],
                limitation=config.get("max_gen_len"),
                temperature=config.get("temperature")
            )
            response = response[0]
            if task_name == "gsm8k":
                reward = reward_function(response, target_solution)
            elif task_name == "mbpp":
                result = verify_answer(response, example['checker'])
                reward = 1 if result['correct'] else 0
            print(reward)
            total_rewards.append(reward)
            detailed_results.append({
                "index": i,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "target": target_solution[:100] + "..." if len(target_solution) > 100 else target_solution,
                "reward": reward
            })
    
    avg_reward = np.mean(total_rewards)
    success_rate = np.sum(total_rewards) / len(total_rewards)
    
    return {
        "average_reward": avg_reward,
        "success_rate": success_rate,
        "total_samples": total_samples,
        "detailed_results": detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Tasks")
    parser.add_argument("--config", type=str, default="config/grpo_s1_eval.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()


    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["dtype"], torch.bfloat16)
    task_name = config["task"]
    if task_name == "gsm8k":
        test_dataset = create_gsm8k_dataset()['test']
        reward_function = gsm8k_reward_function
    elif task_name == "mbpp":
        dataset_dict = create_mbpp_dataset()
        test_dataset = dataset_dict['test']
        reward_function = None 
    else:
        print(f"Unsupported task: {task_name}")
        return
    
    print(f"\nEvaluating model on {task_name.upper()}...")
    print(f"Loading checkpoint: {config['checkpoint_dir']}")
    
    model = LanguageModel(
        model_path=config['checkpoint_dir'],
        target_device=config['device'],
        torch_dtype=dtype,
    )

    eval_results = evaluate_model(
        model=model,
        tokenizer=model.tokenizer,
        dataset=test_dataset,
        reward_function=reward_function,
        config=config,
        task_name=task_name
    )

    import json
    with open(config['output_file'], 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    
    
if __name__ == "__main__":
    main()
