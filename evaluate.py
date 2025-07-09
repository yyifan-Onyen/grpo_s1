import argparse
import torch
import yaml
from pathlib import Path
from utils.model import LanguageModel
from utils.gsm8k import create_gsm8k_dataset, gsm8k_reward_function
from utils.mbpp import create_mbpp_dataset, verify_answer
import numpy as np
from tqdm import tqdm
# LoRA support
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
# FACT support
from utils.fact_adapter import apply_fact_to_model
import json
import os


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


class LoRAModelWrapper:
    """Wrapper for LoRA models to match LanguageModel interface"""
    def __init__(self, peft_model, checkpoint_dir):
        self.model = peft_model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def generate(self, prompts, limitation=1024, temperature=1.0):
        """Generate completions matching LanguageModel.generate interface"""
        conversations = []
        for prompt in prompts:
            conversations.append(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ))

        inputs = self.tokenizer(
            conversations,
            padding=True,
            padding_side="left", 
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            indices = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=limitation,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            )
            length = inputs["input_ids"].shape[1]
            completions = self.tokenizer.batch_decode(
                indices[:, length:],
                skip_special_tokens=True
            )
        return completions


class FACTModelWrapper:
    """Wrapper for FACT models to match LanguageModel interface"""
    def __init__(self, fact_model, tokenizer):
        self.model = fact_model
        self.tokenizer = tokenizer
    
    def generate(self, prompts, limitation=1024, temperature=1.0):
        """Generate completions matching LanguageModel.generate interface"""
        conversations = []
        for prompt in prompts:
            conversations.append(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ))

        inputs = self.tokenizer(
            conversations,
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            indices = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=limitation,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            )
            length = inputs["input_ids"].shape[1]
            completions = self.tokenizer.batch_decode(
                indices[:, length:],
                skip_special_tokens=True
            )
        return completions


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Tasks")
    parser.add_argument("--config", type=str, default="config/grpo_s1_eval.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["dtype"], torch.bfloat16)
    
    # Load dataset
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
    
    # Load model - FACT, LoRA or full checkpoint
    if config.get("is_fact", False):
        print(f"Loading FACT checkpoint: {config['checkpoint_dir']}")
        
        # Load adapter config
        adapter_config_path = os.path.join(config['checkpoint_dir'], "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Load base model
        base_model_path = adapter_config["original_model_path"]
        fact_config = adapter_config["fact_config"]
        
        print(f"Loading base model: {base_model_path}")
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=config['device']
        )
        
        # Apply FACT structure
        target_modules = fact_config.get("fact_target_modules", ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model_fact = apply_fact_to_model(model_base, fact_config, target_modules)
        
        # Load FACT weights
        fact_weights_path = os.path.join(config['checkpoint_dir'], "fact_adapter.bin")
        fact_state_dict = torch.load(fact_weights_path, map_location=config['device'])
        
        # Load weights into model
        missing_keys, unexpected_keys = model_fact.load_state_dict(fact_state_dict, strict=False)
        print(f"Loaded FACT weights - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['checkpoint_dir'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = FACTModelWrapper(model_fact, tokenizer)
        
    elif config.get("is_lora", False):
        print(f"Loading LoRA checkpoint: {config['checkpoint_dir']}")
        model_peft = AutoPeftModelForCausalLM.from_pretrained(
            config['checkpoint_dir'],
            torch_dtype=dtype,
            device_map=config['device']
        )
        model = LoRAModelWrapper(model_peft, config['checkpoint_dir'])
    else:
        print(f"Loading full checkpoint: {config['checkpoint_dir']}")
        model = LanguageModel(
            model_path=config['checkpoint_dir'],
            target_device=config['device'],
            torch_dtype=dtype,
        )

    # Evaluate
    eval_results = evaluate_model(
        model=model,
        tokenizer=model.tokenizer,
        dataset=test_dataset,
        reward_function=reward_function,
        config=config,
        task_name=task_name
    )

    with open(config['output_file'], 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    
    
if __name__ == "__main__":
    main()
