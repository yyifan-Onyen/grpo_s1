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


def evaluate_model(model, tokenizer, dataset, reward_function, args, task_name="gsm8k"):
    model.model.eval()
    total_rewards = []
    total_samples = len(dataset)
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
                limitation=args.max_gen_len,
                temperature=args.temperature
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
                "prompt": prompt,
                "response": response,
                "target": target_solution,
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
    parser = argparse.ArgumentParser(description="Evaluate model on tasks")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True, 
                       help="Model path or checkpoint directory")
    parser.add_argument("--model-type", type=str, default="full", choices=["full", "lora", "fact"],
                       help="Type of model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run evaluation on")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                       help="Model dtype")
    
    # Task configuration
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "mbpp"],
                       help="Task to evaluate on")
    
    # Generation parameters
    parser.add_argument("--max-gen-len", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature")
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    # Load dataset
    if args.task == "gsm8k":
        test_dataset = create_gsm8k_dataset()['test']
        reward_function = gsm8k_reward_function
    elif args.task == "mbpp":
        dataset_dict = create_mbpp_dataset()
        test_dataset = dataset_dict['test']
        reward_function = None 
    else:
        print(f"Unsupported task: {args.task}")
        return
    
    print(f"\nEvaluating model on {args.task.upper()}...")
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Total samples: {len(test_dataset)}")
    
    # Load model - FACT, LoRA or full checkpoint
    if args.model_type == "fact":
        print(f"Loading FACT checkpoint: {args.model}")
        
        # Load adapter config
        adapter_config_path = os.path.join(args.model, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Load base model
        base_model_path = adapter_config["original_model_path"]
        fact_config = adapter_config["fact_config"]
        
        print(f"Loading base model: {base_model_path}")
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=args.device
        )
        
        # Apply FACT structure
        target_modules = fact_config.get("fact_target_modules", ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model_fact = apply_fact_to_model(model_base, fact_config, target_modules)
        
        # Load FACT weights
        fact_weights_path = os.path.join(args.model, "fact_adapter.bin")
        fact_state_dict = torch.load(fact_weights_path, map_location=args.device)
        
        # Load weights into model
        missing_keys, unexpected_keys = model_fact.load_state_dict(fact_state_dict, strict=False)
        print(f"Loaded FACT weights - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = FACTModelWrapper(model_fact, tokenizer)
        
    elif args.model_type == "lora":
        print(f"Loading LoRA checkpoint: {args.model}")
        model_peft = AutoPeftModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=args.device
        )
        model = LoRAModelWrapper(model_peft, args.model)
        
    else:  # full checkpoint
        print(f"Loading full checkpoint: {args.model}")
        model = LanguageModel(
            model_path=args.model,
            target_device=args.device,
            torch_dtype=dtype,
        )
    
    # Evaluate model
    print("Starting evaluation...")
    eval_results = evaluate_model(
        model=model,
        tokenizer=model.tokenizer,
        dataset=test_dataset,
        reward_function=reward_function,
        args=args,
        task_name=args.task
    )
    
    # Print results
    print(f"\nEvaluation completed!")
    print(f"Success Rate: {eval_results['success_rate']:.4f}")
    print(f"Average Reward: {eval_results['average_reward']:.4f}")
    print(f"Total Samples: {eval_results['total_samples']}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"Results saved to: {args.output}")
    

if __name__ == "__main__":
    main()
