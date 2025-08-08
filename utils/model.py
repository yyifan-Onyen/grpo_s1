# from typing import List, Dict, Tuple, Optional, Union
# import torch
# from peft import LoraConfig, TaskType, get_peft_model
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# class LanguageModel(object):
#     """A wrapper class for language models from HuggingFace."""
#     def __init__(
#         self,
#         model_path: str,
#         target_device: str = "cuda",
#         torch_dtype: str = "auto",
#         attn_impl: str = "sdpa",
#         lora_config: Optional[Dict] = None
#     ):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
#         self.device = target_device
        
#         # Load model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map=target_device if target_device != "cpu" else None,
#             torch_dtype=torch_dtype,
#             attn_implementation=attn_impl,
#         )
        
#         # If target device is CPU, don't specify device_map but move manually
#         if target_device == "cpu":
#             self.model = self.model.to("cpu")
        
#         print(f"Model {model_path} loaded on device: {target_device}")
        
#         if lora_config is not None:
#             self.model = get_peft_model(
#                 self.model,
#                 peft_config=LoraConfig(
#                     task_type=TaskType.CAUSAL_LM,
#                     r=lora_config["lora_rank"],
#                     lora_alpha=lora_config["lora_alpha"],
#                     lora_dropout=lora_config["lora_dropout"]
#                 )
#             )
                
#         self.eos_token = self.tokenizer.eos_token
#         self.eos_token_id = self.tokenizer.eos_token_id  
#         self.pad_token_id = self.tokenizer.pad_token_id

#     def generate(
#         self,
#         prompts: List[str],
#         limitation: int = 1024,
#         temperature: float = 1.0,
#         verbose: bool = False
#     ) -> Union[List[str], Tuple[List[str], torch.Tensor, torch.Tensor]]:
#         """Generate text completions based on the provided prompts."""
#         conversations = []
#         for prompt in prompts:
#             conversations.append(self.tokenizer.apply_chat_template(
#                 [{"role": "user", "content": prompt}],
#                 tokenize=False,
#                 add_generation_prompt=True
#             ))

#         inputs = self.tokenizer(
#             conversations,
#             padding=True,
#             padding_side="left",
#             add_special_tokens=False,
#             return_tensors="pt"
#         )
        
#         # Move inputs to model's device
#         device = next(self.model.parameters()).device
#         inputs = inputs.to(device)

#         with torch.no_grad():
#             indices = self.model.generate(
#                 **inputs,
#                 generation_config=GenerationConfig(
#                     max_new_tokens=limitation,
#                     do_sample=True,
#                     temperature=temperature,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                 )
#             )
#             length = inputs["input_ids"].shape[1]
#             completions = self.tokenizer.batch_decode(
#                 indices[:, length:],
#                 skip_special_tokens=True
#             )
#             masks = torch.zeros_like(indices, dtype=torch.bool)
#             masks[:, length:] = True
#             masks[indices == self.tokenizer.pad_token_id] = False

#         if verbose:
#             return completions, indices, masks
#         else:
#             return completions
        

#     def compute_log_probs(self, indices: torch.Tensor) -> torch.Tensor:
#         """Compute log probabilities for the given token indices."""
#         input_ids = indices.to(self.device)
#         attention_mask = input_ids != self.tokenizer.pad_token_id
#         position_ids = attention_mask.long().cumsum(dim=-1) - 1
#         position_ids[attention_mask == 0] = 0

#         input_logits = self.model.forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             use_cache=False
#         )["logits"][:, :-1]

#         log_probs = torch.gather(
#             torch.log_softmax(input_logits, dim=-1),
#             dim=-1,
#             index=input_ids[:, 1:].unsqueeze(-1)
#         ).squeeze(-1)
#         return log_probs


#     def save(self, path: str) -> None:
#         self.tokenizer.save_pretrained(path)
#         self.model.save_pretrained(path)


# if __name__ == "__main__":
#     # path = "Qwen/Qwen2.5-3B-Instruct"
#     path = "ministral/Ministral-3b-instruct"
#     # path = "meta-llama/Llama-3.2-3B-Instruct"
#     model = LanguageModel(path)

#     prompts = [
#         "Who are you?",
#         "What is the capital of France?",
#         "What is the square root of 16?",
#     ]
#     completions, indices, masks = model.generate(
#         prompts,
#         limitation=256,
#         temperature=1.0,
#         verbose=True
#     )
    
#     log_probs = model.compute_log_probs(indices)

#     print("The shape of indices:", indices.shape)
#     print("The shape of masks:", masks.shape)
#     print("The shape of log_probs:", log_probs.shape)
#     print("Indices: ", indices)
#     print("Masks: ", masks)
#     print("Probabilities: ", log_probs)
#     for prompt, completion in zip(prompts, completions):
#         print(f"Prompt: {prompt}")
#         print(f"Completion: {completion}")


from typing import List, Dict, Tuple, Optional, Union
import torch
import json
import os
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .fact_adapter import apply_fact_to_model, count_fact_parameters, analyze_shared_fact_parameters, analyze_all_trainable_parameters, freeze_non_fact_parameters


class LanguageModel(object):
    """A wrapper class for language models from HuggingFace."""
    def __init__(
        self,
        model_path: str,
        target_device: str = "cuda",
        torch_dtype: str = "auto",
        attn_impl: str = "sdpa",
        lora_config: Optional[Dict] = None,
        fact_config: Optional[Dict] = None,
        gradient_checkpointing: bool = False
    ):
        # Store configuration for save() method
        self.original_model_path = model_path
        self.lora_config = lora_config
        self.fact_config = fact_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = target_device
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=target_device if target_device != "cpu" else None,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        
        # If target device is CPU, don't specify device_map but move manually
        if target_device == "cpu":
            self.model = self.model.to("cpu")
        
        print(f"Model {model_path} loaded on device: {target_device}")
        
        # Configure gradient checkpointing if enabled
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print(f"Gradient checkpointing enabled for {model_path}")
        
        # Apply FacT if configured
        if fact_config is not None:
            target_modules = fact_config.get("fact_target_modules", ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.model = apply_fact_to_model(self.model, fact_config, target_modules)
            
            # Freeze all non-FacT parameters to ensure only adaptation parameters are trainable
            freeze_non_fact_parameters(self.model)
            
            fact_params = count_fact_parameters(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            fact_analysis = analyze_shared_fact_parameters(self.model)
            trainable_analysis = analyze_all_trainable_parameters(self.model)
            print(f"Shared FacT applied with rank={fact_config['fact_rank']}, alpha={fact_config['fact_alpha']}, dropout={fact_config['fact_dropout']}")
            print(f"Target modules: {target_modules}")
            print(f"=== FacT Parameter Analysis ===")
            print(f"Shared FacT parameters: {fact_params:,}")
            print(f"  - Shared U/V parameters: {fact_analysis['shared_u_params'] + fact_analysis['shared_v_params']:,}")
            print(f"  - Layer-specific T parameters: {fact_analysis['layer_t_params']:,}")
            print(f"  - Shared ratio: {fact_analysis['shared_ratio']*100:.1f}%")
            print(f"=== All Trainable Parameter Analysis ===")
            print(f"Total trainable parameters: {trainable_analysis['total_trainable']:,}")
            print(f"  - FacT parameters: {trainable_analysis['fact_params']:,}")
            print(f"  - Embedding parameters: {trainable_analysis['embedding_params']:,}")
            print(f"  - LM head parameters: {trainable_analysis['lm_head_params']:,}")
            print(f"  - Other parameters: {trainable_analysis['other_params']:,}")
            print(f"  - FacT ratio in trainable: {trainable_analysis['fact_ratio']*100:.2f}%")
            print(f"Total model parameters: {total_params:,}")
            print(f"Shared FacT ratio: {fact_params/total_params*100:.2f}%")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
        
        # Apply LoRA if configured (only if FacT is not used)
        elif lora_config is not None:
            target_modules = lora_config.get("target_modules", ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.model = get_peft_model(
                self.model,
                peft_config=LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_config["lora_rank"],
                    lora_alpha=lora_config["lora_alpha"],
                    lora_dropout=lora_config["lora_dropout"],
                    target_modules=target_modules,
                    bias="none",
                )
            )
            print(f"LoRA applied with rank={lora_config['lora_rank']}, alpha={lora_config['lora_alpha']}, dropout={lora_config['lora_dropout']}")
            print(f"Target modules: {target_modules}")
                
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id  
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Store gradient checkpointing state for dynamic control
        self.gradient_checkpointing_enabled = gradient_checkpointing

    def generate(
        self,
        prompts: List[str],
        limitation: int = 1024,
        temperature: float = 1.0,
        verbose: bool = False
    ) -> Union[List[str], Tuple[List[str], torch.Tensor, torch.Tensor]]:
        """Generate text completions based on the provided prompts."""
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
        
        # Move inputs to model's device
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
            masks = torch.zeros_like(indices, dtype=torch.bool)
            masks[:, length:] = True
            masks[indices == self.tokenizer.pad_token_id] = False

        if verbose:
            return completions, indices, masks
        else:
            return completions


    def save(self, path: str) -> None:
        """Smart saving method that handles FacT adapters, LoRA adapters, and full models."""
        os.makedirs(path, exist_ok=True)
        
        # Always save the tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Only save adapter configuration if adapters are actually used
        if self.fact_config is not None or self.lora_config is not None:
            adapter_config = {
                "original_model_path": self.original_model_path,
                "fact_config": self.fact_config,
                "lora_config": self.lora_config,
            }
            with open(os.path.join(path, "adapter_config.json"), "w") as f:
                json.dump(adapter_config, f, indent=2)
            print(f"Adapter configuration saved to {path}/adapter_config.json")
        
        # FacT Mode - Save only adapter weights
        if self.fact_config is not None:
            fact_state_dict = {}
            for name, param in self.model.named_parameters():
                if 'fact_' in name:
                    fact_state_dict[name] = param.detach().cpu()
            
            if fact_state_dict:
                torch.save(fact_state_dict, os.path.join(path, "fact_adapter.bin"))
                total_params = sum(p.numel() for p in fact_state_dict.values())
                print(f"FacT adapter saved: {total_params:,} parameters")
            else:
                print("Warning: No FacT parameters found!")
        
        # LoRA/Full model - Use standard save
        else:
            self.model.save_pretrained(path)
            print("Model saved using standard method")


if __name__ == "__main__":
    # path = "Qwen/Qwen2.5-3B-Instruct"
    # path = "ministral/Ministral-3b-instruct"
    path = "meta-llama/Llama-3.2-3B-Instruct"
    model = LanguageModel(path)

    prompts = [
        "Who are you?",
        "What is the capital of France?",
        "What is the square root of 16?",
    ]
    completions, indices, masks = model.generate(
        prompts,
        limitation=256,
        temperature=1.0,
        verbose=True
    )
    log_probs = model.compute_log_probs(indices)

    print("The shape of indices:", indices.shape)
    print("The shape of masks:", masks.shape)
    print("The shape of log_probs:", log_probs.shape)
    print("Indices: ", indices)
    print("Masks: ", masks)
    print("Probabilities: ", log_probs)
    for prompt, completion in zip(prompts, completions):
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
