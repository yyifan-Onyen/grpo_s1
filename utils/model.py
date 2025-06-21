from typing import List, Dict, Tuple, Optional, Union
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class LanguageModel(object):
    """A wrapper class for language models from HuggingFace."""
    def __init__(
        self,
        model_path: str,
        target_device: str = "cuda",
        torch_dtype: str = "auto",
        attn_impl: str = "sdpa",
        lora_config: Optional[Dict] = None
    ):
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
        
        if lora_config is not None:
            self.model = get_peft_model(
                self.model,
                peft_config=LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_config["lora_rank"],
                    lora_alpha=lora_config["lora_alpha"],
                    lora_dropout=lora_config["lora_dropout"]
                )
            )
                
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id  
        self.pad_token_id = self.tokenizer.pad_token_id

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
        

    def compute_log_probs(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for the given token indices."""
        input_ids = indices.to(self.device)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids[attention_mask == 0] = 0

        input_logits = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )["logits"][:, :-1]

        log_probs = torch.gather(
            torch.log_softmax(input_logits, dim=-1),
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return log_probs


    def save(self, path: str) -> None:
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)


if __name__ == "__main__":
    # path = "Qwen/Qwen2.5-3B-Instruct"
    path = "ministral/Ministral-3b-instruct"
    # path = "meta-llama/Llama-3.2-3B-Instruct"
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
