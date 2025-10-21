import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from utils.model import LanguageModel

"""
export VLLM_USE_V1=0
This script is used to evaluate the model on the MATH, MBPP, and GSM8K datasets.
CUDA_VISIBLE_DEVICES=7 python zero_shot_eval.py --model Qwen/Qwen2.5-3B-Instruct
CUDA_VISIBLE_DEVICES=6 python zero_shot_eval.py --model HuggingFaceTB/SmolLM3-3B
"""


def load_math_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "solution" not in item and "answer" in item:
            item["solution"] = item["answer"]
    return data


def load_mbpp_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_gsm8k_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate_dataset(model: LanguageModel, prompts: List[str], targets: List[str], task: str,
                     batch_size: int = 256, max_len: int = 1024, temperature: float = 0.7) -> Tuple[int, int]:
    if task == "math":
        from verifiers.verifier_math import math_reward_function as reward_fn
        def scorer(output: str, target: str) -> int:
            try:
                return int(bool(reward_fn(output, target)))
            except Exception:
                return 0
    elif task == "code":
        from verifiers.verifier_coding import verify_answer as verify
        def scorer(output: str, target: str) -> int:
            try:
                return 1 if verify(output, target).get("correct", False) else 0
            except Exception:
                return 0
    elif task == "gsm":
        from utils.gsm8k import gsm8k_reward_function as reward_fn
        def scorer(output: str, target: str) -> int:
            try:
                return int(bool(reward_fn(output, target)))
            except Exception:
                return 0
    else:
        raise ValueError(f"Unknown task: {task}")

    correct = 0
    total = 0
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]
        # Use vLLM generation with temperature 0.7
        outputs = model.generate(
            prompts=batch_prompts,
            limitation=max_len,
            temperature=temperature,
            verbose=False,
        )
        for out, tgt in zip(outputs, batch_targets):
            correct += scorer(out, tgt)
            total += 1
    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation with vLLM (pass@1)")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    args = parser.parse_args()

    # Data roots align with training
    data_root = Path("/workspace/data")
    math_path = data_root / "MATH" / "test.json"
    mbpp_path = data_root / "MBPP" / "test.json"
    gsm8k_path = data_root / "GSM8k" / "test.json"

    # Build model with vLLM for fast decoding
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = LanguageModel(
        model_path=args.model,
        target_device=device,
        torch_dtype=dtype,
        attn_impl="flash",
        fact_config=None,
        lora_config=None,
        gradient_checkpointing=False,
        use_vllm=torch.cuda.is_available(),
        vllm_gpu_memory_utilization=0.6,
        vllm_gpu_id=0,
        vllm_max_model_len=32768,
    )

    # Load datasets
    math_data = load_math_json(str(math_path)) if math_path.exists() else []
    mbpp_data = load_mbpp_json(str(mbpp_path)) if mbpp_path.exists() else []
    gsm8k_data = load_gsm8k_json(str(gsm8k_path)) if gsm8k_path.exists() else []

    # Prepare prompts/targets
    math_prompts = [s["prompt"] for s in math_data]
    math_targets = [s.get("solution", s.get("answer")) for s in math_data]

    mbpp_prompts = [s["prompt"] for s in mbpp_data]
    mbpp_targets = [s["answer"] for s in mbpp_data]

    gsm_prompts = [s["prompt"] for s in gsm8k_data]
    gsm_targets = [s["answer"] for s in gsm8k_data]

    # Evaluate
    math_c, math_t = evaluate_dataset(model, math_prompts, math_targets, task="math") if math_prompts else (0, 0)
    mbpp_c, mbpp_t = evaluate_dataset(model, mbpp_prompts, mbpp_targets, task="code") if mbpp_prompts else (0, 0)
    gsm_c, gsm_t = evaluate_dataset(model, gsm_prompts, gsm_targets, task="gsm") if gsm_prompts else (0, 0)

    def fmt(acc_c: int, acc_t: int) -> str:
        return f"{(acc_c/acc_t):.4f}" if acc_t > 0 else "NA"

    print(f"[Zero-shot] Model: {args.model}")
    print(f"MATH pass@1: {fmt(math_c, math_t)} ({math_c}/{math_t})")
    print(f"MBPP pass@1: {fmt(mbpp_c, mbpp_t)} ({mbpp_c}/{mbpp_t})")
    print(f"GSM8K pass@1: {fmt(gsm_c, gsm_t)} ({gsm_c}/{gsm_t})")


if __name__ == "__main__":
    main()
