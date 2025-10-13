from vllm import LLM, SamplingParams
try:
    # Preferred import path per latest docs
    from vllm.lora.request import LoRARequest  # type: ignore
except Exception:
    try:
        # Fallback older path
        from vllm import LoRARequest  # type: ignore
    except Exception:
        LoRARequest = None  # type: ignore

# Base model and LoRA adapter path
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER_DIR = "/workspace/checkpoints/qwen_lora/Qwen2.5-3B-Instruct_final_step_70"
ADAPTER_NAME = "qwen_lora"


if __name__ == "__main__":
    print(f"[demo] Loading base model: {BASE_MODEL}")
    engine = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        enable_lora=True,
    )
    if LoRARequest is None:
        print("[demo] WARNING: vLLM LoRARequest not available; cannot attach LoRA at request time.")

    # Simple prompt for a quick sanity check
    prompts = [
        "You are a helpful math assistant. Solve the problem and answer with only the final number. 1 + 2 = ?",
    ]

    sampling = SamplingParams(
        temperature=1.0,
        top_k=50,
        top_p=0.7,
        max_tokens=128,
        n=1,
    )

    gen_kwargs = {}
    if LoRARequest is not None:
        try:
            # LoRARequest(name, lora_int_id, lora_path)
            gen_kwargs["lora_request"] = LoRARequest(
                lora_name=ADAPTER_NAME,
                lora_int_id=1,
                lora_path=LORA_ADAPTER_DIR,
            )
            print(f"[demo] Using LoRARequest(name='{ADAPTER_NAME}', id=1, path='{LORA_ADAPTER_DIR}')")
        except Exception as e:
            print(f"[demo] LoRARequest unusable: {e}")

    print("[demo] Generating...")
    outputs = engine.generate(prompts, sampling, **gen_kwargs)

    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        print(f"\n[demo] Prompt {i} output (truncated):\n{text[:400]}")

    print("[demo] Done.")