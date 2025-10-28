from vllm import LLM, SamplingParams

def main():
    model = LLM(
        model="HuggingFaceTB/SmolLM3-3B",
        dtype="bfloat16",                 # 'torch_dtype' 已弃用，用 dtype
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        enable_lora=True,                 # 看到你启了 LoRA；和 chunked prefill 组合仍属试验性，属警告非错误
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = model.generate(["Hello, world!"], sampling)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    # 可选：显式声明 spawn，避免环境默认不一致
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()