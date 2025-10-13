import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import gc
import os
import ipdb
import math
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Optional
from grpo import update_policy, get_log_probs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import random


def _reward_worker(args):
    reward_type, response, target = args
    if reward_type == "math":
        from verifiers.verifier_math import math_reward_function
        return math_reward_function(response, target)
    if reward_type == "code":
        from verifiers.verifier_coding import verify_answer
        try:
            res = verify_answer(response, target)
            return 1 if res.get("correct", False) else 0
        except Exception:
            return 0
    if reward_type == "gsm":
        from utils.gsm8k import gsm8k_reward_function
        return gsm8k_reward_function(response, target)
    raise ValueError(f"Unsupported reward type: {reward_type}")
try:
    import wandb  # Optional logging
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False
from utils.model import LanguageModel
import copy
import json


# Fixed generation/decoding hyperparameters
MAX_GEN_LEN = 1024
TRAIN_TEMPERATURE = 1.0
VAL_TEMPERATURE = 0.7


def prepare_adapter_configs(args):
    """根据命令行参数准备适配器配置"""
    fact_config = None
    lora_config = None
    
    if args.adapter == "fact":
        fact_config = {
            "fact_rank": args.adapter_rank,
            "fact_alpha": args.adapter_alpha,
            "fact_dropout": args.adapter_dropout,
            "fact_scale": args.adapter_scale,
            "fact_target_modules": getattr(args, 'adapter_target_modules')
        }
    elif args.adapter == "lora":
        lora_config = {
            "lora_rank": args.adapter_rank,
            "lora_alpha": args.adapter_alpha,
            "lora_dropout": args.adapter_dropout,
            "target_modules": getattr(args, 'adapter_target_modules')
        }
    
    return fact_config, lora_config


def _rollout_model_batch(
    model_idx: int,
    model: "LanguageModel",
    tokenizer,
    name: str,
    device: str,
    questions: List[dict],
    num_answers: int,
    max_gen_len: int,
    temperature: float,
):
    per_question_eps: List[List[dict]] = [[] for _ in range(len(questions))]

    if len(questions) == 0:
        return per_question_eps

    prompts: List[str] = []
    q_refs: List[int] = []
    for q_idx, sample in enumerate(questions):
        for _ in range(num_answers):
            prompts.append(sample["prompt"])
            q_refs.append(q_idx)

    if getattr(model, "use_vllm", False) and getattr(model, "vllm_engine", None) is not None and getattr(model, "lora_config", None) is not None:
        try:
            model.sync_lora_to_vllm(adapter_name=f"lora_{name}")
        except Exception:
            pass

    if len(prompts) == 0:
        return per_question_eps

    decode_chunk_size = min(len(prompts), 256)
    for start in range(0, len(prompts), decode_chunk_size):
        end = min(start + decode_chunk_size, len(prompts))
        chunk_prompts = prompts[start:end]
        chunk_qrefs = q_refs[start:end]
        completions, indices, masks, vllm_lps = model.generate(
            prompts=chunk_prompts,
            limitation=max_gen_len,
            temperature=temperature,
            verbose=True,
            return_log_probs=bool(getattr(model, "use_vllm", False)),
        )
        for j, result in enumerate(completions):
            q_idx = chunk_qrefs[j]
            full_indices = indices[j]
            mask = masks[j]
            if torch.cuda.is_available() and full_indices.is_cuda:
                try:
                    torch.cuda.synchronize(full_indices.device)
                except Exception:
                    torch.cuda.synchronize()
            fi_cpu = full_indices.detach().contiguous().to("cpu")
            mk_cpu = mask.detach().contiguous().to("cpu")
            prefix_token_ids = fi_cpu[~mk_cpu].tolist()
            generated_token_ids = fi_cpu[mk_cpu].tolist()
            old_lp = None
            if isinstance(vllm_lps, list) and j < len(vllm_lps) and vllm_lps[j] is not None:
                try:
                    old_lp = vllm_lps[j].detach().cpu().tolist()
                except Exception:
                    old_lp = None
            episode = {
                "prefix_token_ids": prefix_token_ids,
                "generated_token_ids": generated_token_ids,
                "reward": None,
                "completion": result,
                "model_name": name,
                "model_idx": model_idx,
                "old_log_probs": old_lp,
            }
            per_question_eps[q_idx].append(episode)
        del completions, indices, masks, vllm_lps
        torch.cuda.empty_cache()
        gc.collect()

    return per_question_eps


def _parallel_rollout_all_models(
    models: List["LanguageModel"],
    tokenizers,
    model_names: List[str],
    target_devices: List[str],
    questions: List[dict],
    num_answers: int,
    max_gen_len: int,
    temperature: float,
    reward_fn,
):
    if len(questions) == 0:
        return []

    aggregated: List[List[dict]] = [[] for _ in range(len(questions))]

    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {
            pool.submit(
                _rollout_model_batch,
                model_idx,
                model,
                tokenizers[model_idx],
                model_names[model_idx],
                target_devices[model_idx],
                questions,
                num_answers,
                max_gen_len,
                temperature,
            ): model_idx
            for model_idx, model in enumerate(models)
        }
        for fut in as_completed(futures):
            per_question_eps = fut.result()
            for q_idx, eps in enumerate(per_question_eps):
                # 不在此处计算 reward，避免线程环境触发 signal.alarm()
                # 统一在主线程/进程池阶段计算并回填
                aggregated[q_idx].extend(eps)

    return aggregated


def main():
    parser = ArgumentParser(description="GRPO Multi-Model Training")
    
    # 模型配置
    parser.add_argument("--models", nargs="+", required=True,
                       help="List of pretrained model paths")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                       help="Model dtype")
    
    # 适配器配置
    parser.add_argument("--adapter", choices=["none", "lora", "fact"], default="none",
                       help="Type of adapter to use")
    parser.add_argument("--adapter-rank", type=int, default=16,
                       help="Adapter rank")
    parser.add_argument("--adapter-alpha", type=int, default=32,
                       help="Adapter alpha")
    parser.add_argument("--adapter-dropout", type=float, default=0.05,
                       help="Adapter dropout")
    parser.add_argument("--adapter-scale", type=float, default=1.0,
                       help="Adapter scale (for FACT)")
    parser.add_argument("--adapter-target-modules", nargs="+", 
                       default=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       help="Target modules for adapter")
    
    # 训练配置
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Number of questions per batch for policy update")
    # 固定推理参数：训练温度=1.0，验证温度=0.7，生成长度=1024（不再通过 args 配置）
    parser.add_argument("--num-answers", type=int, default=4,
                       help="Number of answers per question")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # PPO分批参数
    parser.add_argument("--ppo-mini-batch-size", type=int, default=16,
                       help="Mini-batch size (episodes) used inside each PPO update. 0 表示使用一次性全部 episodes。")
    parser.add_argument("--ppo-micro-batch-size", type=int, default=8,
                       help="Micro-batch size per forward/backward pass (用于梯度累积/避免 OOM)。")
    
    # GRPO参数
    parser.add_argument("--epsilon-low", type=float, default=0.2,
                       help="PPO clipping lower bound")
    parser.add_argument("--epsilon-high", type=float, default=0.2,
                       help="PPO clipping upper bound")
    parser.add_argument("--beta", type=float, default=0.003,
                       help="KL divergence coefficient (default aligns with VERL)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--loss-type", default="grpo",
                       help="Loss type")
    parser.add_argument("--ppo-epochs", type=int, default=2,
                       help="Number of PPO epochs per batch (VERL-style multi-pass)")
    parser.add_argument("--loss-aggregation", choices=["token-mean", "episode-mean"], default="token-mean",
                       help="Loss reduction across tokens vs episodes (token-mean matches VERL)")
    parser.add_argument("--advantage-clip", type=float, default=0.0,
                       help="Clip advantages to [-clip, clip]; 0 disables clipping")
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages", action="store_false",
                       help="Disable per-group advantage whitening")
    parser.set_defaults(normalize_advantages=True)
    
    # 数据/任务配置（通过参数指定数据根路径与任务类型：math 或 code）
    parser.add_argument("--task-type", choices=["math", "code"], default="math",
                       help="Task type to determine verifier and dataset schema")
    parser.add_argument("--data-root", type=str,
                       default="/home/local/PARTNERS/yz646/grpo_s1/data",
                       help="Root directory containing datasets (expects MATH/ or CODE/ subfolders)")
    # 阈值模式（选择跨模型样本）
    parser.add_argument("--mode", choices=["random", "threshold", "weight"], default="threshold",
                       help="Mixing strategy: 'random' half self/half others (random); 'threshold' prefer self positives then other-model positives; 'weight' = threshold sampling + per-batch LR adaptation.")
    parser.add_argument("--reward-threshold", type=float, default=1.0,
                       help="Reward threshold for accepting cross-model episodes when --mode=threshold (e.g., 1.0 for correct-only).")
    # 验证配置（以策略更新步 step 为单位）
    parser.add_argument("--val-interval", type=int, default=2,
                       help="Validation frequency in policy update steps (step)")
    parser.add_argument("--val-batch-size", type=int, default=64,
                       help="Validation decode batch size (number of prompts decoded in parallel)")
    
    # 解码/加速选项
    parser.add_argument("--use-vllm", action="store_true", default=False,
                       help="Use vLLM engine for fast decoding (generation). Training still uses HF model.")
    parser.add_argument("--vllm-gpu-mem", type=float, default=0.7,
                       help="vLLM gpu_memory_utilization ratio (0-1). Lower if startup OOM.")
    parser.add_argument("--vllm-gpu", type=int, default=None,
                       help="vLLM target GPU index. If CUDA_VISIBLE_DEVICES is set, it's local index; otherwise absolute GPU id.")
    parser.add_argument("--vllm-gpus", nargs="+", type=int, default=None,
                       help="Per-model vLLM GPU local indices (within CUDA_VISIBLE_DEVICES). Provide N entries for N models.")
    parser.add_argument("--vllm-max-model-len", type=int, default=32768,
                       help="Max model length for vLLM to cap KV cache size and avoid OOM (e.g., 32768).")
    parser.add_argument("--rollout-batch-size", type=int, default=64,
                       help="Decode batch size for rollout (number of prompts decoded together across questions).")

    
    # 输出配置
    parser.add_argument("--ckpt-dir", default="checkpoints/grpo_direct",
                       help="Checkpoint directory")
    parser.add_argument("--ckpt-interval", type=int, default=3000,
                       help="Checkpoint save interval")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum number of policy update steps (step) to run. If unset, iterate over full dataset.")
    parser.add_argument("--save-final", action="store_true", default=True,
                       help="Save a final checkpoint at the end of training")
    parser.add_argument("--no-save-final", dest="save_final", action="store_false",
                       help="Do not save a final checkpoint at the end of training")
    parser.add_argument("--use-ref-model", action="store_true", default=True,
                       help="Use reference model for KL divergence")
    parser.add_argument("--no-ref-model", dest="use_ref_model", action="store_false",
                       help="Don't use reference model")
    
    # 日志/追踪配置
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom Weights & Biases run name; if unset, auto-generated")
    
    args = parser.parse_args()

    global _WANDB_AVAILABLE

    # Initialize Weights & Biases if available (user will login via CLI)
    if _WANDB_AVAILABLE:
        try:
            # 尝试使用环境变量或内置默认值登录 wandb（默认值仅为方便本地调试）
            _WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "1c01f395e45cd03487bdb9c72cbefe7cdef54426")
            try:
                if _WANDB_API_KEY:
                    wandb.login(key=_WANDB_API_KEY, relogin=True)
            except Exception:
                pass
            _run_name = args.run_name or f"GRPO_{'_'.join([Path(p).name for p in (args.models or [])])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="grpo-s1", name=_run_name, config=vars(args))
        except Exception as _e:
            print(f"[wandb] init failed: {_e}. Continue without wandb.")
            _WANDB_AVAILABLE = False
        else:
            # Ensure all metrics use training step as global x-axis
            try:
                wandb.define_metric("train/step")
                wandb.define_metric("train/*", step_metric="train/step")
                wandb.define_metric("val/*", step_metric="train/step")
            except Exception as _e:
                print(f"[wandb] define_metric failed: {_e}")
    
    # 准备适配器配置
    fact_config, lora_config = prepare_adapter_configs(args)
    
    pretrained_model_paths = [Path(path) for path in args.models]
    

    max_supported_models = 3
    num_models = min(len(pretrained_model_paths), max_supported_models)
    pretrained_model_paths = pretrained_model_paths[:num_models]
    

    is_multi_model_mode = num_models > 1
    selected_mode = getattr(args, "mode", "threshold")
    if selected_mode == "random" and not is_multi_model_mode:
        print("[mode] '--mode=random' requires at least two models; falling back to 'threshold'.")
        selected_mode = "threshold"
    setattr(args, "mode", selected_mode)
    mode_description = f"{'Multi-model collaborative' if is_multi_model_mode else 'Single-model basic'} GRPO"
    print(f"Running {mode_description} with {num_models} model(s)")
    print(f"Mixing mode: {selected_mode}")
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    if num_gpus < num_models:
        print(f"Error: Need at least {num_models} GPUs for manual assignment, but only {num_gpus} available")
        return
    

    models = []
    ref_models = []  # Reference models for KL divergence
    model_names = [Path(path).name for path in pretrained_model_paths]
    target_devices = [f"cuda:{i}" for i in range(num_models)]
    print(target_devices)
    # exit()
    
    for i, (model_path, name, device) in enumerate(zip(pretrained_model_paths, model_names, target_devices)):
        print(f"Loading {name} on {device}...")
        
        # Load main model
        # Choose per-model vLLM GPU id if provided
        per_model_vllm_gpu = None
        try:
            if getattr(args, "vllm_gpus", None) is not None and len(args.vllm_gpus) > i:
                per_model_vllm_gpu = int(args.vllm_gpus[i])
            else:
                per_model_vllm_gpu = getattr(args, "vllm_gpu", None)
        except Exception:
            per_model_vllm_gpu = getattr(args, "vllm_gpu", None)

        model = LanguageModel(
            model_path=str(model_path),
            target_device=device,
            torch_dtype=dtype,
            attn_impl="flash",
            fact_config=fact_config,
            lora_config=lora_config,
            gradient_checkpointing=True,
            use_vllm=getattr(args, "use_vllm", False),
            vllm_gpu_memory_utilization=getattr(args, "vllm_gpu_mem", None),
            vllm_gpu_id=per_model_vllm_gpu,
            vllm_max_model_len=getattr(args, "vllm_max_model_len", None),
        )
        model.model.train()
        models.append(model)
        
        # Load reference model (copy of initial model for KL divergence)
        if args.use_ref_model:
            print(f"  Loading reference model for {name}...")
            ref_model = LanguageModel(
                model_path=str(model_path),
                target_device=device,
                torch_dtype=dtype,
                attn_impl="flash",
                fact_config=None,  # 参考模型不使用适配器
                lora_config=None,  # 参考模型不使用适配器
                use_vllm=False,
            )
            ref_model.model.eval()  # Keep reference model in eval mode
            # Freeze reference model parameters
            for param in ref_model.model.parameters():
                param.requires_grad = False
            ref_models.append(ref_model)
        else:
            ref_models.append(None)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  {name} loaded successfully on {device}")

    tokenizers = [model.tokenizer for model in models]

    # FacT + vLLM：训练开始前进行一次初始合并并启动 vLLM 引擎用于随后的 rollout
    try:
        for model, name in zip(models, model_names):
            if getattr(args, "use_vllm", False) and getattr(model, "fact_config", None) is not None:
                ok = model.refresh_vllm_merged_engine(
                    gpu_memory_utilization=getattr(args, "vllm_gpu_mem", 0.6),
                    max_model_len=getattr(args, "vllm_max_model_len", None),
                )
                print(f"[FacT] Initial merged vLLM for {name}: {'OK' if ok else 'FAIL'}")
    except Exception as _e:
        print(f"[FacT] Initial merged vLLM refresh error: {_e}")

    # 依据 task-type 解析数据路径（由 data-root + 子目录 推断）
    _data_root = args.data_root
    if args.task_type == "math":
        train_path = f"{_data_root}/MATH/train.json"
        test_path = f"{_data_root}/MATH/test.json"
    elif args.task_type == "code":
        # 使用 MBPP 作为代码数据集
        train_path = f"{_data_root}/MBPP/train.json"
        test_path = f"{_data_root}/MBPP/test.json"
    else:
        print(f"Unsupported task type: {args.task_type}")
        return

    # 加载数据集与选择对应奖励函数
    def load_math_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if "solution" not in item and "answer" in item:
                item["solution"] = item["answer"]
        return data


    def load_mbpp_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # MBPP 数据包含 prompt 与 answer（answer 中是断言，可作为 checker 使用）
        missing = [i for i, it in enumerate(data) if "prompt" not in it or "answer" not in it]
        if len(missing) > 0:
            print(f"Warning: {len(missing)} samples missing 'prompt' or 'answer' in {path}")
        return data

    if args.task_type == "math":
        train_dataset = load_math_json(train_path)
        test_dataset = load_math_json(test_path)
        from verifiers.verifier_math import math_reward_function as _math_reward_function
        def reward_fn(response, sample):
            return _math_reward_function(response, sample["solution"])
    elif args.task_type == "code":
        train_dataset = load_mbpp_json(train_path)
        test_dataset = load_mbpp_json(test_path)
        from verifiers.verifier_coding import verify_answer as _code_verify_answer
        def reward_fn(response, sample):
            try:
                # MBPP 使用 answer 字段作为 checker（内含断言）
                return 1 if _code_verify_answer(response, sample["answer"])['correct'] else 0
            except Exception as _e:
                print(f"[verifier] code verify failed: {_e}")
                return 0
    else:
        print(f"Unsupported task type: {args.task_type}")
        return

    # 始终准备用于验证的测试集：MATH、MBPP、GSM8K
    math_test_path = f"{_data_root}/MATH/test.json"
    mbpp_test_path = f"{_data_root}/MBPP/test.json"
    gsm8k_test_path = f"{_data_root}/GSM8k/test.json"
    math_test_dataset = load_math_json(math_test_path)
    mbpp_test_dataset = load_mbpp_json(mbpp_test_path)
    def load_gsm8k_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            missing = [i for i, it in enumerate(data) if "prompt" not in it or "answer" not in it]
            if len(missing) > 0:
                print(f"Warning: {len(missing)} samples missing 'prompt' or 'answer' in {path}")
            return data
        except Exception as _e:
            print(f"Warning: failed to load GSM8K test set at {path}: {_e}")
            return []
    gsm8k_test_dataset = load_gsm8k_json(gsm8k_test_path)

    def run_validation(step_idx: int):
        if not _WANDB_AVAILABLE:
            return
        reward_executor = ProcessPoolExecutor(
            max_workers=max(1, len(models)), mp_context=mp.get_context("spawn")
        )
        try:
            # 导入验证函数（数学、代码、GSM8K）
            from verifiers.verifier_math import math_reward_function as _val_math_reward
            from verifiers.verifier_coding import verify_answer as _val_code_verify
            try:
                from utils.gsm8k import gsm8k_reward_function as _val_gsm8k_reward
            except Exception:
                _val_gsm8k_reward = None
            _max_s = None  # Always evaluate full dataset
            _val_batch = args.val_batch_size  # 批量验证的batch大小

            def _validate_one(_model, _name, _device):
                # Ensure vLLM engine uses current LoRA weights if applicable
                try:
                    if getattr(_model, "use_vllm", False) and getattr(_model, "vllm_engine", None) is not None and getattr(_model, "lora_config", None) is not None:
                        ok = _model.sync_lora_to_vllm(adapter_name=f"lora_{_name}")
                        if not ok:
                            print(f"[VAL] {_name}: vLLM LoRA sync failed; consider disabling vLLM for validation.")
                except Exception as _se:
                    print(f"[VAL] {_name}: LoRA sync error: {_se}")

                # 优先使用持久化的 vLLM 引擎（线程安全），必要时退回一次性合并评估
                def _gen_eval(_prompts: List[str]):
                    try:
                        if getattr(args, "use_vllm", False):
                            if getattr(_model, "fact_config", None) is not None and getattr(_model, "vllm_engine", None) is not None:
                                return _model.generate(prompts=_prompts, limitation=MAX_GEN_LEN, temperature=VAL_TEMPERATURE, verbose=False)
                            elif getattr(_model, "fact_config", None) is not None:
                                # FacT 无持久引擎时的兜底方案
                                return _model.eval_with_vLLM_on_merged(_prompts, limitation=MAX_GEN_LEN, temperature=VAL_TEMPERATURE)
                    except Exception:
                        pass
                    return _model.generate(prompts=_prompts, limitation=MAX_GEN_LEN, temperature=VAL_TEMPERATURE, verbose=False)

                def _extract_completions(gen_out):
                    # 兼容 generate 返回 (completions, indices, masks, log_probs) 的情况
                    if isinstance(gen_out, tuple):
                        try:
                            return gen_out[0]
                        except Exception:
                            return []
                    return gen_out

                # 1) 验证 MATH（使用数学验证）
                math_correct = 0
                math_total = 0
                _math_samples = math_test_dataset[:_max_s] if _max_s else math_test_dataset
                for i in range(0, len(_math_samples), _val_batch):
                    batch = _math_samples[i:i+_val_batch]
                    prompts = [s["prompt"] for s in batch]
                    gen_out = _gen_eval(prompts)
                    completions = _extract_completions(gen_out)
                    for j, completion in enumerate(completions):
                        sample = batch[j]
                        target_solution = sample.get("solution", sample.get("answer"))
                        try:
                            score = reward_executor.submit(
                                _reward_worker, ("math", completion, target_solution)
                            ).result()
                        except Exception as _me:
                            print(f"[verifier] MATH verify failed: {_me}")
                            score = 0
                        math_correct += int(score)
                        math_total += 1
                math_acc = math_correct / max(1, math_total)

                # 2) 验证 MBPP（使用代码验证，MBPP 的 answer 作为 checker）
                mbpp_correct = 0
                mbpp_total = 0
                _mbpp_samples = mbpp_test_dataset[:_max_s] if _max_s else mbpp_test_dataset
                for i in range(0, len(_mbpp_samples), _val_batch):
                    batch = _mbpp_samples[i:i+_val_batch]
                    prompts = [s["prompt"] for s in batch]
                    gen_out = _gen_eval(prompts)
                    completions = _extract_completions(gen_out)
                    for j, completion in enumerate(completions):
                        sample = batch[j]
                        try:
                            result = _val_code_verify(completion, sample["answer"])  # 使用 MBPP 的断言作为 checker
                            score = 1 if result.get("correct", False) else 0
                        except Exception as _ve:
                            print(f"[verifier] MBPP verify failed: {_ve}")
                            score = 0
                        mbpp_correct += int(score)
                        mbpp_total += 1
                mbpp_acc = mbpp_correct / max(1, mbpp_total)

                # 3) 验证 GSM8K（使用数学式验证器）
                gsm8k_acc = None
                gsm8k_total = 0
                if _val_gsm8k_reward is not None and gsm8k_test_dataset:
                    gsm8k_correct = 0
                    gsm8k_total = 0
                    _gsm_samples = gsm8k_test_dataset[:_max_s] if _max_s else gsm8k_test_dataset
                    for i in range(0, len(_gsm_samples), _val_batch):
                        batch = _gsm_samples[i:i+_val_batch]
                        prompts = [s["prompt"] for s in batch]
                        gen_out = _gen_eval(prompts)
                        completions = _extract_completions(gen_out)
                        for j, completion in enumerate(completions):
                            sample = batch[j]
                            try:
                                score = reward_executor.submit(
                                    _reward_worker, ("gsm", completion, sample["answer"])
                                ).result()
                            except Exception as _ge:
                                print(f"[verifier] GSM8K verify failed: {_ge}")
                                score = 0
                            gsm8k_correct += int(score)
                            gsm8k_total += 1
                    gsm8k_acc = gsm8k_correct / max(1, gsm8k_total)

                return {
                    "name": _name,
                    "math_acc": math_acc,
                    "math_total": math_total,
                    "mbpp_acc": mbpp_acc,
                    "mbpp_total": mbpp_total,
                    "gsm8k_acc": gsm8k_acc,
                    "gsm8k_total": gsm8k_total,
                }

            results = []
            # Run validation sequentially per model to avoid signal/thread interaction
            for model, name, device in zip(models, model_names, target_devices):
                try:
                    results.append(_validate_one(model, name, device))
                except Exception as _e:
                    print(f"[VAL] worker failed: {_e}")

            # 统一日志输出，避免多线程同时写 wandb
            for r in results:
                name = r["name"]
                wandb.log({f"val/pass@1_math/{name}": float(r["math_acc"]), "train/step": step_idx}, step=step_idx)
                print(f"[VAL] step {step_idx} | {name} [MATH]: pass@1={r['math_acc']:.4f} on {r['math_total']} samples")

                wandb.log({f"val/pass@1_mbpp/{name}": float(r["mbpp_acc"]), "train/step": step_idx}, step=step_idx)
                print(f"[VAL] step {step_idx} | {name} [MBPP]: pass@1={r['mbpp_acc']:.4f} on {r['mbpp_total']} samples")

                if r.get("gsm8k_acc") is not None:
                    wandb.log({f"val/pass@1_gsm8k/{name}": float(r["gsm8k_acc"]), "train/step": step_idx}, step=step_idx)
                    print(f"[VAL] step {step_idx} | {name} [GSM8K]: pass@1={r['gsm8k_acc']:.4f} on {r['gsm8k_total']} samples")
        except Exception as _e:
            print(f"[val] failed: {_e}")
        finally:
            reward_executor.shutdown(wait=True)

    optimizers = []
    for model in models:
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=args.lr
        )
        optimizers.append(optimizer)

    for i, (model, name, device) in enumerate(zip(models, model_names, target_devices)):
        print(f"Model {i+1} ({name}) on {device}: {type(model.model)}")
        params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"  Parameters: {params:,}")

    batch_step = 0  # Batch (update) counter
    planned_env_steps = len(train_dataset) * args.epochs
    batch_size = max(1, args.batch_size)  # Questions per PPO update
    planned_updates = math.ceil(planned_env_steps / batch_size)
    max_updates = args.max_steps if args.max_steps is not None else planned_updates

    # GRPO training parameters
    num_completions_per_prompt = args.num_answers
    epsilon_low = args.epsilon_low
    epsilon_high = args.epsilon_high
    beta = args.beta  # KL regularization
    loss_type = args.loss_type
    ppo_mini_batch_size = max(0, args.ppo_mini_batch_size)
    ppo_micro_batch_size = max(1, args.ppo_micro_batch_size)
    ppo_epochs = max(1, args.ppo_epochs)
    loss_aggregation = args.loss_aggregation
    advantage_clip = args.advantage_clip if args.advantage_clip and args.advantage_clip > 0 else None
    normalize_advantages = bool(args.normalize_advantages)
    
    # Per-batch LR adaptation (enabled only when mode == 'weight')
    lr_adapt_enabled = (getattr(args, "mode", "threshold") == "weight")
    base_lrs = [args.lr for _ in range(num_models)]
    # EMA of self rollout mean reward per model
    lr_adapt_ema = [None for _ in range(num_models)]
    # Holds the latest batch's self rollout mean reward per model (reset after each update)
    batch_self_reward_mean = [None for _ in range(num_models)]

    # Storage for old log probabilities (for PPO clipping)
    old_log_probs_storage = [None] * num_models
    
    # Batch episode storage for each model
    batch_episodes_storage = [[] for _ in range(num_models)]
    
    training_done = False
    decode_buffer: List[dict] = [] if not is_multi_model_mode else None
    mm_decode_buffer: List[dict] = [] if is_multi_model_mode else None
    rollout_batch_size = max(1, getattr(args, "rollout_batch_size", 64))
    # Global reward executor for training-time reward computation
    train_reward_executor = ProcessPoolExecutor(
        max_workers=max(1, num_models), mp_context=mp.get_context("spawn")
    )
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1} of {args.epochs}")
        for question_idx, question in enumerate(train_dataset):
            if training_done:
                break
            if max_updates is not None and batch_step >= max_updates:
                training_done = True
                break
            print(
                f"(update {batch_step}/{max_updates}) - "
                f"Question {question_idx + 1}/{len(train_dataset)}"
            )
            
            if is_multi_model_mode and mm_decode_buffer is not None:
                mm_decode_buffer.append(question)
                should_decode_now = (
                    len(mm_decode_buffer) >= batch_size or 
                    question_idx == len(train_dataset) - 1
                )
                if should_decode_now:
                    decode_start = time.perf_counter()
                    per_question_eps = _parallel_rollout_all_models(
                        models=models,
                        tokenizers=tokenizers,
                        model_names=model_names,
                        target_devices=target_devices,
                        questions=mm_decode_buffer,
                        num_answers=num_completions_per_prompt,
                    max_gen_len=MAX_GEN_LEN,
                    temperature=TRAIN_TEMPERATURE,
                        reward_fn=reward_fn,
                    )
                    decode_duration = time.perf_counter() - decode_start

                    # Compute rewards in process pool to avoid signal/alarm issues
                    if args.task_type in ("math", "code"):
                        reward_type = "math" if args.task_type == "math" else "code"
                        reward_tasks = []
                        ep_refs = []  # (q_idx, ep)
                        for q_idx, eps in enumerate(per_question_eps):
                            target = mm_decode_buffer[q_idx].get("solution", mm_decode_buffer[q_idx].get("answer"))
                            for ep in eps:
                                response = ep.get("completion", "")
                                reward_tasks.append((reward_type, response, target))
                                ep_refs.append((q_idx, ep))
                        if reward_tasks:
                            try:
                                results = list(train_reward_executor.map(_reward_worker, reward_tasks))
                                for (q_idx, ep), score in zip(ep_refs, results):
                                    ep["reward"] = score
                            except Exception as _tre:
                                print(f"[reward] training reward pool failed: {_tre}")
                                # Fallback to inline compute (serial)
                                for (q_idx, ep) in ep_refs:
                                    try:
                                        ep["reward"] = reward_fn(ep.get("completion"), mm_decode_buffer[q_idx])
                                    except Exception:
                                        ep["reward"] = 0

                    assemble_start = time.perf_counter()
                    per_model_assembly = {}

                    for model_idx in range(num_models):
                        tokenizer = tokenizers[model_idx]
                        name = model_names[model_idx]
                        model_episodes: List[dict] = []
                        per_model_start = time.perf_counter()
                        # Track self rollout reward mean for this model in this flush
                        _self_reward_sum = 0.0
                        _self_reward_cnt = 0

                        # Batch build own prefixes for all questions in this flush
                        try:
                            conv_texts = [
                                tokenizer.apply_chat_template(
                                    [{"role": "user", "content": sample["prompt"]}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                                for sample in mm_decode_buffer
                            ]
                        except Exception:
                            conv_texts = []

                        try:
                            _tok = tokenizer(conv_texts, add_special_tokens=False, return_tensors=None)
                            own_prefix_list = _tok["input_ids"] if isinstance(_tok, dict) else [[] for _ in conv_texts]
                        except Exception:
                            own_prefix_list = []
                            for sample in mm_decode_buffer:
                                try:
                                    conv = tokenizer.apply_chat_template(
                                        [{"role": "user", "content": sample["prompt"]}],
                                        tokenize=False,
                                        add_generation_prompt=True,
                                    )
                                    ids = tokenizer(conv, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
                                except Exception:
                                    ids = []
                                own_prefix_list.append(ids)

                        # Collect cross-model completions and batch-tokenize once
                        cross_texts = []
                        cross_meta = []  # (q_idx, ep)
                        cross_ep_ids = []  # track ep id for mapping back after batch tokenization
                        for q_idx, eps in enumerate(per_question_eps):
                            for ep in eps:
                                if ep.get("model_idx") != model_idx:
                                    if args.mode in ("threshold", "weight"):
                                        try:
                                            rwd = ep.get("reward", None)
                                            if rwd is None or float(rwd) < float(getattr(args, "reward_threshold", 1.0)):
                                                continue
                                        except Exception:
                                            continue
                                    cross_texts.append(ep.get("completion", ""))
                                    cross_meta.append((q_idx, ep))
                                    cross_ep_ids.append(id(ep))
                        cross_ids_list = []
                        if len(cross_texts) > 0:
                            try:
                                _tok_c = tokenizer(cross_texts, add_special_tokens=False, return_tensors=None)
                                cross_ids_list = _tok_c["input_ids"] if isinstance(_tok_c, dict) else []
                            except Exception as e:
                                print(f"    Warning: Failed batch retokenize completions: {e}")
                                cross_ids_list = []
                        # Build mapping ep_id -> retokenized ids
                        ep_id_to_ids = {}
                        try:
                            for _ids, _epid in zip(cross_ids_list, cross_ep_ids):
                                ep_id_to_ids[_epid] = _ids
                        except Exception:
                            ep_id_to_ids = {}

                        # Per-question selection ensuring exactly num_completions_per_prompt episodes per question
                        vocab_size = getattr(tokenizer, "vocab_size", None)
                        reward_threshold = float(getattr(args, "reward_threshold", 1.0))

                        for q_idx, sample in enumerate(mm_decode_buffer):
                            own_prefix = own_prefix_list[q_idx] if q_idx < len(own_prefix_list) else []
                            own_eps_q = [ep for ep in per_question_eps[q_idx] if ep.get("model_idx") == model_idx]
                            # Accumulate self rollout rewards for LR adaptation (weight mode)
                            if lr_adapt_enabled and own_eps_q:
                                for _ep in own_eps_q:
                                    _r = _ep.get("reward", None)
                                    if _r is None:
                                        continue
                                    try:
                                        _self_reward_sum += float(_r)
                                        _self_reward_cnt += 1
                                    except Exception:
                                        pass

                            group: list[dict] = []

                            if args.mode == "random":
                                other_eps_q = [ep for ep in per_question_eps[q_idx] if ep.get("model_idx") != model_idx]
                                own_quota = num_completions_per_prompt // 2
                                cross_quota = num_completions_per_prompt - own_quota

                                own_selected = []
                                if own_quota > 0 and own_eps_q:
                                    if len(own_eps_q) >= own_quota:
                                        own_selected = random.sample(own_eps_q, own_quota)
                                    else:
                                        own_selected = own_eps_q.copy()
                                while len(own_selected) < own_quota and own_eps_q:
                                    own_selected.append(random.choice(own_eps_q))

                                cross_selected = []
                                if cross_quota > 0 and other_eps_q:
                                    if len(other_eps_q) >= cross_quota:
                                        cross_selected = random.sample(other_eps_q, cross_quota)
                                    else:
                                        cross_selected = other_eps_q.copy()
                                while len(cross_selected) < cross_quota and other_eps_q:
                                    cross_selected.append(random.choice(other_eps_q))

                                for ep in own_selected:
                                    group.append({
                                        "prefix_token_ids": ep["prefix_token_ids"],
                                        "generated_token_ids": ep["generated_token_ids"],
                                        "reward": ep.get("reward"),
                                        "completion": ep.get("completion"),
                                        "old_log_probs": ep.get("old_log_probs"),
                                    })

                                for ep in cross_selected:
                                    new_completion_ids = ep_id_to_ids.get(id(ep))
                                    if new_completion_ids is None:
                                        try:
                                            _single = tokenizer(
                                                ep.get("completion", ""),
                                                return_tensors="pt",
                                                add_special_tokens=False,
                                            )
                                            new_completion_ids = _single["input_ids"][0].tolist()
                                        except Exception:
                                            new_completion_ids = []
                                    converted_ids = []
                                    valid = True
                                    for tid in new_completion_ids:
                                        try:
                                            tid_int = int(tid)
                                        except Exception:
                                            valid = False
                                            break
                                        if vocab_size is not None and not (0 <= tid_int < vocab_size):
                                            valid = False
                                            break
                                        converted_ids.append(tid_int)
                                    if not valid:
                                        continue
                                    group.append({
                                        "prefix_token_ids": own_prefix, 
                                        "generated_token_ids": converted_ids,
                                        "reward": ep.get("reward"),
                                        "completion": ep.get("completion"),
                                        "old_log_probs": ep.get("old_log_probs"),
                                    })

                            else:  # args.mode in {"threshold", "weight"}
                                own_pos = [
                                    ep
                                    for ep in own_eps_q
                                    if ep.get("reward") is not None and float(ep.get("reward", 0.0)) >= reward_threshold
                                ]
                                own_neg = [ep for ep in own_eps_q if ep not in own_pos]

                                cross_pos = []
                                for ep in per_question_eps[q_idx]:
                                    if ep.get("model_idx") == model_idx:
                                        continue
                                    try:
                                        rwd = ep.get("reward", None)
                                        if rwd is None or float(rwd) < reward_threshold:
                                            continue
                                    except Exception:
                                        continue
                                    new_completion_ids = ep_id_to_ids.get(id(ep))
                                    if new_completion_ids is None:
                                        try:
                                            _single = tokenizer(
                                                ep.get("completion", ""),
                                                return_tensors="pt",
                                                add_special_tokens=False,
                                            )
                                            new_completion_ids = _single["input_ids"][0].tolist()
                                        except Exception:
                                            new_completion_ids = []
                                    converted_ids = []
                                    valid = True
                                    for tid in new_completion_ids:
                                        try:
                                            tid_int = int(tid)
                                        except Exception:
                                            valid = False
                                            break
                                        if vocab_size is not None and not (0 <= tid_int < vocab_size):
                                            valid = False
                                            break
                                        converted_ids.append(tid_int)
                                    if not valid:
                                        continue
                                    cross_pos.append({
                                        "prefix_token_ids": own_prefix,
                                        "generated_token_ids": converted_ids,
                                        "reward": ep.get("reward"),
                                        "completion": ep.get("completion"),
                                        "old_log_probs": ep.get("old_log_probs"),
                                    })

                                for ep in own_pos:
                                    group.append({
                                        "prefix_token_ids": ep["prefix_token_ids"],
                                        "generated_token_ids": ep["generated_token_ids"],
                                        "reward": ep["reward"],
                                        "completion": ep.get("completion"),
                                        "old_log_probs": ep.get("old_log_probs"),
                                    })

                                need = max(0, num_completions_per_prompt - len(group))
                                if need > 0 and cross_pos:
                                    random.shuffle(cross_pos)
                                    group.extend(cross_pos[:need])

                                need2 = max(0, num_completions_per_prompt - len(group))
                                if need2 > 0 and own_neg:
                                    own_neg_shuffled = own_neg.copy()
                                    random.shuffle(own_neg_shuffled)
                                    for ep in own_neg_shuffled[:need2]:
                                        group.append({
                                            "prefix_token_ids": ep["prefix_token_ids"],
                                            "generated_token_ids": ep["generated_token_ids"],
                                            "reward": ep["reward"],
                                            "completion": ep.get("completion"),
                                            "old_log_probs": ep.get("old_log_probs"),
                                        })

                            while len(group) < num_completions_per_prompt and own_eps_q:
                                filler_ep = random.choice(own_eps_q)
                                group.append({
                                    "prefix_token_ids": filler_ep["prefix_token_ids"],
                                    "generated_token_ids": filler_ep["generated_token_ids"],
                                    "reward": filler_ep.get("reward"),
                                    "completion": filler_ep.get("completion"),
                                    "old_log_probs": filler_ep.get("old_log_probs"),
                                })

                            if len(group) > num_completions_per_prompt:
                                group = group[:num_completions_per_prompt]

                            model_episodes.extend(group)

                        batch_episodes_storage[model_idx].extend(model_episodes)
                        # Save per-model self rollout mean for this flush
                        if lr_adapt_enabled:
                            batch_self_reward_mean[model_idx] = (
                                (_self_reward_sum / _self_reward_cnt) if _self_reward_cnt > 0 else 0.0
                            )
                        per_model_assembly[name] = time.perf_counter() - per_model_start

                    mm_decode_buffer = []
                    torch.cuda.empty_cache()
                    gc.collect()

                    assemble_duration = time.perf_counter() - assemble_start
                    if per_model_assembly:
                        per_model_str = ", ".join(
                            f"{name}:{dur:.2f}s" for name, dur in per_model_assembly.items()
                        )
                    else:
                        per_model_str = "no-model-data"
                    print(
                        f"[timing][mm-rollout] decode={decode_duration:.2f}s, "
                        f"assemble={assemble_duration:.2f}s ({per_model_str})"
                    )
            else:
                # Single-model fast path: accumulate and decode across questions in one vLLM batch
                decode_buffer.append(question)
                # Flush condition
                should_decode_now = len(decode_buffer) >= rollout_batch_size or question_idx == len(train_dataset) - 1
                if should_decode_now:
                    model_idx = 0
                    model, tokenizer, name, device = models[0], tokenizers[0], model_names[0], target_devices[0]
                    prompts = []
                    sample_refs = []
                    for sample in decode_buffer:
                        for _ in range(num_completions_per_prompt):
                            prompts.append(sample["prompt"])
                            sample_refs.append(sample)
                    # Ensure vLLM engine uses current LoRA weights for training-time rollout
                    try:
                        if getattr(model, "use_vllm", False) and getattr(model, "vllm_engine", None) is not None and getattr(model, "lora_config", None) is not None:
                            ok = model.sync_lora_to_vllm(adapter_name=f"lora_{name}")
                            if not ok:
                                print(f"[TRAIN] {name}: vLLM LoRA sync failed; consider disabling vLLM for training rollout.")
                    except Exception as _se:
                        print(f"[TRAIN] {name}: LoRA sync error: {_se}")
                    engine_label = "vLLM" if getattr(model, "use_vllm", False) and getattr(model, "vllm_engine", None) is not None else "HF"
                    print(f"  {name} ({device}): Decoding {len(prompts)} prompts with {engine_label}...")
                    # Decode in chunks to avoid OOM (cap per-call prompts)
                    decode_chunk_size = min(len(prompts), 256)
                    model_rewards = []
                    processed = 0
                    for start_idx in range(0, len(prompts), decode_chunk_size):
                        end_idx = min(start_idx + decode_chunk_size, len(prompts))
                        chunk_prompts = prompts[start_idx:end_idx]
                        chunk_refs = sample_refs[start_idx:end_idx]
                        completions, indices, masks, vllm_lps = model.generate(
                            prompts=chunk_prompts,
                            limitation=MAX_GEN_LEN,
                            temperature=TRAIN_TEMPERATURE,
                            verbose=True,
                            return_log_probs=bool(getattr(model, "use_vllm", False)),
                        )
                        for j, result in enumerate(completions):
                            sample = chunk_refs[j]
                            # Defer reward computation; fill later via process pool
                            reward_result = None
                            full_indices = indices[j]
                            mask = masks[j]
                            # Avoid GPU boolean indexing to prevent launching kernels that can surface prior async errors
                            if torch.cuda.is_available() and full_indices.is_cuda:
                                try:
                                    torch.cuda.synchronize(full_indices.device)
                                except Exception:
                                    torch.cuda.synchronize()
                            fi_cpu = full_indices.detach().contiguous().to("cpu")
                            mk_cpu = mask.detach().contiguous().to("cpu")
                            prefix_token_ids = fi_cpu[~mk_cpu].tolist()
                            generated_token_ids = fi_cpu[mk_cpu].tolist()
                            old_lp = None
                            if isinstance(vllm_lps, list) and j < len(vllm_lps) and vllm_lps[j] is not None:
                                try:
                                    old_lp = vllm_lps[j].detach().cpu().tolist()
                                except Exception:
                                    old_lp = None
                            if old_lp is None:
                                try:
                                    # Clone to avoid "inference tensors" autograd error
                                    input_ids = full_indices[:-1].clone().unsqueeze(0).to(device)
                                    target_ids = full_indices[1:].clone().unsqueeze(0).to(device)
                                    target_masks = mask[1:].clone().unsqueeze(0).to(device)

                                    # Temporarily use inference-friendly mode for log_prob computation
                                    prev_training = model.model.training
                                    prev_use_cache = getattr(model.model.config, "use_cache", True)
                                    had_gc = getattr(model.model, "is_gradient_checkpointing", False)
                                    model.model.eval()
                                    try:
                                        model.model.gradient_checkpointing_disable()
                                    except Exception:
                                        pass
                                    try:
                                        model.model.config.use_cache = True
                                    except Exception:
                                        pass
                                    with torch.inference_mode():
                                        lp = get_log_probs(
                                            model.model,
                                            input_ids,
                                            target_ids,
                                            target_masks,
                                            pad_token_id=tokenizer.pad_token_id,
                                            vocab_size=model.model.config.vocab_size,
                                            return_logits=False,
                                        )
                                    old_lp = lp[0][target_masks[0]].detach().cpu().tolist()
                                except Exception as _e:
                                    print(f"[TRAIN] old_log_probs fallback failed: {_e}")
                                    old_lp = None
                                finally:
                                    # Restore training-time settings
                                    try:
                                        model.model.config.use_cache = prev_use_cache
                                    except Exception:
                                        pass
                                    if had_gc:
                                        try:
                                            model.model.gradient_checkpointing_enable()
                                        except Exception:
                                            pass
                                    if prev_training:
                                        model.model.train()
                            episode_data = {
                                "prefix_token_ids": prefix_token_ids,
                                "generated_token_ids": generated_token_ids,
                                "reward": reward_result,
                                "completion": result,
                                "model_name": name,
                                "model_idx": model_idx,
                                "old_log_probs": old_lp,
                            }
                            batch_episodes_storage[0].append(episode_data)
                            # model_rewards appended later
                        processed += (end_idx - start_idx)
                        del completions, indices, masks, vllm_lps
                        torch.cuda.empty_cache()
                        gc.collect()
                    # Compute rewards for single-model path
                    if args.task_type in ("math", "code"):
                        reward_type = "math" if args.task_type == "math" else "code"
                        reward_tasks = []
                        ep_ptrs = []
                        # Build tasks from decode_buffer and the episodes appended in this flush
                        ep_slice = batch_episodes_storage[0][-len(prompts):] if len(prompts) > 0 else []
                        idx = 0
                        for sample in decode_buffer:
                            target = sample.get("solution", sample.get("answer"))
                            for _ in range(num_completions_per_prompt):
                                if idx < len(ep_slice):
                                    response = ep_slice[idx].get("completion", "")
                                    reward_tasks.append((reward_type, response, target))
                                    ep_ptrs.append(ep_slice[idx])
                                idx += 1
                        if reward_tasks:
                            try:
                                results = list(train_reward_executor.map(_reward_worker, reward_tasks))
                            except Exception as _sre:
                                print(f"[reward] single-model reward pool failed: {_sre}")
                                results = []
                                for ep, task in zip(ep_ptrs, reward_tasks):
                                    try:
                                        res = _reward_worker(task)
                                    except Exception:
                                        res = 0
                                    results.append(res)
                            model_rewards = []
                            for ep, score in zip(ep_ptrs, results):
                                ep["reward"] = score
                                model_rewards.append(score)
                            # Record self rollout mean reward for LR adaptation
                            if lr_adapt_enabled:
                                try:
                                    batch_self_reward_mean[0] = float(np.mean(model_rewards)) if len(model_rewards) > 0 else 0.0
                                except Exception:
                                    batch_self_reward_mean[0] = 0.0
                            print(f"    {name}: Batch rollout finished. Mean reward: {np.mean(model_rewards):.3f}")
                        else:
                            print(f"    {name}: Batch rollout finished. No reward tasks queued.")
                    del prompts, sample_refs
                    decode_buffer = []
                    torch.cuda.empty_cache()
                    gc.collect()
            # Check if we should perform batch policy update
            should_update = False
            if (question_idx + 1) % batch_size == 0:  # Reached batch size
                should_update = True
                print(f"\n🎯 Batch {batch_step + 1} completed! Performing batch policy update...")
            elif question_idx == len(train_dataset) - 1:  # Last question in epoch
                should_update = True
                print(f"\n🎯 End of epoch! Performing final batch policy update...")
            
            if should_update:
                batch_step += 1
                policy_timing_start = time.perf_counter()
                
                # Prepare per-model jobs (truncate/group and pre-log rewards)
                jobs = []
                for model_idx, (model, optimizer, tokenizer, name, device) in enumerate(zip(models, optimizers, tokenizers, model_names, target_devices)):
                    final_episodes = batch_episodes_storage[model_idx]
                    
                    if len(final_episodes) == 0:
                        print(f"    Warning: {name} has no episodes in batch, skipping...")
                        continue
                    
                    # Ensure episodes are properly grouped for advantage calculation
                    if len(final_episodes) % num_completions_per_prompt != 0:
                        print(f"    Warning: {name} has {len(final_episodes)} episodes, not divisible by {num_completions_per_prompt}")
                        if not is_multi_model_mode:
                            final_episodes = final_episodes[:len(final_episodes) - (len(final_episodes) % num_completions_per_prompt)]
                    
                    if len(final_episodes) == 0:
                        print(f"    Warning: {name} has no valid episodes after grouping, skipping...")
                        continue

                    if _WANDB_AVAILABLE:
                        reward_values = [ep.get("reward") for ep in final_episodes if ep.get("reward") is not None]
                        if reward_values:
                            try:
                                reward_mean = float(np.mean(reward_values))
                                wandb.log({
                                    f"reward/mean/{name}": reward_mean,
                                    f"reward/success_rate/{name}": reward_mean,
                                    "train/step": batch_step,
                                }, step=batch_step)
                            except Exception as _e:
                                print(f"[wandb] reward log failed: {_e}")

                    # Apply per-batch LR adaptation for weight mode
                    if lr_adapt_enabled:
                        # Use the latest batch's self rollout mean if available; otherwise fallback to previous EMA or 0.0
                        _m = batch_self_reward_mean[model_idx]
                        if _m is None:
                            _m = lr_adapt_ema[model_idx] if lr_adapt_ema[model_idx] is not None else 0.0
                        # Update EMA
                        if lr_adapt_ema[model_idx] is None:
                            lr_adapt_ema[model_idx] = float(_m)
                        else:
                            lr_adapt_ema[model_idx] = 0.3 * float(_m) + 0.7 * float(lr_adapt_ema[model_idx])
                        _mhat = float(lr_adapt_ema[model_idx])
                        # Warmup for first 2 updates
                        if batch_step <= 2:
                            _w = 1.0
                        else:
                            try:
                                _w = 1.0 / (1.0 + math.exp(5.0 * (_mhat - 0.5)))
                            except Exception:
                                _w = 1.0
                            # Clamp to [0.3, 2.0]
                            _w = max(0.3, min(2.0, _w))
                        # Apply LR multiplier to optimizer param groups
                        _scaled_lr = base_lrs[model_idx] * _w
                        try:
                            for pg in optimizer.param_groups:
                                pg["lr"] = _scaled_lr
                        except Exception:
                            pass
                        print(f"  {name} ({device}): Batch updating policy with {len(final_episodes)} episodes... (lr-mult={_w:.3f}, m_self={_m:.3f}, m_hat={_mhat:.3f})")
                    else:
                        print(f"  {name} ({device}): Batch updating policy with {len(final_episodes)} episodes...")
                    
                    jobs.append({
                        "model_idx": model_idx,
                        "hf_model": model.model,
                        "optimizer": optimizer,
                        "pad_token_id": tokenizer.pad_token_id,
                        "device": device,
                        "name": name,
                        "episodes": final_episodes,
                        "ref_model": (ref_models[model_idx].model if ref_models[model_idx] is not None else None),
                        "old_log_probs": old_log_probs_storage[model_idx],
                        "has_fact": getattr(model, "fact_config", None) is not None,
                    })

                # Define worker for parallel policy update
                def _do_update(job):
                    idx = job["model_idx"]
                    name = job["name"]
                    device = job["device"]
                    update_start = time.perf_counter()
                    update_result = update_policy(
                        model=job["hf_model"],
                        optimizer=job["optimizer"],
                        episodes=job["episodes"],
                        pad_token_id=job["pad_token_id"],
                        max_grad_norm=args.max_grad_norm,
                        device=torch.device(device),
                        dtype=dtype,
                        num_completions_per_prompt=num_completions_per_prompt,
                        ref_model=job["ref_model"],
                        old_log_probs=job["old_log_probs"],
                        epsilon_low=epsilon_low,
                        epsilon_high=epsilon_high,
                        beta=beta,
                        loss_type=loss_type,
                        ppo_mini_batch_size=ppo_mini_batch_size,
                        ppo_micro_batch_size=ppo_micro_batch_size,
                        ppo_epochs=ppo_epochs,
                        loss_aggregation=loss_aggregation,
                        normalize_adv_by_std=normalize_advantages,
                        advantage_clip=advantage_clip,
                    )
                    update_duration = time.perf_counter() - update_start
                    return idx, name, device, update_result, job["has_fact"], update_duration

                # Run updates in parallel across models
                if len(jobs) > 0:
                    per_model_update = {}
                    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                        futures = [pool.submit(_do_update, jb) for jb in jobs]
                        for fut in as_completed(futures):
                            model_idx, name, device, update_result, has_fact, update_duration = fut.result()

                            if update_result.get("current_log_probs") is not None:
                                old_log_probs_storage[model_idx] = update_result["current_log_probs"]
                    
                            print(
                                f"    {name}: Batch updated - Loss: {update_result['loss']:.4f}, "
                          f"Grad norm: {update_result['grad_norm']:.4f}, "
                          f"Entropy: {update_result['entropy']:.4f}, "
                          f"KL div: {update_result['kl_div']:.4f}, "
                                f"Clip ratio: {update_result['clip_ratio']:.4f}"
                            )

                            if _WANDB_AVAILABLE:
                                try:
                                    wandb.log({
                                        f"train/loss/{name}": float(update_result["loss"]),
                                        f"train/grad_norm/{name}": float(update_result["grad_norm"]),
                                        f"train/entropy/{name}": float(update_result["entropy"]),
                                        f"train/kl_div/{name}": float(update_result["kl_div"]),
                                        f"train/clip_ratio/{name}": float(update_result["clip_ratio"]),
                                        "train/step": batch_step,
                                    }, step=batch_step)
                                except Exception as _e:
                                    print(f"[wandb] log failed: {_e}")
                    
                            # Refresh vLLM FacT engine if needed (do this in main thread to avoid engine races)
                            model = models[model_idx]
                            if getattr(args, "use_vllm", False) and has_fact:
                                try:
                                    refresh_ok = model.refresh_vllm_merged_engine(
                                        gpu_memory_utilization=getattr(args, "vllm_gpu_mem", 0.6),
                                        max_model_len=getattr(args, "vllm_max_model_len", None),
                                    )
                                    if not refresh_ok:
                                        print(f"[FacT] {name}: vLLM engine refresh failed; continuing with HF rollout.")
                                except Exception as _refresh_e:
                                    print(f"[FacT] {name}: vLLM refresh error: {_refresh_e}")
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    per_model_update[name] = update_duration

                    policy_total_duration = time.perf_counter() - policy_timing_start
                    if per_model_update:
                        per_model_str = ", ".join(
                            f"{name}:{dur:.2f}s" for name, dur in per_model_update.items()
                        )
                    else:
                        per_model_str = "no-updates"
                    print(
                        f"[timing][policy] total={policy_total_duration:.2f}s ({per_model_str})"
                    )
                else:
                    print(
                        f"[timing][policy] total={time.perf_counter() - policy_timing_start:.2f}s (no jobs)"
                    )
                # Restore optimizers' LR to base after update (weight mode)
                if lr_adapt_enabled:
                    try:
                        for _mi, _opt in enumerate(optimizers):
                            for _pg in _opt.param_groups:
                                _pg["lr"] = base_lrs[_mi]
                    except Exception:
                        pass
                
                # Clear batch storage
                batch_episodes_storage = [[] for _ in range(num_models)]
                # Reset batch self reward record after consuming it
                batch_self_reward_mean = [None for _ in range(num_models)]
                print(f"✅ Batch {batch_step} update completed!\n")

                if max_updates is not None and batch_step >= max_updates:
                    training_done = True

                # 在策略更新步上进行验证
                if _WANDB_AVAILABLE and args.val_interval and args.val_interval > 0 and (batch_step % args.val_interval == 0):
                    run_validation(batch_step)

                # 在策略更新步上保存检查点
                if args.ckpt_interval and args.ckpt_interval > 0 and (batch_step % args.ckpt_interval == 0):
                    for model_idx, (model, name) in enumerate(zip(models, model_names)):
                        ckpt_path = Path(args.ckpt_dir) / f"{name}_step_{batch_step}"
                        model.save(str(ckpt_path))
                        print(f"Saved checkpoint for {name} at step {batch_step}")
                        torch.cuda.empty_cache()
                        gc.collect()

            if training_done:
                break

            # 验证改为在策略更新步（batch_step）后触发；此处不再按 env_step 触发
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # Optionally add cleanup or memory logging keyed by batch_step if needed

            # 检查点改为在策略更新步上保存，由更新块统一处理

    # Save final checkpoint(s) at the end of training if requested
    if args.save_final:
        for model_idx, (model, name) in enumerate(zip(models, model_names)):
            final_ckpt_path = Path(args.ckpt_dir) / f"{name}_final_step_{batch_step}"
            model.save(str(final_ckpt_path))
            print(f"Saved final checkpoint for {name} at step {batch_step}")
            torch.cuda.empty_cache()
            gc.collect()

    # Shutdown training-time reward executor
    try:
        train_reward_executor.shutdown(wait=True)
    except Exception:
        pass

    # Finish wandb run
    if _WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception as _e:
            print(f"[wandb] finish failed: {_e}")

    print("Training completed!")


if __name__ == "__main__":
    main()
