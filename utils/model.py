from typing import List, Dict, Tuple, Optional, Union, Any
from contextlib import nullcontext
import torch
import json
import os
import tempfile
import shutil
import time
from datetime import datetime
import gc
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .fact_adapter import (
    apply_fact_to_model,
    count_fact_parameters,
    analyze_shared_fact_parameters,
    analyze_all_trainable_parameters,
    freeze_non_fact_parameters,
)
from .fact_adapter import merge_fact_adapters_to_dense_copy
from .fact_adapter import SharedFacTLinear  # for detecting FacT-wrapped modules
try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams as VLLMSamplingParams
    try:
        # Prefer new import path
        from vllm.lora.request import LoRARequest as VLLMLoRARequest  # type: ignore
    except Exception:
        try:
            from vllm import LoRARequest as VLLMLoRARequest  # type: ignore
        except Exception:
            VLLMLoRARequest = None  # type: ignore
    _VLLM_AVAILABLE = True
except Exception:
    VLLMEngine = None
    VLLMSamplingParams = None
    VLLMLoRARequest = None
    _VLLM_AVAILABLE = False

if _VLLM_AVAILABLE:
    try:
        from .vllm_worker import VLLMWorkerProxy  # type: ignore
    except Exception:
        VLLMWorkerProxy = None  # type: ignore
else:  # pragma: no cover
    VLLMWorkerProxy = None  # type: ignore


def _normalise_logprobs(raw: Optional[List[Any]], token_ids: List[int]) -> Optional[List[Optional[float]]]:
    """Convert heterogeneous logprob payloads to python floats (None -> NaN)."""
    if raw is None:
        return None
    normalised: List[Optional[float]] = []
    for idx, tk in enumerate(raw):
        if tk is None:
            normalised.append(None)
            continue
        try:
            if hasattr(tk, "logprob"):
                normalised.append(float(tk.logprob))
            elif isinstance(tk, dict) and idx < len(token_ids):
                chosen = token_ids[idx]
                if chosen in tk:
                    normalised.append(float(tk[chosen]))
                elif tk:
                    normalised.append(float(max(tk.values())))
                else:
                    normalised.append(None)
            else:
                normalised.append(None)
        except Exception:
            normalised.append(None)
    return normalised


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
        gradient_checkpointing: bool = False,
        use_vllm: bool = False,
        vllm_gpu_memory_utilization: Optional[float] = None,
        vllm_gpu_id: Optional[int] = None,
        vllm_max_model_len: Optional[int] = None,
    ):
        # Store configuration for save() method
        self.original_model_path = model_path
        self.lora_config = lora_config
        self.fact_config = fact_config
        # persist vLLM preferences
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        
        def _ts() -> str:
            return datetime.now().strftime("%H:%M:%S.%f")
        print(f"[{_ts()}][LM] init start: model={model_path}, device={target_device}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # Prefer left padding for decoder-only models to ensure correct generation
        try:
            if getattr(self.tokenizer, "padding_side", None) != "left":
                self.tokenizer.padding_side = "left"
        except Exception:
            pass
        print(f"[{_ts()}][LM] tokenizer loaded: model={model_path}", flush=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = target_device
        # Remember requested attention backend
        self.attn_impl = attn_impl

        # vLLM runtime handles (initialized later; worker preferred, inline as fallback)
        self.vllm_target_gpu = vllm_gpu_id
        self.vllm_engine = None
        self.vllm_worker: Optional[VLLMWorkerProxy] = None
        self.use_vllm = bool(use_vllm and _VLLM_AVAILABLE)

        # Load model with attention backend selection (auto prefers flash_attention_2)
        def _load_with_attn(impl: str):
            print(f"[{_ts()}][LM] from_pretrained enter: impl={impl}, model={model_path}", flush=True)
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=target_device if target_device != "cpu" else None,
                torch_dtype=torch_dtype,
                attn_implementation=impl,
            )

        selected_impl = None
        if attn_impl == "flash":
            try:
                self.model = _load_with_attn("flash_attention_2")
                selected_impl = "flash_attention_2"
            except Exception as _e:
                print(f"[attn] flash_attention_2 failed: {_e}. Falling back to sdpa.")
                self.model = _load_with_attn("sdpa")
                selected_impl = "sdpa"
        elif attn_impl == "auto":
            try:
                self.model = _load_with_attn("flash_attention_2")
                selected_impl = "flash_attention_2"
            except Exception:
                self.model = _load_with_attn("sdpa")
                selected_impl = "sdpa"
        else:
            self.model = _load_with_attn("sdpa")
            selected_impl = "sdpa"
        print(f"[{_ts()}][LM] HF model instantiated: impl={selected_impl}, model={model_path}", flush=True)
        
        # If target device is CPU, don't specify device_map but move manually
        if target_device == "cpu":
            self.model = self.model.to("cpu")

        try:
            if torch.cuda.is_available() and target_device.startswith("cuda"):
                dev_idx = int(str(target_device).split(":")[-1]) if ":" in str(target_device) else 0
                free, total = torch.cuda.mem_get_info(dev_idx)
                print(f"[{_ts()}][LM] cuda mem (GiB): free={free/2**30:.2f}, total={total/2**30:.2f} on {target_device}", flush=True)
        except Exception:
            pass
        print(f"[{_ts()}][LM] Model loaded on device: {target_device} for {model_path}", flush=True)
        
        # Configure gradient checkpointing if enabled
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            try:
                # Ensure inputs require grad so checkpoint works end-to-end
                self.model.enable_input_require_grads()
            except Exception:
                pass
            print(f"[{_ts()}][LM] Gradient checkpointing enabled for {model_path}", flush=True)
        
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
            # vLLM 无法在线注入 FacT；如需 vLLM rollout，需要合并为 dense 后重建引擎
            if self.use_vllm:
                print("[FacT] vLLM rollout需要合并后重建或使用外部热更机制。")
        
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

        # Optional vLLM engine for fast generation (HF model still used for training)
        if self.use_vllm and not _VLLM_AVAILABLE:
            self.use_vllm = False
        if self.use_vllm:
            try:
                visible_cuda = self._resolve_vllm_visible_device(self.vllm_target_gpu)
                print(f"[{_ts()}][vLLM] init start: model={model_path}, visible_cuda={visible_cuda}", flush=True)
                if torch_dtype == torch.bfloat16:
                    dtype_str = "bfloat16"
                elif torch_dtype == torch.float16:
                    dtype_str = "float16"
                elif torch_dtype == torch.float32:
                    dtype_str = "float32"
                else:
                    dtype_str = "auto"
                gpu_util = 0.6 if vllm_gpu_memory_utilization is None else float(vllm_gpu_memory_utilization)
                if VLLMWorkerProxy is not None:
                    try:
                        self.vllm_worker = VLLMWorkerProxy(
                            model_path=model_path,
                            dtype=dtype_str,
                            visible_cuda=visible_cuda,
                            gpu_memory_utilization=gpu_util,
                            max_model_len=vllm_max_model_len,
                            enable_lora=True,
                            enforce_eager=True,
                        )
                        print(
                            f"[vLLM] worker enabled for {model_path} on "
                            f"CUDA_VISIBLE_DEVICES={self.vllm_worker.visible_cuda or 'auto'}"
                        , flush=True)
                    except Exception as _e:
                        print(f"[vLLM] worker init failed for {model_path}: {_e}. Trying inline engine.")
                        self.vllm_worker = None
                if self.vllm_worker is None:
                    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                    target_cvd = visible_cuda
                    if target_cvd is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = target_cvd
                    try:
                        kwargs = dict(
                            model=model_path,
                            dtype=dtype_str,
                            tensor_parallel_size=1,
                            trust_remote_code=True,
                            gpu_memory_utilization=gpu_util,
                            enable_lora=True,
                            enforce_eager=True,  # disable CUDA graph to avoid illegal memory access issues
                        )
                        if vllm_max_model_len is not None:
                            try:
                                self.vllm_engine = VLLMEngine(**kwargs, max_seq_len=int(vllm_max_model_len))
                            except TypeError:
                                self.vllm_engine = VLLMEngine(**kwargs, max_model_len=int(vllm_max_model_len))
                        else:
                            self.vllm_engine = VLLMEngine(**kwargs)
                        print(f"[{_ts()}][vLLM] engine enabled for {model_path}", flush=True)
                    finally:
                        if target_cvd is not None:
                            if prev_cvd is None:
                                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                            else:
                                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
            except Exception as _e:
                print(f"[vLLM] init failed for {model_path}: {_e}. Fallback to HF generate.")
                self.vllm_worker = None
                self.vllm_engine = None
                self.use_vllm = False

        # vLLM LoRA state
        self.vllm_lora_adapter_name: Optional[str] = None
        self.vllm_lora_adapter_path: Optional[str] = None
        self.vllm_lora_loaded: bool = False
        # Track last FacT-merged directory for vLLM refresh cycles
        self._fact_last_merged_dir: Optional[str] = None

        # 不再尝试接入 VERL SPMD rollout
        self.verl_rollout = None

    def _resolve_vllm_visible_device(self, target_index: Optional[int]) -> Optional[str]:
        """Map local GPU index to physical CUDA id for worker binding."""
        if target_index is None:
            return None
        try:
            # Negative index disables binding
            if int(target_index) < 0:
                return None
            cur_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cur_cvd:
                devices = [p.strip() for p in cur_cvd.split(",") if p.strip()]
                if 0 <= int(target_index) < len(devices):
                    return devices[int(target_index)]
                return str(target_index)
            return str(target_index)
        except Exception:
            return str(target_index)

    def export_fact_merged(self, path: str) -> str:
        """Export a dense model with FacT merged into Linear weights.

        Returns the output directory path.
        """
        if self.fact_config is None:
            raise RuntimeError("export_fact_merged() called but no FacT adapters are present.")
        os.makedirs(path, exist_ok=True)
        # Create merged copy and save
        merged_model = merge_fact_adapters_to_dense_copy(self.model)
        try:
            merged_model.save_pretrained(path)
        except Exception as _e:
            # Fallback to manual state_dict save if needed
            print(f"[FacT] save_pretrained failed ({_e}), saving state_dict instead.")
            self.model.config.save_pretrained(path)
            torch.save(merged_model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        # Ensure tokenizer assets exist alongside for vLLM
        try:
            self.tokenizer.save_pretrained(path)
        except Exception:
            pass
        return path

    # 已移除 VERL SPMD rollout 初始化

    def eval_with_vLLM_on_merged(self, prompts: List[str], limitation: int = 1024, temperature: float = 0.7) -> List[str]:
        """Run evaluation with vLLM on a temporary merged-dense copy of the FacT model.

        Exports to a temp dir, spins up a one-off vLLM engine, generates, then cleans up.
        Falls back to HF generate if vLLM is unavailable.
        """
        if not _VLLM_AVAILABLE:
            # Fallback
            return self.generate(prompts, limitation=limitation, temperature=temperature, verbose=False)  # type: ignore
        tmp_dir = tempfile.mkdtemp(prefix="fact_merged_")
        try:
            out_dir = os.path.join(tmp_dir, "merged")
            self.export_fact_merged(out_dir)
            # Prepare vLLM dtype string
            model_dtype = next(self.model.parameters()).dtype
            if model_dtype == torch.bfloat16:
                dstr = "bfloat16"
            elif model_dtype == torch.float16:
                dstr = "float16"
            elif model_dtype == torch.float32:
                dstr = "float32"
            else:
                dstr = "auto"
            # Optionally scope CVD to the vLLM GPU index, if provided
            prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            target_cvd = None
            try:
                tgt_idx = getattr(self, "vllm_target_gpu", None)
                if tgt_idx is not None:
                    cur_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if cur_cvd:
                        lst = [p.strip() for p in cur_cvd.split(",") if p.strip()]
                        if 0 <= int(tgt_idx) < len(lst):
                            target_cvd = lst[int(tgt_idx)]
                        else:
                            target_cvd = str(tgt_idx)
                    else:
                        target_cvd = str(tgt_idx)
                    os.environ["CUDA_VISIBLE_DEVICES"] = target_cvd
            except Exception:
                target_cvd = None

            # Start engine and generate
            engine = None
            try:
                engine = VLLMEngine(
                    model=out_dir,
                    dtype=dstr,
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.6,
                    enforce_eager=True,
                )
                sp = VLLMSamplingParams(temperature=temperature, max_tokens=limitation)
                conversations = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                    ) for p in prompts
                ]
                outputs = engine.generate(conversations, sampling_params=sp)
                completions = [o.outputs[0].text for o in outputs]
            finally:
                # Restore visibility
                if target_cvd is not None:
                    if prev_cvd is None:
                        try:
                            del os.environ["CUDA_VISIBLE_DEVICES"]
                        except Exception:
                            pass
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
                try:
                    del engine
                except Exception:
                    pass
            return completions
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def refresh_vllm_merged_engine(self, gpu_memory_utilization: float = 0.6, max_model_len: Optional[int] = None) -> bool:
        """Merge FacT into dense copy and (re)build a persistent vLLM engine for rollout.

        Returns True if engine is ready; False otherwise.
        """
        if not _VLLM_AVAILABLE:
            print("[vLLM] Not available. Cannot refresh merged engine.")
            return False
        if self.fact_config is None:
            print("[FacT] No FacT adapters present; refresh_vllm_merged_engine skipped.")
            return False
        if self.vllm_worker is not None:
            tmp_root = tempfile.mkdtemp(prefix="fact_merged_vllm_")
            merged_dir = os.path.join(tmp_root, "merged")
            try:
                self.export_fact_merged(merged_dir)
                self.vllm_worker.load_weights(merged_dir)
                if self._fact_last_merged_dir and os.path.isdir(self._fact_last_merged_dir):
                    shutil.rmtree(self._fact_last_merged_dir, ignore_errors=True)
                self._fact_last_merged_dir = tmp_root
                print("[vLLM] Worker merged FacT weights are ready for rollout.")
                return True
            except Exception as _e:
                print(f"[vLLM] worker failed to load merged FacT weights: {_e}")
                shutil.rmtree(tmp_root, ignore_errors=True)
                return False
        # Export merged model to a temp dir
        tmp_root = tempfile.mkdtemp(prefix="fact_merged_vllm_")
        merged_dir = os.path.join(tmp_root, "merged")
        try:
            self.export_fact_merged(merged_dir)
        except Exception as _e:
            print(f"[FacT] export_fact_merged failed: {_e}")
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass
            return False
        # Optionally scope CVD to a specific GPU index for vLLM
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        target_cvd = None
        try:
            tgt_idx = getattr(self, "vllm_target_gpu", None)
            if tgt_idx is not None:
                cur_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cur_cvd:
                    lst = [p.strip() for p in cur_cvd.split(",") if p.strip()]
                    if 0 <= int(tgt_idx) < len(lst):
                        target_cvd = lst[int(tgt_idx)]
                    else:
                        target_cvd = str(tgt_idx)
                else:
                    target_cvd = str(tgt_idx)
                os.environ["CUDA_VISIBLE_DEVICES"] = target_cvd
        except Exception:
            target_cvd = None
        # dtype string for vLLM
        model_dtype = next(self.model.parameters()).dtype
        if model_dtype == torch.bfloat16:
            dstr = "bfloat16"
        elif model_dtype == torch.float16:
            dstr = "float16"
        elif model_dtype == torch.float32:
            dstr = "float32"
        else:
            dstr = "auto"
        try:
            # Dispose previous engine if any
            try:
                if self.vllm_engine is not None:
                    try:
                        shutdown = getattr(self.vllm_engine, "shutdown", None)
                        if callable(shutdown):
                            shutdown()
                            # allow background worker to exit cleanly before re-init
                            time.sleep(0.2)
                    except Exception:
                        pass
                    del self.vllm_engine
                    if torch.cuda.is_available():
                        try:
                            current_device = torch.cuda.current_device()
                            device_count = torch.cuda.device_count()
                            for dev_idx in range(device_count):
                                with torch.cuda.device(dev_idx):
                                    torch.cuda.empty_cache()
                            torch.cuda.set_device(current_device)
                        except Exception:
                            torch.cuda.empty_cache()
                    gc.collect()
            except Exception:
                pass
            _kwargs = dict(
                model=merged_dir,
                dtype=dstr,
                tensor_parallel_size=1,
                trust_remote_code=True,
                gpu_memory_utilization=float(gpu_memory_utilization),
                enforce_eager=True,
            )
            if max_model_len is not None:
                try:
                    self.vllm_engine = VLLMEngine(**_kwargs, max_seq_len=int(max_model_len))
                except TypeError:
                    self.vllm_engine = VLLMEngine(**_kwargs, max_model_len=int(max_model_len))
            else:
                self.vllm_engine = VLLMEngine(**_kwargs)
            # Clean previous merged dir, keep current for lifecycle
            if self._fact_last_merged_dir and os.path.isdir(self._fact_last_merged_dir):
                try:
                    shutil.rmtree(self._fact_last_merged_dir, ignore_errors=True)
                except Exception:
                    pass
            self._fact_last_merged_dir = tmp_root
            self.use_vllm = True
            print("[vLLM] Merged FacT engine is ready for rollout.")
            return True
        except Exception as _e:
            print(f"[vLLM] failed to build engine from merged FacT model: {_e}")
            self.use_vllm = False
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass
            return False
        finally:
            if target_cvd is not None:
                if prev_cvd is None:
                    try:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    except Exception:
                        pass
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd

    def generate(
        self,
        prompts: List[str],
        limitation: int = 1024,
        temperature: float = 1.0,
        verbose: bool = False,
        return_log_probs: bool = False,
    ) -> Union[List[str], Tuple[List[str], torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]]:
        """Generate text completions (vLLM preferred, HF fallback)."""
        conversations = []
        for prompt in prompts:
            conversations.append(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ))

        sampling_kwargs: Dict[str, Any] = {
            "temperature": float(temperature),
            "max_tokens": int(limitation),
            "n": 1,
        }
        if return_log_probs:
            sampling_kwargs["logprobs"] = 1

        use_vllm_path = self.use_vllm and (self.vllm_worker is not None or self.vllm_engine is not None)
        worker_payload: Optional[List[Dict[str, Any]]] = None
        worker_use_lora = (
            self.vllm_worker is not None
            and isinstance(self.vllm_lora_adapter_name, str)
            and isinstance(self.vllm_lora_adapter_path, str)
        )

        if use_vllm_path and self.vllm_worker is not None:
            worker_payload = self.vllm_worker.generate(
                conversations=conversations,
                sampling=sampling_kwargs,
                use_lora=worker_use_lora,
            )
            completions = [item.get("text", "") for item in worker_payload]
        elif use_vllm_path:
            lora_kwargs: Dict[str, Any] = {}
            if (
                VLLMLoRARequest is not None
                and isinstance(self.vllm_lora_adapter_name, str)
                and isinstance(self.vllm_lora_adapter_path, str)
            ):
                try:
                    lora_kwargs["lora_request"] = VLLMLoRARequest(
                        lora_name=self.vllm_lora_adapter_name,
                        lora_int_id=abs(hash(self.vllm_lora_adapter_name)) % (10**9),
                        lora_path=self.vllm_lora_adapter_path,
                    )
                except Exception:
                    pass
            if VLLMSamplingParams is None:
                raise RuntimeError("VLLM sampling parameters unavailable for inline engine.")
            try:
                sampling = VLLMSamplingParams(**sampling_kwargs)
            except TypeError:
                sampling_kwargs.pop("logprobs", None)
                sampling = VLLMSamplingParams(**sampling_kwargs)
            with torch.no_grad():
                try:
                    outputs = self.vllm_engine.generate(conversations, sampling, **lora_kwargs)
                except TypeError:
                    outputs = self.vllm_engine.generate(conversations, sampling)
            worker_payload = []
            for out in outputs:
                slot = out.outputs[0]
                token_ids = list(slot.token_ids)
                worker_payload.append(
                    {
                        "text": slot.text,
                        "token_ids": token_ids,
                        "logprobs": _normalise_logprobs(slot.logprobs, token_ids),
                    }
                )
            completions = [entry["text"] for entry in worker_payload]
        else:
            # HF fallback: generate with Hugging Face model
            batch = self.tokenizer(
                conversations,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            prev_training = self.model.training
            prev_use_cache = getattr(self.model.config, "use_cache", True)
            had_gc = getattr(self.model, "is_gradient_checkpointing", False)
            self.model.eval()
            try:
                self.model.gradient_checkpointing_disable()
            except Exception:
                pass
            # Honor existing config.use_cache set by callers; do not force True here.
            with torch.no_grad():
                # Prefer BF16 autocast to reduce overflow risk; fallback to FP16 or disable.
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    try:
                        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    except Exception:
                        try:
                            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                        except Exception:
                            amp_ctx = nullcontext()
                else:
                    amp_ctx = nullcontext()
                with amp_ctx:
                    gen_out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=int(limitation),
                        do_sample=(float(temperature) > 0.0),
                        temperature=float(temperature) if float(temperature) > 0.0 else 1.0,
                        eos_token_id=self.eos_token_id,
                        pad_token_id=self.pad_token_id,
                        use_cache=getattr(self.model.config, "use_cache", True),
                        return_dict_in_generate=False,
                        output_scores=False,
                    )
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            completions = []
            gen_ids_list: List[List[int]] = []
            for i in range(sequences.size(0)):
                pref_len = int(attn_mask[i].sum().item())
                gen_ids = sequences[i][pref_len:].tolist()
                gen_ids_list.append(gen_ids)
                try:
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                except Exception:
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                completions.append(text)
            try:
                self.model.config.use_cache = prev_use_cache
            except Exception:
                pass
            if had_gc:
                try:
                    self.model.gradient_checkpointing_enable()
                except Exception:
                    pass
            if prev_training:
                self.model.train()
            worker_payload = [
                {"text": txt, "token_ids": ids, "logprobs": None}
                for txt, ids in zip(completions, gen_ids_list)
            ]
        if worker_payload is None:
            worker_payload = []
        # Prefer token ids from vLLM if available to avoid retokenization mismatch
        # Batch-tokenize conversation prefixes to reduce per-item overhead
        try:
            _prefix_tok = self.tokenizer(
                conversations,
                add_special_tokens=False,
                return_tensors=None,
            )
            # Accept BatchEncoding or dict-like; only check for key presence
            if _prefix_tok is not None and ("input_ids" in _prefix_tok):
                prefix_ids_list = _prefix_tok["input_ids"]
                # Normalize container to a Python list (handles tensors/ndarrays)
                if hasattr(prefix_ids_list, "tolist"):
                    prefix_ids_list = prefix_ids_list.tolist()
                else:
                    try:
                        prefix_ids_list = list(prefix_ids_list)
                    except Exception:
                        prefix_ids_list = []
                # Ensure each row is a plain list of ints
                _norm = []
                for row in prefix_ids_list:
                    if hasattr(row, "tolist"):
                        _norm.append(row.tolist())
                    elif isinstance(row, list):
                        _norm.append(row)
                    else:
                        try:
                            _norm.append(list(row))
                        except Exception:
                            _norm.append([])
                prefix_ids_list = _norm
            else:
                prefix_ids_list = []
        except Exception:
            prefix_ids_list = []
        if not prefix_ids_list or len(prefix_ids_list) != len(conversations):
            if not prefix_ids_list:
                print("[tokenizer] fallback to per-item prefix tokenization (empty batch result)")
            else:
                print(
                    f"[tokenizer] fallback to per-item prefix tokenization (batch size mismatch: "
                    f"{len(prefix_ids_list)} != {len(conversations)})"
                )
            prefix_ids_list = []
            for conv in conversations:
                try:
                    ids = self.tokenizer(conv, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
                except Exception:
                    ids = []
                prefix_ids_list.append(ids)
        gen_ids_list: List[List[int]] = []
        vllm_log_probs_list: Optional[List[torch.Tensor]] = [] if return_log_probs else None
        for conv, payload in zip(conversations, worker_payload):
            token_ids = payload.get("token_ids") if isinstance(payload, dict) else None
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            elif isinstance(token_ids, tuple):
                token_ids = list(token_ids)
            elif not isinstance(token_ids, list):
                text = ""
                if isinstance(payload, dict):
                    text = payload.get("text", "")
                try:
                    token_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
                except Exception:
                    token_ids = []
            else:
                token_ids = list(token_ids)
            gen_ids_list.append(token_ids)
            if return_log_probs:
                log_probs = None
                if isinstance(payload, dict):
                    log_probs = payload.get("logprobs")
                if log_probs is None:
                    vllm_log_probs_list.append(None)  # type: ignore[arg-type]
                else:
                    tensor_vals = [
                        float("nan") if lp is None else float(lp) for lp in log_probs  # type: ignore[arg-type]
                    ]
                    vllm_log_probs_list.append(
                        torch.tensor(tensor_vals, dtype=torch.float32, device="cpu")
                    )
        if not prefix_ids_list or not gen_ids_list:
            max_len = 0
        else:
            max_len = max(len(p) + len(g) for p, g in zip(prefix_ids_list, gen_ids_list))
        pad_id = self.tokenizer.pad_token_id
        indices_list = []
        masks_list = []
        for p_ids, g_ids in zip(prefix_ids_list, gen_ids_list):
            # Ensure both sides are python lists before concatenation to avoid
            # TypeError when g_ids may be a tuple/array-like from vLLM outputs.
            if not isinstance(p_ids, list):
                try:
                    p_ids = list(p_ids)
                except Exception:
                    p_ids = []
            if not isinstance(g_ids, list):
                try:
                    g_ids = list(g_ids)
                except Exception:
                    g_ids = []
            seq = p_ids + g_ids
            pad_len = max_len - len(seq)
            indices_list.append(seq + [pad_id] * max(pad_len, 0))
            mask = [False] * len(p_ids) + [True] * len(g_ids) + [False] * max(pad_len, 0)
            masks_list.append(mask)
        if len(indices_list) == 0:
            indices = torch.empty((0, 0), dtype=torch.long, device="cpu")
            masks = torch.empty((0, 0), dtype=torch.bool, device="cpu")
        else:
            # Build tensors on CPU to reduce peak GPU memory during rollout
            indices = torch.tensor(indices_list, dtype=torch.long, device="cpu")
            masks = torch.tensor(masks_list, dtype=torch.bool, device="cpu")

        if verbose:
            return completions, indices, masks, vllm_log_probs_list
        else:
            return completions



    def sync_lora_to_vllm(self, adapter_name: str = "peft") -> bool:
        """Export current PEFT LoRA adapter and load it into vLLM engine.

        Returns True if successfully loaded; False otherwise.
        """
        if not self.use_vllm or (self.vllm_worker is None and self.vllm_engine is None):
            return False
        if self.lora_config is None:
            # No LoRA applied on HF model
            return False
        # Save current LoRA adapter to a temp dir (PEFT format expected by vLLM)
        try:
            tmp_root = os.path.join(tempfile.gettempdir(), "vllm_lora_adapters")
            os.makedirs(tmp_root, exist_ok=True)
            adapter_dir = os.path.join(tmp_root, f"{adapter_name}")
            # Always overwrite to keep latest weights
            if not os.path.isdir(adapter_dir):
                os.makedirs(adapter_dir, exist_ok=True)
            # For PEFT models, this writes adapter_model.* and adapter_config.json
            self.model.save_pretrained(adapter_dir)
        except Exception as _e:
            print(f"[vLLM/LoRA] Failed to save adapter: {_e}")
            return False
        # Record adapter info for per-request LoRA injection
        self.vllm_lora_adapter_name = adapter_name
        self.vllm_lora_adapter_path = adapter_dir
        self.vllm_lora_loaded = True
        print(f"[vLLM/LoRA] Prepared adapter '{adapter_name}' at {adapter_dir}")
        if self.vllm_worker is not None:
            try:
                self.vllm_worker.set_lora_adapter(adapter_name, adapter_dir, load_into_engine=True)
            except Exception as _e:
                print(f"[vLLM/LoRA] Worker-level load failed: {_e}")
        elif self.vllm_engine is not None:
            try:
                if hasattr(self.vllm_engine, "load_lora_adapter"):
                    self.vllm_engine.load_lora_adapter(adapter_name, adapter_dir)  # type: ignore
                    print("[vLLM/LoRA] Also loaded adapter into engine via load_lora_adapter")
            except Exception as _e:
                print(f"[vLLM/LoRA] Engine-level load (optional) failed: {_e}")
        return True


    def shutdown_vllm(self) -> None:
        """Stop worker or inline vLLM engine if running."""
        if self.vllm_worker is not None:
            try:
                self.vllm_worker.shutdown()
            except Exception:
                pass
            self.vllm_worker = None
        if self.vllm_engine is not None:
            try:
                shutdown = getattr(self.vllm_engine, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            except Exception:
                pass
            self.vllm_engine = None

    def __del__(self) -> None:
        try:
            self.shutdown_vllm()
        except Exception:
            pass

    def save(self, path: str) -> None:
        """Save checkpoint.
        - If LoRA: export PEFT adapter in root dir (vLLM-compatible), plus tokenizer and training meta.
        - If FacT: save adapter-only weights.
        - Else: save full model snapshot (config + state_dict).
        """
        os.makedirs(path, exist_ok=True)

        # Always save the tokenizer
        try:
            self.tokenizer.save_pretrained(path)
        except Exception:
            pass

        # Save training meta separately to avoid clobbering PEFT adapter_config.json
        meta = {
            "original_model_path": self.original_model_path,
            "fact_config": self.fact_config,
            "lora_config": self.lora_config,
        }
        try:
            with open(os.path.join(path, "training_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

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
            return

        # LoRA Mode - Export PEFT adapter (vLLM-compatible)
        if self.lora_config is not None:
            try:
                self.model.save_pretrained(path)  # PeftModel: writes adapter_model.* + adapter_config.json
                print(f"LoRA adapter saved in PEFT format at {path}")
            except Exception as _e:
                print(f"Warning: failed to save PEFT LoRA adapter: {_e}")
            return

        # Full model snapshot if no adapters are used
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        try:
            self.model.config.save_pretrained(path)
        except Exception:
            pass
        torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        print("Model saved (manual state_dict)")

    # -------------------- vLLM hot update (dense/FacT, TP=1) --------------------
    def _get_vllm_internal_model(self):
        """Locate vLLM's internal model object for load_weights.

        Returns None if vLLM engine is not available.
        """
        if not (self.use_vllm and self.vllm_engine is not None):
            return None
        candidates = [
            # Common paths across vLLM versions
            "llm_engine.model_executor.driver_worker.worker.model_runner.model",
            "llm_engine.model_executor.driver_worker.model_runner.model",
            "llm_engine.model_executor.model_runner.model",
            # Some builds expose an alias `engine`
            "engine.llm_engine.model_executor.driver_worker.worker.model_runner.model",
            "engine.model_executor.driver_worker.worker.model_runner.model",
        ]
        obj = self.vllm_engine
        for path in candidates:
            cur = obj
            ok = True
            for seg in path.split("."):
                if not hasattr(cur, seg):
                    ok = False
                    break
                cur = getattr(cur, seg)
            if ok:
                return cur
        # Fallback: bounded BFS to locate an object exposing `load_weights`
        try:
            from collections import deque

            def has_loader(x):
                try:
                    return hasattr(x, "load_weights") and callable(getattr(x, "load_weights"))
                except Exception:
                    return False

            seen = set()
            q = deque()
            seed = [obj]
            if hasattr(obj, "llm_engine"):
                seed.append(getattr(obj, "llm_engine"))
            for s in seed:
                q.append((s, 0))
                try:
                    seen.add(id(s))
                except Exception:
                    pass

            MAX_DEPTH = 4
            MAX_NODES = 2048
            visited_nodes = 0

            while q and visited_nodes < MAX_NODES:
                cur, depth = q.popleft()
                visited_nodes += 1
                if has_loader(cur):
                    return cur
                if depth >= MAX_DEPTH:
                    continue
                # Iterate attributes conservatively
                try:
                    for name in dir(cur):
                        if name.startswith("__"):
                            continue
                        # Skip obvious huge or irrelevant attrs
                        if name in ("__dict__", "__class__", "__weakref__"):
                            continue
                        try:
                            nxt = getattr(cur, name)
                        except Exception:
                            continue
                        # Skip basic types
                        if isinstance(nxt, (int, float, str, bytes, bool, tuple, list, dict, set)):
                            continue
                        # Skip callables
                        if callable(nxt):
                            continue
                        nid = None
                        try:
                            nid = id(nxt)
                        except Exception:
                            nid = None
                        if nid is not None and nid in seen:
                            continue
                        if nid is not None:
                            seen.add(nid)
                        q.append((nxt, depth + 1))
                except Exception:
                    continue
        except Exception:
            pass
        return None

    @torch.no_grad()
    def hot_update_vllm_full_weights(self) -> bool:
        """Hot update vLLM weights in-place (no engine restart).

        - Dense/FacT: stream tensors per-parameter to worker via update_weights (OpenRLHF-style).

        Only recommended when tensor_parallel_size == 1 and engine colocates in the same process.
        Returns True if successfully dispatched to vLLM.
        """

        if self.vllm_worker is not None:
            # Build (name, tensor) pairs. For FacT, merge on the fly for target Linear, skip fact_*.
            sd = self.model.state_dict()
            weights: List[Tuple[str, torch.Tensor]] = []

            # Collect FacT merged weights first if FacT is enabled
            fact_wrapped: Dict[str, SharedFacTLinear] = {}
            if self.fact_config is not None:
                for name, module in self.model.named_modules():
                    if isinstance(module, SharedFacTLinear):
                        fact_wrapped[name] = module

            visited: set[str] = set()

            # 1) Emit merged weights for FacT-wrapped Linear modules
            for mname, mod in fact_wrapped.items():
                original = mod.original_layer
                W_orig = original.weight.detach()
                device = W_orig.device
                dtype = W_orig.dtype
                U = mod.fact_u.weight.detach().to(device=device, dtype=dtype)
                T = mod.fact_t.weight.detach().to(device=device, dtype=dtype)
                V = mod.fact_v.weight.detach().to(device=device, dtype=dtype)
                scaling = float(mod.scale) * (float(mod.alpha) / max(1.0, float(mod.rank)))
                W_add = (V @ T @ U) * scaling
                W_eff = W_orig + W_add
                weight_key = f"{mname}.weight"
                weights.append((weight_key, W_eff.detach().to("cpu").contiguous()))
                visited.add(weight_key)
                if original.bias is not None:
                    bias_key = f"{mname}.bias"
                    weights.append((bias_key, original.bias.detach().to("cpu").contiguous()))
                    visited.add(bias_key)

            # 2) Emit the rest tensors (skip FacT internals and original_layer.*)
            for k, v in sd.items():
                if ".fact_" in k or ".original_layer." in k:
                    continue
                if k in visited:
                    continue
                weights.append((k, v.detach().to("cpu").contiguous()))

            try:
                # Send to worker for in-place load
                # Large lists: chunk to avoid IPC payload limits
                CHUNK = 1024
                for i in range(0, len(weights), CHUNK):
                    chunk = weights[i:i+CHUNK]
                    self.vllm_worker._request({"op": "update_weights", "weights": chunk})
                # Reset prefix cache if any lingering state
                try:
                    self.vllm_worker._request({"op": "reset_prefix_cache"})  # optional, ignore if unknown
                except Exception:
                    pass
                return True
            except Exception as exc:
                print(f"[vLLM] worker hot update failed: {exc}")
                return False

        internal_model = self._get_vllm_internal_model()
        if internal_model is None:
            return False

        # Optional: For MoE models you may need to patch weight_loader
        try:
            # Import from vendored VERL if available
            from reference_works.verl.verl.utils.vllm.patch import (
                patch_vllm_moe_model_weight_loader,
            )

            try:
                patch_vllm_moe_model_weight_loader(internal_model)
            except Exception:
                pass
        except Exception:
            # Not required for Qwen/LLaMA/Phi dense models
            pass

        # Build map of FacT wrapped modules so we can emit merged weights with original HF keys
        fact_wrapped: Dict[str, SharedFacTLinear] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, SharedFacTLinear):
                fact_wrapped[name] = module

        sd = self.model.state_dict()
        visited: set[str] = set()

        # Inspect internal model's expected keys to build a mapper
        try:
            sd_keys = list(getattr(internal_model, "state_dict")().keys())  # type: ignore[attr-defined]
        except Exception:
            sd_keys = []

        def _has_prefix(prefix: str) -> bool:
            return any(k.startswith(prefix) for k in sd_keys)

        layers_top = _has_prefix("layers.")
        layers_under_model = (not layers_top) and _has_prefix("model.layers.")
        embed_top = _has_prefix("embed_tokens.")
        embed_under_model = (not embed_top) and _has_prefix("model.embed_tokens.")
        norm_top = _has_prefix("norm.")
        norm_under_model = (not norm_top) and _has_prefix("model.norm.")

        def _map_key(k: str) -> str:
            # Normalize some common HF prefixes
            if k.startswith("transformer."):
                k = "model." + k[len("transformer."):]
            if layers_top and k.startswith("model.layers."):
                k = k[len("model."):]
            elif layers_under_model and k.startswith("layers."):
                k = "model." + k
            if embed_top and k.startswith("model.embed_tokens."):
                k = k[len("model."):]
            elif embed_under_model and k.startswith("embed_tokens."):
                k = "model." + k
            if norm_top and k.startswith("model.norm."):
                k = k[len("model."):]
            elif norm_under_model and k.startswith("norm."):
                k = "model." + k
            return k

        sd_set = set(sd_keys)

        # If internal expects fused QKV, fuse when possible
        expect_fused_qkv = any(key.endswith("self_attn.qkv_proj.weight") for key in sd_keys)

        def gen():
            # 1) Emit merged weights for FacT-wrapped Linear modules
            for mname, mod in fact_wrapped.items():
                original = mod.original_layer
                W_orig = original.weight.detach()
                device = W_orig.device
                dtype = W_orig.dtype
                U = mod.fact_u.weight.detach().to(device=device, dtype=dtype)
                T = mod.fact_t.weight.detach().to(device=device, dtype=dtype)
                V = mod.fact_v.weight.detach().to(device=device, dtype=dtype)
                scaling = float(mod.scale) * (float(mod.alpha) / max(1.0, float(mod.rank)))
                W_add = (V @ T @ U) * scaling
                W_eff = W_orig + W_add

                weight_key = _map_key(f"{mname}.weight")
                yield weight_key, W_eff.detach().cpu().contiguous()
                visited.add(weight_key)

                if original.bias is not None:
                    bias_key = _map_key(f"{mname}.bias")
                    yield bias_key, original.bias.detach().cpu().contiguous()
                    visited.add(bias_key)

            # 2) Emit the rest tensors with mapping and optional QKV fusion
            # For fusion, buffer q/k/v per layer prefix
            fuse_buckets: Dict[str, Dict[str, torch.Tensor]] = {}

            def _flush_fused(prefix: str):
                bucket = fuse_buckets.get(prefix)
                if not bucket or not all(x in bucket for x in ("q", "k", "v")):
                    return False
                try:
                    qkv = torch.cat([bucket["q"], bucket["k"], bucket["v"]], dim=0)
                except Exception:
                    return False
                fused_key = f"{prefix}.self_attn.qkv_proj.weight"
                mapped = _map_key(fused_key)
                if mapped in sd_set:
                    yield mapped, qkv.detach().cpu().contiguous()
                    return True
                return False

            for k, v in sd.items():
                if ".fact_" in k or ".original_layer." in k:
                    continue
                mapped = _map_key(k)
                if mapped in visited:
                    continue
                # QKV fusion handling
                if expect_fused_qkv and (
                    mapped.endswith(".self_attn.q_proj.weight") or
                    mapped.endswith(".self_attn.k_proj.weight") or
                    mapped.endswith(".self_attn.v_proj.weight")
                ):
                    try:
                        base, tail = mapped.split(".self_attn.", 1)
                        if tail.startswith("q_proj.weight"):
                            fuse_buckets.setdefault(base, {})["q"] = v.detach()
                        elif tail.startswith("k_proj.weight"):
                            fuse_buckets.setdefault(base, {})["k"] = v.detach()
                        elif tail.startswith("v_proj.weight"):
                            fuse_buckets.setdefault(base, {})["v"] = v.detach()
                        # Try to flush when complete
                        res = False
                        for out in _flush_fused(base) or []:
                            yield out  # type: ignore[misc]
                            res = True
                        if res:
                            visited.add(f"{base}.self_attn.qkv_proj.weight")
                        continue
                    except Exception:
                        pass
                # Default: emit mapped key if present in internal state
                if sd_set and mapped not in sd_set:
                    # Skip keys that internal model doesn't have
                    continue
                tensor = v.detach().cpu().contiguous()
                yield mapped, tensor

            # Flush any remaining incomplete fusion buckets by emitting separate keys
            if expect_fused_qkv:
                for base, bucket in fuse_buckets.items():
                    for tag in ("q", "k", "v"):
                        if tag in bucket:
                            tail = {"q": "q_proj.weight", "k": "k_proj.weight", "v": "v_proj.weight"}[tag]
                            mapped = _map_key(f"{base}.self_attn.{tail}")
                            if not sd_set or mapped in sd_set:
                                yield mapped, bucket[tag].detach().cpu().contiguous()

        internal_model.load_weights(gen())

        # Best-effort clear prefix cache if supported by engine to avoid stale KV
        try:
            if hasattr(self.vllm_engine, "reset_prefix_cache"):
                self.vllm_engine.reset_prefix_cache()
        except Exception:
            pass
        return True


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
