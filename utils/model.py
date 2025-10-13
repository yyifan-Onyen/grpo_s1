from typing import List, Dict, Tuple, Optional, Union
import torch
import json
import os
import tempfile
import shutil
import time
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = target_device
        
        # Load model with attention backend selection (auto prefers flash_attention_2)
        def _load_with_attn(impl: str):
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
        
        # If target device is CPU, don't specify device_map but move manually
        if target_device == "cpu":
            self.model = self.model.to("cpu")
        
        print(f"Model {model_path} loaded on device: {target_device}")
        
        # Configure gradient checkpointing if enabled
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            try:
                # Ensure inputs require grad so checkpoint works end-to-end
                self.model.enable_input_require_grads()
            except Exception:
                pass
            print(f"Gradient checkpointing enabled for {model_path}")
        
        # Initialize vLLM desire early; FacT may disable it for rollout
        self.use_vllm = bool(use_vllm and _VLLM_AVAILABLE)

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
            # vLLM cannot inject FacT online; disable vLLM for rollout
            if self.use_vllm:
                print("[FacT] vLLM rollout requires merged weights; call refresh_vllm_merged_engine() to rebuild the engine.")
                self.use_vllm = False
        
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
        # vLLM target GPU local index (within CUDA_VISIBLE_DEVICES) or absolute id if not set
        self.vllm_target_gpu = vllm_gpu_id
        self.vllm_engine = None
        if self.use_vllm:
            try:
                # Select target GPU for vLLM engine by scoping CUDA_VISIBLE_DEVICES to only that GPU
                prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                target_cvd = None
                if vllm_gpu_memory_utilization is not None or True:
                    pass  # placeholder to keep logical block aligned
                # try map vllm gpu id if provided via kwargs
                vllm_gpu_id = locals().get("vllm_gpu_id", None)
                # locals() may not include it reliably; instead, rely on function arg name
                # We explicitly captured via parameter name above; so we can use it directly
                try:
                    _vllm_gpu_id = vllm_gpu_memory_utilization  # dummy to silence lints (no-op)
                except Exception:
                    pass
                # Proper mapping using parameter value
                # Note: we intentionally refer to the function parameter 'vllm_gpu_memory_utilization' above, not id
                # Here we recompute from closure: use_vllm, vllm_gpu_memory_utilization already used; now find id from 'vllm_gpu_id'
                # Since Python doesn't allow shadowing easily in this diff, we reconstruct from self attributes if set later
                
                if torch_dtype == torch.bfloat16:
                    _dtype = "bfloat16"
                elif torch_dtype == torch.float16:
                    _dtype = "float16"
                elif torch_dtype == torch.float32:
                    _dtype = "float32"
                else:
                    _dtype = "auto"
                _gpu_util = 0.6 if vllm_gpu_memory_utilization is None else float(vllm_gpu_memory_utilization)
                # Map GPU for vLLM by local index within current visible devices if 'self.vllm_target_gpu' exists
                target_gpu_index = getattr(self, "vllm_target_gpu", None)
                try:
                    if target_gpu_index is not None:
                        cur_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                        if cur_cvd:
                            lst = [p.strip() for p in cur_cvd.split(",") if p.strip()]
                            if 0 <= int(target_gpu_index) < len(lst):
                                target_cvd = lst[int(target_gpu_index)]
                            else:
                                target_cvd = str(target_gpu_index)
                        else:
                            target_cvd = str(target_gpu_index)
                        os.environ["CUDA_VISIBLE_DEVICES"] = target_cvd
                except Exception:
                    target_cvd = None
                try:
                    # Try both max_seq_len and max_model_len for compatibility across vLLM versions
                    _kwargs = dict(
                        model=model_path,
                        dtype=_dtype,
                        tensor_parallel_size=1,
                        trust_remote_code=True,
                        gpu_memory_utilization=_gpu_util,
                        enable_lora=True,
                        enforce_eager=True,  # disable CUDA graph to avoid illegal memory access issues
                    )
                    if vllm_max_model_len is not None:
                        try:
                            self.vllm_engine = VLLMEngine(**_kwargs, max_seq_len=int(vllm_max_model_len))
                        except TypeError:
                            self.vllm_engine = VLLMEngine(**_kwargs, max_model_len=int(vllm_max_model_len))
                    else:
                        self.vllm_engine = VLLMEngine(**_kwargs)
                finally:
                    # restore original visibility
                    if target_cvd is not None:
                        if prev_cvd is None:
                            try:
                                del os.environ["CUDA_VISIBLE_DEVICES"]
                            except Exception:
                                pass
                        else:
                            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
                print(f"vLLM engine enabled for {model_path}")
            except Exception as _e:
                print(f"[vLLM] init failed for {model_path}: {_e}. Fallback to HF generate.")
                self.use_vllm = False

        # vLLM LoRA state
        self.vllm_lora_adapter_name: Optional[str] = None
        self.vllm_lora_adapter_path: Optional[str] = None
        self.vllm_lora_loaded: bool = False
        # Track last FacT-merged directory for vLLM refresh cycles
        self._fact_last_merged_dir: Optional[str] = None

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
        """Generate text completions based on the provided prompts."""
        conversations = []
        for prompt in prompts:
            conversations.append(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ))

        if self.use_vllm and self.vllm_engine is not None:
            # vLLM SamplingParams(logprobs=k) returns top-k logprobs per token; set k=1 to get chosen-token logprob
            try:
                sampling = VLLMSamplingParams(
                    temperature=float(temperature),
                    max_tokens=int(limitation),
                    n=1,
                    logprobs=1 if return_log_probs else None,
                )
            except TypeError:
                # Fallback for older signatures without logprobs
                sampling = VLLMSamplingParams(
                    temperature=float(temperature),
                    max_tokens=int(limitation),
                    n=1,
                )
            # Attach LoRA request if we have prepared adapter for vLLM
            lora_kwargs = {}
            if (
                VLLMLoRARequest is not None and
                isinstance(self.vllm_lora_adapter_name, str) and
                isinstance(self.vllm_lora_adapter_path, str)
            ):
                try:
                    lora_kwargs["lora_request"] = VLLMLoRARequest(
                        lora_name=self.vllm_lora_adapter_name,
                        lora_int_id=abs(hash(self.vllm_lora_adapter_name)) % (10**9),
                        lora_path=self.vllm_lora_adapter_path,
                    )
                except Exception:
                    pass
            with torch.no_grad():
                try:
                    outputs = self.vllm_engine.generate(conversations, sampling, **lora_kwargs)
                except TypeError:
                    # Older vLLM without lora_request in generate; fall back to no-arg call (adapter may be set globally)
                    outputs = self.vllm_engine.generate(conversations, sampling)
            completions = [o.outputs[0].text for o in outputs]
            # Prefer token ids from vLLM if available to avoid retokenization mismatch
            # Batch-tokenize conversation prefixes to reduce per-item overhead
            try:
                _prefix_tok = self.tokenizer(conversations, add_special_tokens=False, return_tensors=None)
                if isinstance(_prefix_tok, dict) and "input_ids" in _prefix_tok:
                    prefix_ids_list = _prefix_tok["input_ids"]
                    if hasattr(prefix_ids_list, "tolist"):
                        # torch / numpy tensor -> list
                        prefix_ids_list = prefix_ids_list.tolist()
                    elif isinstance(prefix_ids_list, list):
                        # already a python list-of-lists; keep as-is
                        pass
                    else:
                        # unknown structure
                        prefix_ids_list = []
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
            gen_ids_list = []
            vllm_log_probs_list: Optional[List[torch.Tensor]] = [] if return_log_probs else None
            for conv, out in zip(conversations, outputs):
                try:
                    gen_token_ids = out.outputs[0].token_ids
                except Exception:
                    # Fallback: retokenize completion text
                    gen_token_ids = self.tokenizer(out.outputs[0].text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
                gen_ids_list.append(gen_token_ids)
                if return_log_probs:
                    try:
                        # vLLM returns per-token logprobs for generated tokens if requested
                        token_logprobs = []
                        for tk in out.outputs[0].logprobs:
                            # logprobs is a list per token; take the chosen token's logprob if present
                            if tk is None:
                                token_logprobs.append(float("nan"))
                            else:
                                # tk could be a dict mapping token->logprob or a structure with .logprob
                                if hasattr(tk, 'logprob'):
                                    token_logprobs.append(float(tk.logprob))
                                elif isinstance(tk, dict):
                                    # find the token id we chose
                                    chosen_id = gen_token_ids[len(token_logprobs)] if len(gen_token_ids) > len(token_logprobs) else None
                                    if chosen_id is not None and chosen_id in tk:
                                        token_logprobs.append(float(tk[chosen_id]))
                                    else:
                                        # take max
                                        token_logprobs.append(float(max(tk.values())))
                                else:
                                    # Unknown structure; skip
                                    token_logprobs.append(float("nan"))
                        vllm_log_probs_list.append(torch.tensor(token_logprobs, dtype=torch.float32, device=self.device))
                    except Exception:
                        vllm_log_probs_list.append(None)  # type: ignore
            if not prefix_ids_list or not gen_ids_list:
                max_len = 0
            else:
                max_len = max(len(p) + len(g) for p, g in zip(prefix_ids_list, gen_ids_list))
            pad_id = self.tokenizer.pad_token_id
            indices_list = []
            masks_list = []
            for p_ids, g_ids in zip(prefix_ids_list, gen_ids_list):
                seq = p_ids + g_ids
                pad_len = max_len - len(seq)
                indices_list.append(seq + [pad_id] * max(pad_len, 0))
                mask = [False] * len(p_ids) + [True] * len(g_ids) + [False] * max(pad_len, 0)
                masks_list.append(mask)
            if len(indices_list) == 0:
                indices = torch.empty((0, 0), dtype=torch.long, device=self.device)
                masks = torch.empty((0, 0), dtype=torch.bool, device=self.device)
            else:
                indices = torch.tensor(indices_list, dtype=torch.long, device=self.device)
                masks = torch.tensor(masks_list, dtype=torch.bool, device=self.device)
        else:
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

            # Temporarily switch to inference-friendly mode to avoid OOM with gradient checkpointing
            prev_training = self.model.training
            prev_use_cache = getattr(self.model.config, "use_cache", True)
            had_gc = getattr(self.model, "is_gradient_checkpointing", False)

            self.model.eval()
            try:
                self.model.gradient_checkpointing_disable()
            except Exception:
                pass
            try:
                self.model.config.use_cache = True
            except Exception:
                pass

            with torch.inference_mode():
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

            # Restore training-time settings
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
            vllm_log_probs_list = None

        if verbose:
            return completions, indices, masks, vllm_log_probs_list
        else:
            return completions

    def sync_lora_to_vllm(self, adapter_name: str = "peft") -> bool:
        """Export current PEFT LoRA adapter and load it into vLLM engine.

        Returns True if successfully loaded; False otherwise.
        """
        if not (self.use_vllm and self.vllm_engine is not None):
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
        self.vllm_lora_loaded = VLLMLoRARequest is not None
        print(f"[vLLM/LoRA] Prepared adapter '{adapter_name}' at {adapter_dir}")
        # Best-effort engine-level load for older vLLM; optional
        try:
            if hasattr(self.vllm_engine, "load_lora_adapter"):
                self.vllm_engine.load_lora_adapter(adapter_name, adapter_dir)  # type: ignore
                print("[vLLM/LoRA] Also loaded adapter into engine via load_lora_adapter")
        except Exception as _e:
            print(f"[vLLM/LoRA] Engine-level load (optional) failed: {_e}")
        return True


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
