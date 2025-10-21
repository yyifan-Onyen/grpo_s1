"""Spawn-based vLLM worker to isolate CUDA device selection per model."""
from __future__ import annotations

import os
import traceback
import multiprocessing as mp
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

import torch

try:  # Optional safetensors support; fall back to torch.load.
    from safetensors.torch import load_file as safetensors_load  # type: ignore
except Exception:  # pragma: no cover
    safetensors_load = None  # type: ignore


def _serialize_logprobs(raw: Optional[List[Any]], token_ids: List[int]) -> Optional[List[Optional[float]]]:
    """Convert vLLM logprobs to a serialisable list."""
    if raw is None:
        return None
    serialised: List[Optional[float]] = []
    for idx, tk in enumerate(raw):
        if tk is None:
            serialised.append(None)
            continue
        try:
            if hasattr(tk, "logprob"):
                serialised.append(float(tk.logprob))
            elif isinstance(tk, dict) and token_ids and idx < len(token_ids):
                chosen = token_ids[idx]
                if chosen in tk:
                    serialised.append(float(tk[chosen]))
                elif tk:
                    serialised.append(float(max(tk.values())))
                else:
                    serialised.append(None)
            else:
                serialised.append(None)
        except Exception:
            serialised.append(None)
    return serialised


def _find_internal_model(engine: Any) -> Optional[Any]:
    """Locate a vLLM-internal object exposing `load_weights`.

    Tries known attribute paths for common vLLM versions, then falls back to a bounded BFS.
    """
    # 1) Try explicit candidate paths (adapted from utils/model.py)
    candidates = [
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.model_runner.model",
        # Some builds expose an alias `engine`
        "engine.llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "engine.model_executor.driver_worker.worker.model_runner.model",
    ]

    def _resolve_path(obj: Any, path: str) -> Optional[Any]:
        cur = obj
        for seg in path.split("."):
            if not hasattr(cur, seg):
                return None
            try:
                cur = getattr(cur, seg)
            except Exception:
                return None
        return cur

    for p in candidates:
        target = _resolve_path(engine, p)
        if target is not None and hasattr(target, "load_weights") and callable(getattr(target, "load_weights")):
            return target

    # 2) Fallback: bounded BFS over attributes
    try:
        seen = set()
        q = deque([(engine, 0)])
        while q:
            cur, depth = q.popleft()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            try:
                if hasattr(cur, "load_weights") and callable(getattr(cur, "load_weights")):
                    return cur
            except Exception:
                pass
            if depth >= 5:  # search one level deeper than before
                continue
            for attr in dir(cur):
                if attr.startswith("__"):
                    continue
                # Skip obvious large/irrelevant attributes
                if attr in ("__dict__", "__class__", "__weakref__"):
                    continue
                try:
                    nxt = getattr(cur, attr)
                except Exception:
                    continue
                if isinstance(nxt, (int, float, str, bytes, bool)):
                    continue
                # Allow tuples/lists/dicts by enqueuing their elements conservatively
                if isinstance(nxt, (tuple, list, set)):
                    for item in list(nxt)[:8]:
                        try:
                            if id(item) not in seen and not callable(item):
                                q.append((item, depth + 1))
                        except Exception:
                            continue
                    continue
                if isinstance(nxt, dict):
                    for k, v in list(nxt.items())[:8]:
                        try:
                            if id(v) not in seen and not callable(v):
                                q.append((v, depth + 1))
                        except Exception:
                            continue
                    continue
                if callable(nxt):
                    continue
                q.append((nxt, depth + 1))
    except Exception:  # pragma: no cover
        pass
    return None


def _iter_weights_from_dir(model_dir: str, key_mapper: Optional[Any] = None) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield tensors from safetensors/torch binaries in directory order.

    Optionally apply a `key_mapper(name:str)->str` to adapt HF keys to
    vLLM's internal naming for specific architectures/versions.
    """
    tensor_files: List[str] = []
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name.endswith((".safetensors", ".bin")):
                tensor_files.append(os.path.join(root, name))
    tensor_files.sort()

    if key_mapper is None:
        def key_mapper(x: str) -> str:  # type: ignore
            return x

    for path in tensor_files:
        if path.endswith(".safetensors") and safetensors_load is not None:
            tensors = safetensors_load(path)  # type: ignore[arg-type]
        else:
            tensors = torch.load(path, map_location="cpu")
        try:
            for key, value in tensors.items():
                if torch.is_tensor(value):
                    out_key = key_mapper(key)
                    yield out_key, value.detach().cpu().contiguous()
        finally:
            del tensors


def _worker_main(cmd_queue: mp.Queue, resp_queue: mp.Queue, config: Dict[str, Any]) -> None:
    """Entry point for the worker process."""
    try:
        visible_cuda = config.get("visible_cuda")
        if visible_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_cuda)

        from vllm import LLM, SamplingParams  # pylint: disable=import-error
        try:
            from vllm.lora.request import LoRARequest  # type: ignore
        except Exception:  # pragma: no cover
            try:
                from vllm import LoRARequest  # type: ignore
            except Exception:
                LoRARequest = None  # type: ignore

        engine_kwargs = dict(
            model=config["model_path"],
            dtype=config["dtype"],
            tensor_parallel_size=1,
            trust_remote_code=config.get("trust_remote_code", True),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.6)),
            enable_lora=config.get("enable_lora", True),
            enforce_eager=config.get("enforce_eager", True),
        )
        max_model_len = config.get("max_model_len")
        if max_model_len is not None:
            try:
                engine = LLM(**engine_kwargs, max_seq_len=int(max_model_len))
            except TypeError:
                engine = LLM(**engine_kwargs, max_model_len=int(max_model_len))
        else:
            engine = LLM(**engine_kwargs)

        state: Dict[str, Optional[str]] = {"lora_name": None, "lora_path": None}
        resp_queue.put({"status": "ready", "device": visible_cuda, "pid": os.getpid()})

        while True:
            cmd = cmd_queue.get()
            op = cmd.get("op")
            if op == "generate":
                sampling = SamplingParams(**cmd["sampling"])
                use_lora = cmd.get("use_lora", False)
                lora_request = None
                if use_lora and state["lora_name"] and state["lora_path"] and 'LoRARequest' in locals():
                    try:
                        lora_request = LoRARequest(
                            lora_name=state["lora_name"],
                            lora_int_id=abs(hash(state["lora_name"])) % (10**9),
                            lora_path=state["lora_path"],
                        )
                    except Exception:
                        lora_request = None
                outputs = engine.generate(
                    cmd["conversations"],
                    sampling,
                    **({"lora_request": lora_request} if lora_request is not None else {}),
                )
                payload = []
                for item in outputs:
                    slot = item.outputs[0]
                    token_ids = list(slot.token_ids)
                    payload.append(
                        {
                            "text": slot.text,
                            "token_ids": token_ids,
                            "logprobs": _serialize_logprobs(slot.logprobs, token_ids),
                        }
                    )
                resp_queue.put({"status": "ok", "result": payload})
            elif op == "load_weights":
                model_dir = cmd["model_dir"]
                try:
                    internal_model = _find_internal_model(engine)
                    if internal_model is not None:
                        # 直接使用内部模型热更新（基于 state_dict 键前缀更稳健地判定映射）
                        try:
                            # 1) 读取内部模型的键，推断 vLLM loader 期望的前缀
                            try:
                                sd_keys = list(getattr(internal_model, "state_dict")().keys())  # type: ignore
                            except Exception:
                                sd_keys = []

                            def _has_prefix(prefix: str) -> bool:
                                return any(k.startswith(prefix) for k in sd_keys)

                            # 目标前缀偏好：如果存在顶层 layers.* 则使用顶层，否则使用 model.layers.*
                            layers_top = _has_prefix("layers.")
                            layers_under_model = (not layers_top) and _has_prefix("model.layers.")
                            # embed/norm 同理
                            embed_top = _has_prefix("embed_tokens.")
                            embed_under_model = (not embed_top) and _has_prefix("model.embed_tokens.")
                            norm_top = _has_prefix("norm.")
                            norm_under_model = (not norm_top) and _has_prefix("model.norm.")

                            # 打印一次关键拓扑信息，方便定位
                            try:
                                cls_name = type(internal_model).__name__
                            except Exception:
                                cls_name = str(internal_model)
                            print(
                                f"[vLLM/worker] load_weights target={cls_name}, "
                                f"layers_top={layers_top}, layers_under_model={layers_under_model}, "
                                f"embed_top={embed_top}, embed_under_model={embed_under_model}, "
                                f"norm_top={norm_top}, norm_under_model={norm_under_model}",
                                flush=True,
                            )

                            # 针对 Qwen/Llama 且 vLLM < 0.8.5 强制使用顶层前缀（剥掉 model.）
                            try:
                                import vllm as _vllm_mod  # type: ignore
                                from packaging import version as _v
                                _vllm_ver = _v.parse(getattr(_vllm_mod, "__version__", "0.0.0"))
                            except Exception:
                                _vllm_ver = None
                            if _vllm_ver is not None:
                                force_top = ("Qwen" in cls_name or "Llama" in cls_name) and (_vllm_ver < _v.parse("0.8.5"))
                            else:
                                force_top = ("Qwen" in cls_name or "Llama" in cls_name)
                            if force_top:
                                if layers_under_model and not layers_top:
                                    layers_top = True
                                    layers_under_model = False
                                # 对 embed/norm 同步为顶层（若当前在 model. 下）
                                if embed_under_model and not embed_top:
                                    embed_top = True
                                    embed_under_model = False
                                if norm_under_model and not norm_top:
                                    norm_top = True
                                    norm_under_model = False
                                print("[vLLM/worker] force top-level prefixes for Qwen/Llama (<0.8.5)", flush=True)

                            # 不进行模型家族的强制覆盖；完全依赖 state_dict 动态判定，
                            # 以适配 vLLM v0.8.4 这类仅存在 model.layers.* 的结构。

                            def _map_key(k: str) -> str:
                                # 规范化 transformer. → model.
                                if k.startswith("transformer."):
                                    k = "model." + k[len("transformer."):]

                                # 层级映射：layers
                                if layers_top:
                                    if k.startswith("model.layers."):
                                        k = k[len("model."):]
                                elif layers_under_model:
                                    if k.startswith("layers."):
                                        k = "model." + k

                                # 嵌入映射：embed_tokens
                                if embed_top:
                                    if k.startswith("model.embed_tokens."):
                                        k = k[len("model.") :]
                                elif embed_under_model:
                                    if k.startswith("embed_tokens."):
                                        k = "model." + k

                                # 归一化层映射：norm（如最终层归一化）
                                if norm_top:
                                    if k.startswith("model.norm."):
                                        k = k[len("model.") :]
                                elif norm_under_model:
                                    if k.startswith("norm."):
                                        k = "model." + k

                                return k

                            # 少量预览映射（不打印张量，仅打印名称），帮助排查
                            try:
                                preview = []
                                cnt = 0
                                for root, _, files in os.walk(model_dir):
                                    for name in files:
                                        if name.endswith((".safetensors", ".bin")):
                                            # 只取前2个文件的前若干键做映射预览
                                            if cnt >= 2:
                                                break
                                            path = os.path.join(root, name)
                                            try:
                                                if path.endswith(".safetensors") and safetensors_load is not None:
                                                    tensors = safetensors_load(path)  # type: ignore[arg-type]
                                                else:
                                                    tensors = torch.load(path, map_location="cpu")
                                                i = 0
                                                for key in tensors.keys():
                                                    preview.append((key, _map_key(key)))
                                                    i += 1
                                                    if i >= 5:
                                                        break
                                                del tensors
                                            except Exception:
                                                pass
                                            cnt += 1
                                    if cnt >= 2:
                                        break
                                if preview:
                                    print("[vLLM/worker] key mapping preview (src -> dst):", flush=True)
                                    for src, dst in preview[:10]:
                                        print(f"  {src}  ->  {dst}", flush=True)
                            except Exception:
                                pass

                            # 迭代加载（支持 QKV 融合：将 q/k/v 三权重沿 dim=0 拼为 qkv_proj）
                            def _iter_with_qkv_fusion():
                                fuse_buckets: Dict[str, Dict[str, torch.Tensor]] = {}
                                def maybe_emit_fused(prefix: str):
                                    bucket = fuse_buckets.get(prefix)
                                    if not bucket:
                                        return None
                                    if all(x in bucket for x in ("q", "k", "v")):
                                        try:
                                            qkv = torch.cat([bucket["q"], bucket["k"], bucket["v"]], dim=0)
                                            fused_key = f"{prefix}.self_attn.qkv_proj.weight"
                                            # 清理，避免重复
                                            del fuse_buckets[prefix]
                                            return (fused_key, qkv)
                                        except Exception:
                                            return None
                                    return None

                                for k, t in _iter_weights_from_dir(model_dir, key_mapper=_map_key):
                                    # 仅在 Qwen/Llama 路径上尝试融合
                                    if (".self_attn.q_proj.weight" in k) or (".self_attn.k_proj.weight" in k) or (".self_attn.v_proj.weight" in k):
                                        try:
                                            base, tail = k.split(".self_attn.", 1)
                                            prefix = base
                                            if tail.startswith("q_proj.weight"):
                                                fuse_buckets.setdefault(prefix, {})["q"] = t
                                            elif tail.startswith("k_proj.weight"):
                                                fuse_buckets.setdefault(prefix, {})["k"] = t
                                            elif tail.startswith("v_proj.weight"):
                                                fuse_buckets.setdefault(prefix, {})["v"] = t
                                            fused = maybe_emit_fused(prefix)
                                            if fused is not None:
                                                yield fused[0], fused[1]
                                            continue
                                        except Exception:
                                            pass
                                    # 非 q/k/v 权重直接透传
                                    yield k, t

                                # 结束时，若仍有未触发的融合且三者齐备，补充发射
                                for prefix, bucket in list(fuse_buckets.items()):
                                    if all(x in bucket for x in ("q", "k", "v")):
                                        try:
                                            qkv = torch.cat([bucket["q"], bucket["k"], bucket["v"]], dim=0)
                                            yield f"{prefix}.self_attn.qkv_proj.weight", qkv
                                        except Exception:
                                            continue

                            internal_model.load_weights(_iter_with_qkv_fusion())
                        except Exception as inner_e:
                            raise inner_e
                        try:
                            if hasattr(engine, "reset_prefix_cache"):
                                engine.reset_prefix_cache()
                        except Exception:
                            pass
                        resp_queue.put({"status": "ok"})
                        continue
                    # 找不到内部模型：直接返回错误（严格要求使用 load_weights）
                    resp_queue.put({
                        "status": "error",
                        "error": "Unable to locate internal vLLM model for hot updating."
                    })
                    continue
                except Exception as e:
                    resp_queue.put({
                        "status": "error",
                        "error": f"load_weights failed: {e!r}"
                    })
                    continue
            elif op == "update_weights":
                # 逐参数热更新：接受 [(name, tensor), ...] 并一次性加载
                try:
                    weights: List[tuple] = cmd.get("weights", [])
                    internal_model = _find_internal_model(engine)
                    if internal_model is None:
                        resp_queue.put({"status": "error", "error": "internal model not found for update_weights"})
                        continue

                    # 基于内部 state_dict 动态构建键映射（与 load_weights 相同逻辑）
                    try:
                        sd_keys = list(getattr(internal_model, "state_dict")().keys())  # type: ignore
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

                    # 类族+版本下强制顶层
                    try:
                        cls_name = type(internal_model).__name__
                    except Exception:
                        cls_name = str(internal_model)
                    try:
                        import vllm as _vllm_mod2  # type: ignore
                        from packaging import version as _v2
                        _ver2 = _v2.parse(getattr(_vllm_mod2, "__version__", "0.0.0"))
                    except Exception:
                        _ver2 = None
                    if _ver2 is not None:
                        force_top = ("Qwen" in cls_name or "Llama" in cls_name) and (_ver2 < _v2.parse("0.8.5"))
                    else:
                        force_top = ("Qwen" in cls_name or "Llama" in cls_name)
                    if force_top:
                        if layers_under_model and not layers_top:
                            layers_top = True
                            layers_under_model = False
                        if embed_under_model and not embed_top:
                            embed_top = True
                            embed_under_model = False
                        if norm_under_model and not norm_top:
                            norm_top = True
                            norm_under_model = False

                    def _map_key(k: str) -> str:
                        if k.startswith("transformer."):
                            k = "model." + k[len("transformer."):]
                        if layers_top:
                            if k.startswith("model.layers."):
                                k = k[len("model."):]
                        elif layers_under_model:
                            if k.startswith("layers."):
                                k = "model." + k
                        if embed_top:
                            if k.startswith("model.embed_tokens."):
                                k = k[len("model.") :]
                        elif embed_under_model:
                            if k.startswith("embed_tokens."):
                                k = "model." + k
                        if norm_top:
                            if k.startswith("model.norm."):
                                k = k[len("model.") :]
                        elif norm_under_model:
                            if k.startswith("norm."):
                                k = "model." + k
                        return k

                    sd_set = set(sd_keys)

                    # QKV 融合：收集三权重并输出 qkv_proj，仅在需要时生效
                    fuse_buckets: Dict[str, Dict[str, torch.Tensor]] = {}

                    def _iter_from_list(pairs: List[tuple[str, torch.Tensor]]):
                        for n, t in pairs:
                            try:
                                mapped = _map_key(str(n))
                            except Exception:
                                mapped = str(n)
                            # 若是 q/k/v，先缓存，待三者齐备时输出融合键
                            if (
                                mapped.endswith(".self_attn.q_proj.weight") or
                                mapped.endswith(".self_attn.k_proj.weight") or
                                mapped.endswith(".self_attn.v_proj.weight")
                            ):
                                try:
                                    base, tail = mapped.split(".self_attn.", 1)
                                    prefix = base
                                    if tail.startswith("q_proj.weight"):
                                        fuse_buckets.setdefault(prefix, {})["q"] = t
                                    elif tail.startswith("k_proj.weight"):
                                        fuse_buckets.setdefault(prefix, {})["k"] = t
                                    elif tail.startswith("v_proj.weight"):
                                        fuse_buckets.setdefault(prefix, {})["v"] = t
                                    bucket = fuse_buckets.get(prefix)
                                    if bucket and all(x in bucket for x in ("q", "k", "v")):
                                        try:
                                            qkv = torch.cat([bucket["q"], bucket["k"], bucket["v"]], dim=0)
                                            fused_key = f"{prefix}.self_attn.qkv_proj.weight"
                                            if fused_key in sd_set:
                                                yield fused_key, qkv.detach().cpu().contiguous()
                                            del fuse_buckets[prefix]
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                continue
                            tensor = t
                            if not isinstance(tensor, torch.Tensor):
                                # best-effort convert
                                try:
                                    tensor = torch.as_tensor(tensor)
                                except Exception:
                                    continue
                            # 仅提交内部存在的键，避免 KeyError
                            if mapped in sd_set:
                                yield mapped, tensor.detach().cpu().contiguous()

                    # 加载
                    internal_model.load_weights(_iter_from_list(weights))
                    try:
                        if hasattr(engine, "reset_prefix_cache"):
                            engine.reset_prefix_cache()
                    except Exception:
                        pass
                    resp_queue.put({"status": "ok"})
                except Exception as e:
                    resp_queue.put({"status": "error", "error": f"update_weights failed: {e!r}"})
                continue
            elif op == "set_lora":
                state["lora_name"] = cmd.get("name")
                state["lora_path"] = cmd.get("path")
                load_into_engine = cmd.get("load_into_engine", False)
                if load_into_engine and hasattr(engine, "load_lora_adapter") and state["lora_name"] and state["lora_path"]:
                    try:
                        engine.load_lora_adapter(state["lora_name"], state["lora_path"])
                    except Exception:
                        pass
                resp_queue.put({"status": "ok"})
            elif op == "shutdown":
                resp_queue.put({"status": "ok"})
                break
            else:
                resp_queue.put({"status": "error", "error": f"Unknown worker op: {op}"})
    except Exception as exc:  # pragma: no cover
        trace = traceback.format_exc()
        print(f"[vLLM/worker] fatal error: {exc}\n{trace}", flush=True)
        resp_queue.put({"status": "error", "error": repr(exc), "trace": trace})
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class VLLMWorkerProxy:
    """Proxy that communicates with a spawn-based vLLM worker process."""

    def __init__(
        self,
        *,
        model_path: str,
        dtype: str,
        visible_cuda: Optional[str],
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        enable_lora: bool = True,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
        startup_timeout_s: float = 180.0,
    ) -> None:
        ctx = mp.get_context("spawn")
        self._cmd: mp.Queue = ctx.Queue()
        self._resp: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(
                self._cmd,
                self._resp,
                {
                    "model_path": model_path,
                    "dtype": dtype,
                    "visible_cuda": visible_cuda,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "max_model_len": max_model_len,
                    "enable_lora": enable_lora,
                    "enforce_eager": enforce_eager,
                    "trust_remote_code": trust_remote_code,
                },
            ),
            daemon=True,
        )
        self._proc.start()
        ready = self._resp.get(timeout=startup_timeout_s)
        if ready.get("status") != "ready":
            self.shutdown(force=True)
            raise RuntimeError(f"vLLM worker failed to start: {ready}")
        self._visible_cuda = ready.get("device")

    @property
    def visible_cuda(self) -> Optional[str]:
        return self._visible_cuda

    def _request(self, payload: Dict[str, Any], timeout: float = 1800.0) -> Dict[str, Any]:
        if not self._proc.is_alive():
            raise RuntimeError("vLLM worker process is not alive.")
        self._cmd.put(payload)
        reply = self._resp.get(timeout=timeout)
        if reply.get("status") == "error":
            raise RuntimeError(reply.get("error", "vLLM worker error"), reply)
        return reply

    def generate(
        self,
        *,
        conversations: List[str],
        sampling: Dict[str, Any],
        use_lora: bool,
        timeout: float = 1800.0,
    ) -> List[Dict[str, Any]]:
        reply = self._request(
            {"op": "generate", "conversations": conversations, "sampling": sampling, "use_lora": use_lora},
            timeout=timeout,
        )
        return reply["result"]  # type: ignore[return-value]

    def load_weights(self, model_dir: str, timeout: float = 1800.0) -> None:
        self._request({"op": "load_weights", "model_dir": model_dir}, timeout=timeout)

    def set_lora_adapter(self, name: Optional[str], path: Optional[str], load_into_engine: bool = False) -> None:
        self._request({"op": "set_lora", "name": name, "path": path, "load_into_engine": bool(load_into_engine)})

    def shutdown(self, force: bool = False, timeout: float = 30.0) -> None:
        if not self._proc.is_alive():
            return
        try:
            if not force:
                self._request({"op": "shutdown"}, timeout=timeout)
        except Exception:
            pass
        finally:
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.kill()
