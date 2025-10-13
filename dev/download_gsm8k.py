import os
import json
from typing import Dict

from datasets import load_dataset


INSTRUCTION = (
    "You are a knowledgeable Math assistant. Answer the following question "
    "and think step by step. The final answer should be enclosed within \\boxed{{}}, "
    "e.g. \\boxed{{n}}"
)


def _extract_final_answer(answer_text: str) -> str:
    """GSM8K 的官方答案通常含推理，并在行首或行尾用 '#### x' 给出最终数字。
    这里尽量稳健地抽取 '#### ' 之后的内容，并去除多余空白。"""
    if not isinstance(answer_text, str):
        return str(answer_text)
    marker = "####"
    if marker in answer_text:
        tail = answer_text.split(marker, 1)[1]
        return tail.strip().split("\n")[0].strip()
    # 兜底：取最后一行
    return answer_text.strip().split("\n")[-1].strip()


def _to_record(example: Dict, idx: int, split: str) -> Dict:
    prompt = INSTRUCTION + "\n\n" + example["question"].strip()
    ans = _extract_final_answer(example["answer"])  # 仅保存最终答案
    return {
        "prompt": prompt,
        "answer": ans,
        "source": "GSM8K-main",
        "id": f"{split}/{idx}.json",
    }


def save_gsm8k_to_workspace(out_dir: str = "/workspace/data/GSM8k") -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("openai/gsm8k", name="main")

    for split in ("train", "test"):
        records = []
        for i, ex in enumerate(ds[split]):
            records.append(_to_record(ex, i, split))
        out_path = os.path.join(out_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[GSM8K] Saved {split}: {len(records)} to {out_path}")


if __name__ == "__main__":
    save_gsm8k_to_workspace()