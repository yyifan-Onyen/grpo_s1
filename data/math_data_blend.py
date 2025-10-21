import json
import os
import random


def main() -> None:
    math_path = "/workspace/data/MATH/train.json"
    gsm_path = "/workspace/data/GSM8k/train.json"
    out_dir = "/workspace/data/MATH_BLEND"
    out_path = os.path.join(out_dir, "train.json")

    os.makedirs(out_dir, exist_ok=True)

    with open(math_path, "r", encoding="utf-8") as f:
        math_data = json.load(f)
    with open(gsm_path, "r", encoding="utf-8") as f:
        gsm_data = json.load(f)

    blended = (math_data or []) + (gsm_data or [])
    random.shuffle(blended)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blended, f, ensure_ascii=False, indent=2)

    print(f"Blended dataset written: {out_path} | size={len(blended)}")


if __name__ == "__main__":
    main()