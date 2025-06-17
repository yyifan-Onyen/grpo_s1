import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import ipdb
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List

from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer as Qwen2Transformer
from llama_model import Transformer as LlamaTransformer
from ministral_model import Transformer as MinistralTransformer
from tokenizer import Tokenizer
from data_types import Episode


def convert_episode_to_all_tokenizers(episode, tokenizers):
    """convert one episode to all tokenizers"""
    episodes = []

    prefix_text = episode.prefix  
    generated_text = episode.text[len(prefix_text):]  
    
    for tokenizer in tokenizers:
        prefix_tokens = tokenizer.tokenize(prefix_text)
        prefix_token_ids = prefix_tokens.ids
        generated_tokens = tokenizer.tokenize(generated_text)
        generated_token_ids = generated_tokens.ids
        
        new_episode = Episode(
            prefix=prefix_text,
            text=episode.text,  
            prefix_token_ids=prefix_token_ids,
            prefix_tokens=prefix_tokens.tokens,
            generated_token_ids=generated_token_ids,
            is_finished=episode.is_finished,
            reward=episode.reward, 
            reward_info=episode.reward_info,
        )
        episodes.append(new_episode)
    
    return episodes


def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get model paths from config (now it's a list)
    pretrained_model_paths = [Path(path) for path in config["model"]["pretrained_model_path"]]
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    
    # Create three tokenizers
    tokenizer_1 = Tokenizer(str(pretrained_model_paths[0] / "tokenizer.json"))
    tokenizer_2 = Tokenizer(str(pretrained_model_paths[1] / "tokenizer.json"))
    tokenizer_3 = Tokenizer(str(pretrained_model_paths[2] / "tokenizer.json"))

    # Create three train datasets
    train_dataset_1 = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer_1,
        split="train",
        test_size=config["data"]["test_size"],
    )
    train_dataset_2 = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer_2,
        split="train",
        test_size=config["data"]["test_size"],
    )
    train_dataset_3 = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer_3,
        split="train",
        test_size=config["data"]["test_size"],
    )
    
    generator = torch.Generator(device=device)
    train_dataloader_1 = DataLoader(
        train_dataset_1,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )
    train_dataloader_2 = DataLoader(
        train_dataset_2,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )
    train_dataloader_3 = DataLoader(
        train_dataset_3,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    # Create three models
    model_1 = Qwen2Transformer.from_pretrained(pretrained_model_paths[0], device=device).train()
    model_2 = MinistralTransformer.from_pretrained(pretrained_model_paths[1], device=device).train()
    model_3 = LlamaTransformer.from_pretrained(pretrained_model_paths[2], device=device).train()

    # Create three optimizers
    optimizer_1 = MemoryEfficientAdamW(
        model_1.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )
    optimizer_2 = MemoryEfficientAdamW(
        model_2.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )
    optimizer_3 = MemoryEfficientAdamW(
        model_3.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(model_1)
    print(model_2)
    print(model_3)

    # Create lists for easier iteration
    models = [model_1, model_2, model_3]
    optimizers = [optimizer_1, optimizer_2, optimizer_3]
    tokenizers = [tokenizer_1, tokenizer_2, tokenizer_3]
    model_names = ["Qwen2", "Ministral", "Llama"]
    
    print("Models loaded successfully:")
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"Model {i+1} ({name}): {sum(p.numel() for p in model.parameters())} parameters")

    # Shared Experience Collective Learning Loop
    for step in range(1, 10000):  # Set a max step limit
        # Get batches from all three dataloaders (each with their own tokenizer's encoding)
        try:
            batch_1 = next(iter(train_dataloader_1))  # Qwen2 tokenizer
            batch_2 = next(iter(train_dataloader_2))  # Ministral tokenizer  
            batch_3 = next(iter(train_dataloader_3))  # Llama tokenizer
            batches = [batch_1, batch_2, batch_3]
        except StopIteration:
            print("Reached end of dataset")
            break
            
        # All three models generate answers for their respective tokenized questions
        print(f"Step {step}: Generating answers for {len(batch_1.prefix)} questions...")
        original_episodes = []  # Store original episodes from each model
        individual_episodes = []
        
        for i, (model, tokenizer, batch, name) in enumerate(zip(models, tokenizers, batches, model_names)):
            # Each model samples answers using its own tokenizer's batch
            episodes = rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,  # Each model uses its own tokenized batch
                max_gen_len=config["training"]["max_gen_len"],
                num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
            )
            
            if config["training"]["skip_unfinished_episodes"]:
                episodes = [episode for episode in episodes if episode.is_finished]
            
            individual_episodes.append(episodes)
            original_episodes.extend(episodes)  # Collect all original episodes
            print(f"  {name}: Generated {len(episodes)} episodes")
            rewards = [ep.reward for ep in episodes]
            format_rewards = [ep.reward_info["format_reward"] for ep in episodes]
            answer_rewards = [ep.reward_info["answer_reward"] for ep in episodes] 
            if rewards:
                print(f"  {name} Reward Analysis:")
                print(f"    Total reward: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
                print(f"    Range: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")
                print(f"    Format reward: mean={np.mean(format_rewards):.3f}, count_1.0={np.sum(np.array(format_rewards)==1.0)}")
                print(f"    Answer reward: mean={np.mean(answer_rewards):.3f}, count_1.0={np.sum(np.array(answer_rewards)==1.0)}")
                # 检查异常值
                extreme_rewards = [r for r in rewards if abs(r) > 2.0]
                if extreme_rewards:
                    print(f"    EXTREME rewards (>2.0): {len(extreme_rewards)} values: {extreme_rewards[:5]}...")
                unique_rewards, counts = np.unique(rewards, return_counts=True)
                if len(unique_rewards) <= 10:
                    print(f"    Reward distribution: {dict(zip(unique_rewards, counts))}")
                else:
                    print(f"    Percentiles: [0%={np.percentile(rewards, 0):.3f}, 25%={np.percentile(rewards, 25):.3f}, 50%={np.percentile(rewards, 50):.3f}, 75%={np.percentile(rewards, 75):.3f}, 100%={np.percentile(rewards, 100):.3f}]")

        print(f"  Total original episodes: {len(original_episodes)}")

        # Convert all episodes to all tokenizer versions for shared learning
        print("Converting episodes to all tokenizer versions...")
        all_episodes = []
        for episode in original_episodes:
            # Convert each episode to 3 tokenizer versions
            converted_episodes = convert_episode_to_all_tokenizers(episode, tokenizers)
            all_episodes.extend(converted_episodes)
        
        print(f"  Total shared episodes (3x converted): {len(all_episodes)}")

        # Here the rollout part is done. then start the update_policy part.
        
        # Each model learns from ALL episodes (shared experience) - now properly tokenized for each model
        all_results = []
        for i, (model, optimizer, tokenizer, name) in enumerate(zip(models, optimizers, tokenizers, model_names)):
            print(f"  {name}: Learning from {len(all_episodes)} shared episodes...")
            
            # Filter episodes that match this model's tokenizer (every 3rd episode starting from index i)
            model_episodes = [all_episodes[j] for j in range(i, len(all_episodes), 3)]
            print(f"  {name}: Using {len(model_episodes)} episodes tokenized for this model...")
            
            results = update_policy(
                model=model,
                optimizer=optimizer,
                episodes=model_episodes,  # Use episodes tokenized for this specific model
                micro_batch_size=config["training"]["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=config["training"]["max_grad_norm"],
                device=device,
                dtype=dtype,
            )
            all_results.append(results)
            print(f"  {name}: Updated with loss={results['loss']:.4f}, grad_norm={results['grad_norm']:.3f}")
            
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # Compute and log metrics for individual models and shared experience
        for i, (episodes, results, name) in enumerate(zip(individual_episodes, all_results, model_names)):
            # Individual model metrics (from their own generated episodes)
            reward = [episode.reward for episode in episodes]
            formatted_reward = [episode.reward_info["format_reward"] for episode in episodes]
            answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
            num_finished_episodes = sum(episode.is_finished for episode in episodes)
            mean_reward = np.mean(reward) if reward else 0.0
            std_reward = np.std(reward) if reward else 0.0
            success_rate = np.mean(answer_reward) if answer_reward else 0.0
            format_reward_mean = np.mean(formatted_reward) if formatted_reward else 0.0
            mean_response_len = np.mean([len(episode.generated_token_ids) for episode in episodes]) if episodes else 0.0
            
            # Shared experience metrics (what the model learned from)
            grad_norm = results["grad_norm"]
            entropy = results["entropy"]
            lr = optimizers[i].param_groups[0]["lr"]
            loss = results["loss"]
            
            print(f"Step {step} | {name} | Own: reward={mean_reward:.2f}, success={success_rate:.2f} | "
                  f"Shared: loss={loss:.4f}, grad_norm={grad_norm:.3f}")
            
            # Log individual model performance
            tb_writer.add_scalar(f"{name}/individual_mean_reward", mean_reward, step)
            tb_writer.add_scalar(f"{name}/individual_success_rate", success_rate, step)
            tb_writer.add_scalar(f"{name}/individual_format_reward", format_reward_mean, step)
            tb_writer.add_scalar(f"{name}/individual_std_reward", std_reward, step)
            tb_writer.add_scalar(f"{name}/individual_mean_response_len", mean_response_len, step)
            tb_writer.add_scalar(f"{name}/individual_num_finished_episodes", num_finished_episodes, step)
            
            # Log shared learning metrics
            tb_writer.add_scalar(f"{name}/shared_loss", loss, step)
            tb_writer.add_scalar(f"{name}/shared_grad_norm", grad_norm, step)
            tb_writer.add_scalar(f"{name}/shared_entropy", entropy, step)
            tb_writer.add_scalar(f"{name}/learning_rate", lr, step)

        # Overall shared experience metrics (based on original episodes before conversion)
        shared_reward = [episode.reward for episode in original_episodes]
        shared_answer_reward = [episode.reward_info["answer_reward"] for episode in original_episodes]
        shared_mean_reward = np.mean(shared_reward) if shared_reward else 0.0
        shared_success_rate = np.mean(shared_answer_reward) if shared_answer_reward else 0.0
        
        tb_writer.add_scalar("shared/original_episodes", len(original_episodes), step)
        tb_writer.add_scalar("shared/converted_episodes", len(all_episodes), step)
        tb_writer.add_scalar("shared/mean_reward", shared_mean_reward, step)
        tb_writer.add_scalar("shared/success_rate", shared_success_rate, step)
        tb_writer.add_scalar("duration", duration, step)
        
        # Evaluation for all models
        if step % config["training"]["eval_interval"] == 0:
            for i, (model, tokenizer, name) in enumerate(zip(models, tokenizers, model_names)):
                eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
                print(f"Eval success rate for {name}: {eval_success_rate:.2f}")
                tb_writer.add_scalar(f"{name}/eval_success_rate", eval_success_rate, step)

        # Save checkpoints for all models
        if step % config["training"]["ckpt_save_interval"] == 0:
            for i, (model, name) in enumerate(zip(models, model_names)):
                output_file = ckpt_dir / f"{name}_ckpt_{step:06d}.pt"
                torch.save(model.state_dict(), output_file)
                print(f"Saved {name} checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/grpo_3.yaml")
    args = parser.parse_args()
    main(args.config)
