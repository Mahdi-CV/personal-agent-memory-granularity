#!/usr/bin/env python3
"""Train one Qwen3-4B LoRA adapter for the granularity experiments."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--adapter-name", default="tofu")
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--target-modules", default=None)
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.train_jsonl)]
    targets = (
        [item.strip() for item in args.target_modules.split(",") if item.strip()]
        if args.target_modules else list(DEFAULT_TARGETS)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False,
            add_generation_prompt=False)}

    dataset = Dataset.from_list(
        [{"messages": record["messages"]} for record in records])
    dataset = dataset.map(render, remove_columns=dataset.column_names)
    dataset = dataset.map(
        lambda example: tokenizer(
            example["text"], truncation=True, max_length=2048, padding=False),
        remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    ))

    training = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started

    model.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)

    stats = {
        "base": args.base,
        "out_dir": args.out_dir,
        "epochs": args.epochs,
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": targets,
        "n_records": len(records),
        "global_step": int(result.global_step),
        "training_loss": float(result.training_loss),
        "train_seconds": round(elapsed, 1),
    }
    print("TRAIN DONE:", json.dumps(stats), flush=True)
    if args.stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_json)),
                    exist_ok=True)
        with open(args.stats_json, "w") as handle:
            json.dump(stats, handle, indent=2)


if __name__ == "__main__":
    main()
