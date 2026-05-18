import os
import math
import time
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

MODEL_NAME = "roberta-base"

TRAIN_FILE = "data/processed/dapt/all_pretrain_train.txt"
VALID_FILE = "data/processed/dapt/all_pretrain_valid.txt"

BASE_OUTPUT_DIR = "outputs/dapt_lr_search"

MAX_LENGTH = 256
MLM_PROBABILITY = 0.15

PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1

NUM_TRAIN_EPOCHS = 1

# 学习率搜索范围
LR_LIST = [5e-6, 1e-5, 2e-5]

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

LOGGING_STEPS = 100
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3

SEED = 42
USE_FP16 = True
USE_BF16 = False


class SpeedCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = 0
        print("Training started.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step <= 0:
            return

        now = time.time()
        elapsed = now - self.last_log_time
        step_diff = state.global_step - self.last_log_step

        if step_diff > 0:
            sec_per_step = elapsed / step_diff
            print(
                f"[Speed] global_step={state.global_step} | "
                f"sec/step={sec_per_step:.4f}"
            )

        self.last_log_time = now
        self.last_log_step = state.global_step


def format_lr(lr):
    """
    把学习率转成适合文件夹命名的格式。
    例如：
    5e-6 -> 5e-06
    1e-5 -> 1e-05
    """
    return f"{lr:.0e}"


def run_dapt(learning_rate):
    """
    对单个学习率进行一次 RoBERTa 继续预训练。
    """
    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        f"roberta_dapt_lr{format_lr(learning_rate)}"
    )

    os.makedirs(output_dir, exist_ok=True)
    set_seed(SEED)

    print("\n" + "=" * 80)
    print(f"Start DAPT with learning rate: {learning_rate}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # 读取数据
    data_files = {
        "train": TRAIN_FILE,
        "validation": VALID_FILE,
    }
    raw_datasets = load_dataset("text", data_files=data_files)

    print(raw_datasets)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.mask_token is None:
        raise ValueError("当前 tokenizer 没有 mask_token，不能做 MLM。")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # 分词
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # 数据整理器：MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    # 模型
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # 打印大概step数
    train_size = len(tokenized_datasets["train"])
    valid_size = len(tokenized_datasets["validation"])
    effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = math.ceil(train_size / effective_batch_size)
    total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS

    print(f"Train samples: {train_size}")
    print(f"Validation samples: {valid_size}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,

        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",

        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,

        fp16=USE_FP16,
        bf16=USE_BF16,

        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[SpeedCallback()],
    )

    # 训练
    trainer.train()

    # 保存当前学习率下的最终模型
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 评估
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    eval_loss = metrics.get("eval_loss", None)

    if eval_loss is not None:
        try:
            perplexity = math.exp(eval_loss)
            print("Perplexity:", perplexity)
        except OverflowError:
            perplexity = float("inf")
            print("Perplexity: overflow")
    else:
        perplexity = None

    result = {
        "learning_rate": learning_rate,
        "output_dir": output_dir,
        "final_dir": final_dir,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
    }

    # 保存当前学习率的结果
    result_path = os.path.join(output_dir, "dapt_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved result to: {result_path}")
    print(f"Saved final model to: {final_dir}")

    return result

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    all_results = []

    for lr in LR_LIST:
        result = run_dapt(lr)
        all_results.append(result)

    # 汇总所有学习率结果
    summary_path = os.path.join(BASE_OUTPUT_DIR, "dapt_lr_search_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DAPT learning rate search finished.")
    print("=" * 80)

    for r in all_results:
        print(
            f"lr={r['learning_rate']} | "
            f"eval_loss={r['eval_loss']} | "
            f"perplexity={r['perplexity']} | "
            f"final_dir={r['final_dir']}"
        )

    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()