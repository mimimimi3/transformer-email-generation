"""Fine-tune DistilGPT-2 for email continuation.

Creates prompt / continuation pairs from the AESLC-style dataset, fine-tunes under
two prompt conditions (body-only vs subject + body), compares the untouched
pretrained checkpoint to the fine-tuned model, and saves metrics plus samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import evaluate
import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Supported prompt conditions for the research question comparing body-only vs
# subject + body context for continuation quality.
CONDITIONS = ("body_only", "subject_and_body")

# Explicit markers make it easier to separate prompt text from model generations.
CONTINUATION_MARKER = "\nContinuation:"
EMAIL_MARKER = "Email:"
SUBJECT_MARKER = "Subject:"


def load_config(config_path):
    """Loads project settings from the YAML config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    """CLI overrides stay thin; hyperparameters live in the YAML config."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilGPT-2 for email continuation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/distilgpt2_config.yaml",
        help="Path to the DistilGPT-2 YAML config file.",
    )
    parser.add_argument(
        "--condition",
        choices=["both", *CONDITIONS],
        default=None,
        help="Optional override for which prompt condition(s) to run.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional override that caps training examples (smoke tests).",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional override that caps validation/test examples.",
    )
    parser.add_argument(
        "--skip_pretrained_eval",
        action="store_true",
        help="Optional override to skip pretrained baseline evaluation.",
    )
    return parser.parse_args()


def apply_cli_overrides(config, args):
    """Applies only the CLI values that were explicitly provided."""
    if args.condition is not None:
        config["condition"] = args.condition
    if args.max_train_samples is not None:
        config["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        config["max_eval_samples"] = args.max_eval_samples
    if args.skip_pretrained_eval:
        config["skip_pretrained_eval"] = True
    return config


def clean_text(text):
    """Normalizes email / subject text before splitting and prompting."""
    if text is None:
        return ""

    text = str(text)
    # Remove AESLC-style forwarded markers and attachment filename lines.
    text = re.sub(r"<<.*?>>", "", text)
    text = re.sub(
        r"[^\n]*\.(doc|docx|xls|xlsx|pdf|ppt|pptx|csv|zip|jpg|jpeg|gif)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_email_body(email_body):
    """Split a cleaned email into an incomplete draft and a held-out continuation.

    The split is deterministic (character midpoint) so train / validation / test
    examples are reproducible. Prefer a nearby whitespace boundary so we do not
    cut through the middle of a word when possible.
    """
    midpoint = len(email_body) // 2
    if midpoint <= 0 or midpoint >= len(email_body):
        return email_body, ""

    # Search a small window around the midpoint for a cleaner split point.
    window_start = max(0, midpoint - 40)
    window_end = min(len(email_body), midpoint + 40)
    window = email_body[window_start:window_end]

    whitespace_offsets = [index for index, char in enumerate(window) if char.isspace()]
    if whitespace_offsets:
        best_local = min(
            whitespace_offsets,
            key=lambda index: abs((window_start + index) - midpoint),
        )
        # +1 keeps the whitespace with the prompt side of the split.
        split_at = window_start + best_local + 1
    else:
        split_at = midpoint

    prompt = email_body[:split_at].strip()
    continuation = email_body[split_at:].strip()
    return prompt, continuation


def build_prompt_text(condition, subject, email_prompt):
    """Formats the incomplete email into the condition-specific model prompt."""
    if condition == "body_only":
        return f"{EMAIL_MARKER}\n{email_prompt}{CONTINUATION_MARKER}"

    if condition == "subject_and_body":
        # Subject context is the only difference between the two experiment arms.
        return f"{SUBJECT_MARKER}\n{subject}\n{EMAIL_MARKER}\n{email_prompt}{CONTINUATION_MARKER}"

    raise ValueError(f"Unknown condition: {condition}")


def prepare_continuation_dataset(config, condition):
    """Loads, cleans, filters, and formats prompt / continuation pairs."""
    print(f"Loading dataset for condition '{condition}'...")
    dataset = load_dataset(config["dataset_name"])

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset.keys())
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing_splits)}")

    body_column = config["body_column"]
    subject_column = config["subject_column"]
    required_columns = {body_column, subject_column}

    for split_name in ("train", "validation", "test"):
        missing_columns = required_columns - set(dataset[split_name].column_names)
        if missing_columns:
            raise ValueError(f"The {split_name} split is missing required columns: {sorted(missing_columns)}")

    def format_example(example):
        email_body = clean_text(example[body_column])
        subject = clean_text(example[subject_column])
        email_prompt, continuation = split_email_body(email_body)
        prompt_text = build_prompt_text(condition, subject, email_prompt)

        return {
            "subject": subject,
            "raw_email": email_body,
            "email_prompt": email_prompt,
            "continuation": continuation,
            "prompt_text": prompt_text,
            # full_text is what the causal LM sees during teacher-forced training.
            "full_text": prompt_text + " " + continuation,
        }

    dataset = dataset.map(format_example)

    def keep_example(example):
        return (
            len(example["raw_email"]) >= config["min_email_chars"]
            and len(example["subject"]) >= config["min_subject_chars"]
            and len(example["email_prompt"]) >= config["min_prompt_chars"]
            and len(example["continuation"]) >= config["min_continuation_chars"]
        )

    dataset = dataset.filter(keep_example)

    for split_name in ("train", "validation", "test"):
        if len(dataset[split_name]) == 0:
            raise ValueError(f"The {split_name} split is empty after filtering for {condition}.")

    # Optional caps support smoke tests without changing the YAML defaults.
    if config.get("max_train_samples") is not None:
        dataset["train"] = dataset["train"].select(range(min(config["max_train_samples"], len(dataset["train"]))))

    if config.get("max_eval_samples") is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(config["max_eval_samples"], len(dataset["validation"])))
        )
        dataset["test"] = dataset["test"].select(range(min(config["max_eval_samples"], len(dataset["test"]))))

    preview_dir = Path("data/processed")
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"distilgpt2_{condition}_preview.jsonl"
    with open(preview_path, "w", encoding="utf-8") as file:
        for example in dataset["train"].select(range(min(5, len(dataset["train"])))):
            line = json.dumps(
                {
                    "prompt_text": example["prompt_text"],
                    "continuation": example["continuation"],
                    "full_text": example["full_text"],
                },
                ensure_ascii=False,
            )
            file.write(line + "\n")

    print(
        f"[{condition}] train={len(dataset['train'])}, "
        f"validation={len(dataset['validation'])}, test={len(dataset['test'])}"
    )
    print(f"Saved preview examples to {preview_path}")
    return dataset


def load_tokenizer_and_model(model_checkpoint):
    """Loads DistilGPT-2 and configures padding for causal LM training."""
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # GPT-2 tokenizers do not ship with a pad token; reuse EOS so batching works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Right padding is the standard setup for causal-LM fine-tuning.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    # Keep model + generation configs in sync with the tokenizer up front so
    # Transformers does not later print an alignment warning for pad_token_id.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.loss_type = "ForCausalLM"
    return tokenizer, model


@dataclass
class CausalLMPromptCollator:
    """Pads input ids and already-masked labels for causal LM training.

    The default language-modeling collator rebuilds labels from input ids, which
    would undo prompt masking. This collator keeps our custom labels and pads
    them with -100 so padded positions are ignored by the loss.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padding_length = max_length - len(label)
            if padding_length < 0:
                raise ValueError("Label sequence is longer than padded input ids.")
            padded_labels.append(list(label) + [-100] * padding_length)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenizes full sequences and masks prompt tokens in the labels."""

    def tokenize_batch(batch):
        # Truncate prompts to the training window too. Encoding them without a
        # limit triggers Transformers warnings when a prompt exceeds the model
        # max length (1024 for GPT-2), even though full_text is truncated below.
        prompt_encodings = tokenizer(
            batch["prompt_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        full_encodings = tokenizer(
            batch["full_text"],
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )

        labels = []
        for input_ids, prompt_ids in zip(
            full_encodings["input_ids"],
            prompt_encodings["input_ids"],
        ):
            # -100 tells PyTorch CrossEntropyLoss to ignore those positions.
            # If truncation cut into the prompt, mask only the surviving prefix.
            prompt_token_count = min(len(prompt_ids), len(input_ids))
            label_ids = list(input_ids)
            for index in range(prompt_token_count):
                label_ids[index] = -100
            labels.append(label_ids)

        full_encodings["labels"] = labels
        return full_encodings

    columns_to_remove = dataset["train"].column_names
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
    )

    # Drop examples where the prompt alone filled the whole window, leaving no
    # continuation tokens for the loss to train on.
    def has_trainable_continuation(example):
        return any(label != -100 for label in example["labels"])

    before_counts = {split: len(tokenized[split]) for split in tokenized}
    tokenized = tokenized.filter(has_trainable_continuation)
    for split_name, before_count in before_counts.items():
        removed = before_count - len(tokenized[split_name])
        if removed:
            print(
                f"Removed {removed} {split_name} example(s) with no "
                "trainable continuation tokens after truncation."
            )

    return tokenized


def safe_perplexity(loss_value):
    """Converts average cross-entropy loss to perplexity with overflow guards."""
    if loss_value is None:
        return None

    loss_value = float(loss_value)
    try:
        perplexity = math.exp(loss_value)
    except OverflowError:
        return float("inf")

    if math.isinf(perplexity) or math.isnan(perplexity):
        return float("inf")

    return round(perplexity, 4)


def extract_generated_continuation(decoded_text, prompt_text):
    """Removes the prompt from decoded generation so metrics score only new text."""
    text = decoded_text.strip()
    prompt = prompt_text.strip()

    if text.startswith(prompt):
        continuation = text[len(prompt) :].strip()
    elif CONTINUATION_MARKER in text:
        continuation = text.split(CONTINUATION_MARKER, maxsplit=1)[1].strip()
    else:
        continuation = text

    # If the model starts inventing a new Email/Subject section, cut it off.
    for marker in (EMAIL_MARKER, SUBJECT_MARKER, CONTINUATION_MARKER.strip()):
        if marker in continuation:
            continuation = continuation.split(marker, maxsplit=1)[0].strip()

    return continuation


def compute_generation_metrics(prediction_rows, compute_bertscore=True):
    """Computes ROUGE, BLEU, and optional BERTScore on generated continuations."""
    generated = [row["generated_continuation"] for row in prediction_rows]
    references = [row["reference_continuation"] for row in prediction_rows]

    metrics = {}

    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(
        predictions=generated,
        references=references,
        use_stemmer=True,
    )
    metrics["rouge1"] = round(rouge_result["rouge1"], 4)
    metrics["rouge2"] = round(rouge_result["rouge2"], 4)
    metrics["rougeL"] = round(rouge_result["rougeL"], 4)

    bleu = evaluate.load("sacrebleu")
    # SacreBLEU expects each reference wrapped in its own list.
    bleu_result = bleu.compute(
        predictions=generated,
        references=[[reference] for reference in references],
    )
    metrics["bleu"] = round(bleu_result["score"], 4)

    if compute_bertscore:
        bertscore_predictions = [text if text.strip() else "[empty]" for text in generated]
        bertscore_references = [text if text.strip() else "[empty]" for text in references]
        empty_count = sum(
            1 for prediction, reference in zip(generated, references) if not prediction.strip() or not reference.strip()
        )
        if empty_count:
            print(f"Warning: replaced {empty_count} empty continuation(s) with '[empty]' before BERTScore.")

        try:
            bertscore = evaluate.load("bertscore")
            bertscore_result = bertscore.compute(
                predictions=bertscore_predictions,
                references=bertscore_references,
                lang="en",
            )
            metrics["bertscore_precision"] = round(float(np.mean(bertscore_result["precision"])), 4)
            metrics["bertscore_recall"] = round(float(np.mean(bertscore_result["recall"])), 4)
            metrics["bertscore_f1"] = round(float(np.mean(bertscore_result["f1"])), 4)
        except Exception as error:
            print(
                "Warning: BERTScore failed and was skipped "
                f"({type(error).__name__}: {error}). "
                "ROUGE/BLEU metrics were still saved."
            )

    generated_lengths = [len(text.split()) for text in generated]
    metrics["average_generated_continuation_words"] = round(
        float(np.mean(generated_lengths)) if generated_lengths else 0.0,
        2,
    )
    return metrics


def generate_continuations(model, tokenizer, raw_dataset, config, max_examples):
    """Greedily generates continuations from prompt-only inputs."""
    model.eval()
    device = next(model.parameters()).device

    examples = raw_dataset["test"].select(range(min(max_examples, len(raw_dataset["test"]))))

    prediction_rows = []
    for index, example in enumerate(examples):
        prompt_text = example["prompt_text"]
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=config["max_length"],
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            # Greedy decoding keeps evaluation deterministic across runs.
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_continuation = extract_generated_continuation(decoded, prompt_text)

        prediction_rows.append(
            {
                "index": index,
                "condition_prompt": prompt_text,
                "email_prompt": example["email_prompt"],
                "subject": example["subject"],
                "reference_continuation": example["continuation"],
                "generated_continuation": generated_continuation,
            }
        )

    return prediction_rows


def build_trainer(model, tokenizer, tokenized_dataset, config, output_dir):
    """Creates the Hugging Face Trainer for DistilGPT-2 fine-tuning."""
    data_collator = CausalLMPromptCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        weight_decay=config["weight_decay"],
        num_train_epochs=config["num_train_epochs"],
        fp16=config["fp16"] and torch.cuda.is_available(),
        # pin_memory only helps when moving batches to a GPU/accelerator.
        dataloader_pin_memory=torch.cuda.is_available(),
        logging_dir=config.get("logging_dir"),
        logging_steps=25,
        report_to=config.get("report_to", "none"),
        save_total_limit=2,
        # Keep the lowest-validation-loss checkpoint for final evaluation.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    return trainer


def evaluate_model_loss(trainer, tokenized_dataset):
    """Reports validation / test loss and perplexity for a loaded checkpoint."""
    validation_metrics = trainer.evaluate(tokenized_dataset["validation"])
    test_metrics = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    validation_loss = validation_metrics.get("eval_loss")
    test_loss = test_metrics.get("test_loss")

    return {
        "validation_loss": (round(float(validation_loss), 4) if validation_loss is not None else None),
        "validation_perplexity": safe_perplexity(validation_loss),
        "test_loss": round(float(test_loss), 4) if test_loss is not None else None,
        "test_perplexity": safe_perplexity(test_loss),
        "raw_validation_metrics": validation_metrics,
        "raw_test_metrics": test_metrics,
    }


def save_condition_outputs(
    condition,
    output_dir,
    pretrained_metrics,
    fine_tuned_metrics,
    prediction_rows,
    config,
):
    """Writes metrics, prediction CSV, and readable sample files for one condition."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if pretrained_metrics is not None:
        pretrained_path = output_dir / "pretrained_metrics.json"
        with open(pretrained_path, "w", encoding="utf-8") as file:
            json.dump(pretrained_metrics, file, indent=2, ensure_ascii=False)
        print(f"Saved pretrained metrics to {pretrained_path}")

    fine_tuned_path = output_dir / "fine_tuned_metrics.json"
    with open(fine_tuned_path, "w", encoding="utf-8") as file:
        json.dump(fine_tuned_metrics, file, indent=2, ensure_ascii=False)
    print(f"Saved fine-tuned metrics to {fine_tuned_path}")

    predictions_csv_path = output_dir / "continuation_predictions.csv"
    fieldnames = [
        "index",
        "subject",
        "email_prompt",
        "condition_prompt",
        "reference_continuation",
        "generated_continuation",
    ]
    with open(predictions_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"Saved predictions to {predictions_csv_path}")

    samples_txt_path = output_dir / "samples_continuation.txt"
    num_samples = min(config["num_output_samples"], len(prediction_rows))
    with open(samples_txt_path, "w", encoding="utf-8") as file:
        for sample_number, row in enumerate(prediction_rows[:num_samples], start=1):
            file.write("=" * 80 + "\n")
            file.write(f"Example {sample_number} ({condition})\n\n")
            file.write("PROMPT:\n")
            file.write(row["condition_prompt"][:1200] + "\n\n")
            file.write("REFERENCE CONTINUATION:\n")
            file.write(row["reference_continuation"][:1200] + "\n\n")
            file.write("GENERATED CONTINUATION:\n")
            file.write(row["generated_continuation"][:1200] + "\n\n")
    print(f"Saved readable samples to {samples_txt_path}")

    samples_json_path = output_dir / "samples_continuation.json"
    with open(samples_json_path, "w", encoding="utf-8") as file:
        json.dump(prediction_rows[:num_samples], file, indent=2, ensure_ascii=False)


def run_condition(condition, config):
    """Runs pretrained evaluation, fine-tuning, and artifact saving for one arm."""
    print("\n" + "=" * 80)
    print(f"Running DistilGPT-2 continuation condition: {condition}")
    print("=" * 80)

    set_seed(config["seed"])

    raw_dataset = prepare_continuation_dataset(config, condition)
    tokenizer, model = load_tokenizer_and_model(config["model_checkpoint"])
    tokenized_dataset = tokenize_dataset(
        raw_dataset,
        tokenizer,
        config["max_length"],
    )

    condition_output_dir = Path(config["results_dir"]) / condition
    condition_model_dir = Path(config["output_dir"]) / condition
    checkpoint_dir = condition_output_dir / "checkpoints"
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    condition_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        config=config,
        output_dir=checkpoint_dir,
    )

    pretrained_metrics = None
    if not config.get("skip_pretrained_eval", False):
        # Baseline before fine-tuning supports the proposed perplexity-reduction goal.
        print("Evaluating the untouched pretrained DistilGPT-2 checkpoint...")
        pretrained_loss_metrics = evaluate_model_loss(trainer, tokenized_dataset)
        pretrained_predictions = generate_continuations(
            model=trainer.model,
            tokenizer=tokenizer,
            raw_dataset=raw_dataset,
            config=config,
            max_examples=config["num_generate_samples"],
        )
        pretrained_generation_metrics = compute_generation_metrics(
            pretrained_predictions,
            compute_bertscore=config.get("compute_bertscore", True),
        )
        pretrained_metrics = {
            "condition": condition,
            "model_stage": "pretrained",
            "model_checkpoint": config["model_checkpoint"],
            **{key: value for key, value in pretrained_loss_metrics.items() if not key.startswith("raw_")},
            **pretrained_generation_metrics,
        }
        print("Pretrained metrics:")
        print(json.dumps(pretrained_metrics, indent=2))

    print("Starting DistilGPT-2 fine-tuning...")
    trainer.train()

    print("Evaluating the fine-tuned DistilGPT-2 checkpoint...")
    fine_tuned_loss_metrics = evaluate_model_loss(trainer, tokenized_dataset)
    fine_tuned_predictions = generate_continuations(
        model=trainer.model,
        tokenizer=tokenizer,
        raw_dataset=raw_dataset,
        config=config,
        max_examples=config["num_generate_samples"],
    )
    fine_tuned_generation_metrics = compute_generation_metrics(
        fine_tuned_predictions,
        compute_bertscore=config.get("compute_bertscore", True),
    )

    fine_tuned_metrics = {
        "condition": condition,
        "model_stage": "fine_tuned",
        "model_checkpoint": config["model_checkpoint"],
        **{key: value for key, value in fine_tuned_loss_metrics.items() if not key.startswith("raw_")},
        **fine_tuned_generation_metrics,
    }

    # Percent reduction answers the proposal target of 15-30% lower perplexity.
    if (
        pretrained_metrics is not None
        and pretrained_metrics.get("test_perplexity") not in (None, float("inf"))
        and fine_tuned_metrics.get("test_perplexity") not in (None, float("inf"))
        and pretrained_metrics["test_perplexity"] > 0
    ):
        reduction = (
            (pretrained_metrics["test_perplexity"] - fine_tuned_metrics["test_perplexity"])
            / pretrained_metrics["test_perplexity"]
        ) * 100.0
        fine_tuned_metrics["test_perplexity_reduction_pct"] = round(reduction, 2)

    print("Fine-tuned metrics:")
    print(json.dumps(fine_tuned_metrics, indent=2))

    print("Saving final model weights...")
    trainer.save_model(str(condition_model_dir))
    tokenizer.save_pretrained(str(condition_model_dir))

    save_condition_outputs(
        condition=condition,
        output_dir=condition_output_dir,
        pretrained_metrics=pretrained_metrics,
        fine_tuned_metrics=fine_tuned_metrics,
        prediction_rows=fine_tuned_predictions,
        config=config,
    )

    return {
        "condition": condition,
        "pretrained": pretrained_metrics,
        "fine_tuned": fine_tuned_metrics,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    set_seed(config["seed"])

    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    if config.get("logging_dir"):
        Path(config["logging_dir"]).mkdir(parents=True, exist_ok=True)

    selected_condition = config.get("condition", "both")
    conditions = list(CONDITIONS) if selected_condition == "both" else [selected_condition]
    all_results = []

    for condition in conditions:
        result = run_condition(condition, config)
        all_results.append(result)

    summary_path = Path(config["results_dir"]) / "condition_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved condition comparison summary to {summary_path}")
    print("DistilGPT-2 continuation pipeline completed.")


if __name__ == "__main__":
    main()
