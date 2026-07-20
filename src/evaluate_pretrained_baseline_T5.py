import csv # Imports csv to save results
import json # Import json to save metrics and sample outputs
from pathlib import Path # Imports Path to safely create folders and file paths
import numpy as np # Imports numpy for perplexity and JSON-safe metric conversion
import torch # Imports torch
from datasets import DatasetDict # Imports DatasetDict to create a smaller dataset for testing

# Imports the Hugging Face model, tokenizer, trainer, and data collator
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed,)

# Imports the helper functions from the existing T5 training pipeline
from train_t5 import (load_config, prepare_dataset, tokenize_dataset, build_prediction_rows, compute_final_metrics, compute_metrics_builder,)

BASELINE_TEST_SAMPLES = 500 # Number of test examples used for the pretrained baseline

# Pretrained models to be evaluated without fine-tuning on the email dataset
MODEL_CANDIDATES = {
    "pretrained_t5_small": {
        "checkpoint": "google-t5/t5-small",
        "task_prefix": "generate subject: ",
    },
    "pretrained_flan_t5_small": {
        "checkpoint": "google/flan-t5-small",
        "task_prefix": "summarize this email as a subject line: ",
    },
}

# Creates a small dataset for fast baseline evaluation
def make_small_dataset(raw_dataset):
    # Selects a small number of test examples
    small_test = raw_dataset["test"].select(
        range(min(BASELINE_TEST_SAMPLES, len(raw_dataset["test"])))
    )

    # Keeps tiny train and validation splits for the existing tokenizer helper
    small_train = raw_dataset["train"].select(range(1))
    small_validation = raw_dataset["validation"].select(range(1))

    # Returns a smaller DatasetDict.
    return DatasetDict(
        {
            "train": small_train,
            "validation": small_validation,
            "test": small_test,
        }
    )

# Applies a specific task prefix to the formatted input text
def apply_task_prefix(dataset, task_prefix):
    # Rebuilds the input text using the cleaned email body
    def update_example(example):
        return {
            "input_text": task_prefix + example["raw_email"],
        }

    # Applies the new prompt to all splits
    return dataset.map(update_example)

# Converts NumPy values into normal Python values
def make_json_safe(metrics):
    # Creates a cleaned dictionary.
    cleaned = {}

    # Loops through all metric values
    for key, value in metrics.items():
        # Converts NaN values into None
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            cleaned[key] = None

        # Converts NumPy floating values into normal Python floats
        elif isinstance(value, np.floating):
            cleaned[key] = round(float(value), 4)

        # Converts NumPy integers into normal Python integers
        elif isinstance(value, np.integer):
            cleaned[key] = int(value)

        # Keeps normal values unchanged.
        else:
            cleaned[key] = value

    # Returns cleaned metric dictionary.
    return cleaned

# Runs one pretrained baseline evaluation
def evaluate_pretrained_model(model_name, model_info, base_config, small_dataset):
    # Prints progress
    print(f"\nEvaluating the pretrained baseline for: {model_name}")

    # Copies the base config to safely modify it
    config = dict(base_config)

    # Gets the model checkpoint
    model_checkpoint = model_info["checkpoint"]

    # Gets the task prefix for this model
    task_prefix = model_info["task_prefix"]

    # Stores the checkpoint in the config
    config["model_checkpoint"] = model_checkpoint

    # Stores the task prefix in the config
    config["task_prefix"] = task_prefix

    # Keeps BERTScore setting from the config
    config["compute_bertscore"] = base_config.get("compute_bertscore", True)

    # Creates a temporary output folder for Trainer files
    checkpoint_output_dir = Path(
        f"experiments/pretrained_baseline_T5/checkpoints/{model_name}"
    )

    # Makes the folder if it does not exist
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

    # Sets seed for reproducibility
    set_seed(config["seed"])

    # Applies the correct prompt format for this model
    model_dataset = apply_task_prefix(small_dataset, task_prefix)

    # Loads tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Loads pretrained model without fine-tuning
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Tokenizes the dataset
    tokenized_dataset = tokenize_dataset(model_dataset, tokenizer, config)

    # Creates data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
        model=model,)

    # Defines prediction-only settings
    prediction_args = Seq2SeqTrainingArguments(output_dir=str(checkpoint_output_dir),
        per_device_eval_batch_size = config["eval_batch_size"],
        predict_with_generate = True,
        generation_max_length = config["max_target_length"],
        generation_num_beams = config["num_beams"],
        fp16 = False,
        report_to = config["report_to"],)

    # Creates Trainer for prediction only
    trainer = Seq2SeqTrainer(
        model = model,
        args = prediction_args,
        eval_dataset = tokenized_dataset["test"],
        processing_class = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics_builder(tokenizer),)

    # Generates predictions on the test subset
    prediction_output = trainer.predict(tokenized_dataset["test"])

    # Builds readable prediction rows
    prediction_rows = build_prediction_rows(
        prediction_output,
        tokenizer,
        model_dataset,)

    # Computes ROUGE, BLEU, BERTScore, and length metrics
    metrics = compute_final_metrics(prediction_rows, config)

    # Gets test loss from Trainer output
    test_loss = prediction_output.metrics.get("test_loss")

    # Calculates perplexity from test loss if available
    if test_loss is not None:
        test_loss_value = float(test_loss)
        metrics["test_loss"] = round(test_loss_value, 4)
        metrics["perplexity"] = round(float(np.exp(test_loss_value)), 4)

    # Adds model details
    metrics["model_name"] = model_name
    metrics["model_checkpoint"] = model_checkpoint
    metrics["task_prefix"] = task_prefix
    metrics["test_examples"] = len(model_dataset["test"])

    # Cleans metrics for JSON saving
    metrics = make_json_safe(metrics)

    # Returns metrics and prediction rows
    return metrics, prediction_rows

# Saves all pretrained baseline results
def save_results(all_metrics, all_samples):
    # Creates output folder
    output_dir = Path("experiments/pretrained_baseline_T5")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Saves metrics as JSON
    json_path = output_dir / "pretrained_baseline_metrics.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2, ensure_ascii=False)

    # Saves metrics as CSV
    csv_path = output_dir / "pretrained_baseline_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        fieldnames = list(all_metrics[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    # Saves sample predictions as JSON
    samples_path = output_dir / "pretrained_baseline_samples.json"
    with open(samples_path, "w", encoding="utf-8") as file:
        json.dump(all_samples, file, indent=2, ensure_ascii=False)

    # Prints saved file paths
    print(f"\nSaved metrics JSON to {json_path}")
    print(f"Saved metrics CSV to {csv_path}")
    print(f"Saved sample predictions to {samples_path}")


# Main function
def main():
    # Loads the existing T5 config as the base config
    base_config = load_config("configs/t5_config.yaml")

    # Prints whether GPU is available
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Loads, cleans, filters, and formats the dataset
    raw_dataset = prepare_dataset(base_config)

    # Creates a small dataset for quick baseline testing
    small_dataset = make_small_dataset(raw_dataset)

    # Stores all model metrics
    all_metrics = []

    # Stores sample predictions
    all_samples = {}

    # Loops through each pretrained model
    for model_name, model_info in MODEL_CANDIDATES.items():
        # Runs one pretrained baseline
        metrics, prediction_rows = evaluate_pretrained_model(
            model_name,
            model_info,
            base_config,
            small_dataset,)

        # Saves the metrics
        all_metrics.append(metrics)

        # Saves only the first 10 sample predictions
        all_samples[model_name] = prediction_rows[:10]

    # Saves results
    save_results(all_metrics, all_samples)

    # Prints final results
    print("\nPretrained baseline results:")
    for metrics in all_metrics:
        print(metrics)


# Runs main only when this file is executed directly
if __name__ == "__main__":
    main()
