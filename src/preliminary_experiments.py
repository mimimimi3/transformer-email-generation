import csv # Imports csv for saving model comparison results

import json # Imports json for saving the results in JSON format

from pathlib import Path # Imports Path for creating output folders

import numpy as np # Imports numpy for calculating perplexity from loss

import torch # Imports torch

from datasets import DatasetDict # Imports DatasetDict for creating smaller dataset splits.

# Imports Hugging Face model, tokenizer, trainer, and training utilities
from transformers import (AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Imports helper functions from the final T5 training script
from train_t5 import (load_config,
    prepare_dataset,
    tokenize_dataset,
    compute_metrics_builder,
    build_prediction_rows,
    compute_final_metrics,
)


# Defines small subset sizes for preliminary testing
TRAIN_SAMPLES = 256
VALIDATION_SAMPLES = 64
TEST_SAMPLES = 64

# Defines the number of epochs for preliminary testing
PRELIM_EPOCHS = 3

MODEL_CANDIDATES = {"t5-small": "google-t5/t5-small", # Defines the models we want to compare
    "flan-t5-small": "google/flan-t5-small",
}


def make_small_dataset(raw_dataset): # Creates a smaller version of the dataset for fast testing
    return DatasetDict( # Returns a new DatasetDict with small train, validation, and test splits
        {
            "train": raw_dataset["train"].select(range(min(TRAIN_SAMPLES, len(raw_dataset["train"])))),
            "validation": raw_dataset["validation"].select(range(min(VALIDATION_SAMPLES, len(raw_dataset["validation"])))),
            "test": raw_dataset["test"].select(range(min(TEST_SAMPLES, len(raw_dataset["test"])))),
        }
    )


# Runs one preliminary experiment for one model
def run_experiment(model_name, model_checkpoint, base_config, small_dataset):
 
    print(f"\nRunning preliminary experiment for: {model_name}")

    config = dict(base_config) # Copies the config

    config["model_checkpoint"] = model_checkpoint # Sets the model checkpoint

    config["num_train_epochs"] = PRELIM_EPOCHS

    config["compute_bertscore"] = True # Turns on BERTScore for preliminary tests

    config["output_dir"] = f"outputs/preliminary_experiments/checkpoints/{model_name}" # Sets up a separate output folder

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True) # Creates the model output folder

    set_seed(config["seed"]) # Sets seed for reproducibility

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # Loads the tokenizer for this model

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint) # Loads the pretrained sequence-to-sequence model

    tokenized_dataset = tokenize_dataset(small_dataset, tokenizer, config) # Tokenizes the small dataset

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, # Creates data collator for dynamic padding
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(output_dir=config["output_dir"], # Defines the training settings for testing
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        weight_decay=config["weight_decay"],
        num_train_epochs=config["num_train_epochs"],
        predict_with_generate=True,
        generation_max_length=config["max_target_length"],
        generation_num_beams=config["num_beams"],
        fp16=config["fp16"] and torch.cuda.is_available(),
        logging_steps=10,
        report_to=config["report_to"],
    )

    trainer = Seq2SeqTrainer( # Creates the Trainer
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
    )

    trainer.train() # Trains briefly on the small subset

    # Evaluates and generates predictions on the test subset
    prediction_output = trainer.predict(tokenized_dataset["test"])

    prediction_rows = build_prediction_rows( # Builds readable prediction rows
        prediction_output,
        tokenizer,
        small_dataset,
    )

    # Computes final ROUGE and BLEU metrics
    final_metrics = compute_final_metrics(prediction_rows, config)

    # Gets the test loss from prediction metrics
    test_loss = prediction_output.metrics.get("test_loss")

    if test_loss is not None: # Saves test loss and perplexity
        final_metrics["test_loss"] = round(float(test_loss), 4)
        final_metrics["perplexity"] = round(float(np.exp(float(test_loss))), 4)

    final_metrics["model_name"] = model_name # Adds the model name to the result

    final_metrics["model_checkpoint"] = model_checkpoint # Adds model checkpoint to the result

    return final_metrics, prediction_rows # Returns metrics and sample predictions

def save_preliminary_results(all_results, all_samples): # Saves all preliminary experiment results
    
    output_dir = Path("outputs/preliminary_experiments") # Defines the output folder

    output_dir.mkdir(parents=True, exist_ok=True) # Creates an output folder if needed

    csv_path = output_dir / "preliminary_model_results.csv" # Defines the CSV path

    json_path = output_dir / "preliminary_model_results.json" # Defines the JSON path

    samples_path = output_dir / "preliminary_samples.json" # Defines the sample output path

    with open(json_path, "w", encoding="utf-8") as file: # Saves results as JSON
        json.dump(all_results, file, indent=2)

    with open(samples_path, "w", encoding="utf-8") as file: # Saves sample predictions as JSON
        json.dump(all_samples, file, indent=2)

    fieldnames = list(all_results[0].keys()) # Gets all metric names from the first result

    with open(csv_path, "w", encoding="utf-8", newline="") as file: # Save results as a CSV
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved preliminary CSV results to {csv_path}")
    print(f"Saved preliminary JSON results to {json_path}")
    print(f"Saved preliminary sample predictions to {samples_path}")


# Main function for preliminary experiments.
def main():
    # Loads the same config used by the final training script
    base_config = load_config("configs/t5_config.yaml")

    # Loads, cleans, and formats the full dataset
    raw_dataset = prepare_dataset(base_config)

    # Creates a small dataset for preliminary experiments
    small_dataset = make_small_dataset(raw_dataset)

    # Creates a list to store model results
    all_results = []

    # Creates a dictionary to store sample outputs
    all_samples = {}

    # Loops through each candidate model
    for model_name, model_checkpoint in MODEL_CANDIDATES.items():
        # Runs one model experiment
        metrics, prediction_rows = run_experiment(
            model_name,
            model_checkpoint,
            base_config,
            small_dataset,
        )

        # Adds metrics to the results list
        all_results.append(metrics)

        # Saves only a few sample predictions for this model
        all_samples[model_name] = prediction_rows[:5]

    # Saves all results
    save_preliminary_results(all_results, all_samples)

    # Prints final results
    print("\nPreliminary experiment results:")
    for result in all_results:
        print(result)


# Runs main only when this file is executed directly
if __name__ == "__main__":
    main()
