# Imports csv, this can save predictions in spreadsheet-style CSV files
import csv

# Imports json, this can save metrics and samples as JSON files
import json

# Imports re, this can clean text using regular expressions
import re

# Imports Path which can create folders and file paths safely
from pathlib import Path

# Imports evaluate which can calculate ROUGE, BLEU, and BERTScore
import evaluate

# Imports numpy used for metric calculations which use arrays and averages
import numpy as np

# Imports torch as Hugging Face models run using PyTorch
import torch

# Imports yaml which can read settings from configs/t5_config.yaml
import yaml

# Imports load_dataset which can load the Hugging Face dataset
from datasets import load_dataset

# Imports the Hugging Face model, tokenizer, trainer, and training utilities
from transformers import (
    # Loads T5-small as a sequence-to-sequence model.
    # This is used because email bodies have to be analyzed for email subject generation
    AutoModelForSeq2SeqLM,

    # Loads the tokenizer that works with T5-small
    AutoTokenizer,

    # Ensures input examples are padded inside each batch
    DataCollatorForSeq2Seq,

    # Trainer used for sequence-to-sequence models
    Seq2SeqTrainer,

    # Used to Store training settings like batch size, epochs, and learning rate
    Seq2SeqTrainingArguments,

    # Used to make results more reproducible by setting a random seed
    set_seed,
)


# Loads the previously setup YAML config file
def load_config(config_path):
    # Opens the YAML config file in read mode
    with open(config_path, "r", encoding="utf-8") as file:
        # Converts the YAML file into a Python dictionary
        # This allows the rest of the program access settings
        return yaml.safe_load(file)


# Cleans the email body and subject text
def clean_text(text):
    # If the text is missing, this returns an empty string instead of crashing
    if text is None:
        return ""

    # Converts the input to a string to avoid type errors
    text = str(text)

    # Replaces line breaks with spaces, which gives the model cleaner text
    text = text.replace("\n", " ").replace("\r", " ")

    # Replaces multiple spaces, tabs, or weird spacing with just one space
    text = re.sub(r"\s+", " ", text)

    # Removes the extra spaces from the start and end
    return text.strip()


# Loads, cleans, filters, and formats the full dataset
def prepare_dataset(config):
    # Prints progress so we know the dataset is loading
    print("Loading the full dataset...")

    # Loads the dataset from Hugging Face using the dataset name from the config file
    dataset = load_dataset(config["dataset_name"])

    # Confirms that the required dataset splits exist
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset.keys())

    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing_splits)}")

    # Gets the dataset column that contains the email body from the config file
    body_column = config["body_column"]

    # Gets the dataset column that contains the email subject line from the config file
    target_column = config["target_column"]

    # Confirms that the configured columns exist in every dataset split
    required_columns = {body_column, target_column}

    for split_name in ("train", "validation", "test"):
        missing_columns = required_columns - set(dataset[split_name].column_names)

        if missing_columns:
            raise ValueError(
                f"The {split_name} split is missing required columns, they are: "
                f"{sorted(missing_columns)}")

    # Defines process of converting one raw dataset row into a T5 training example
    def format_example(example):
        # Cleans the email body text using the previously defined function
        email_body = clean_text(example[body_column])

        # Cleans the reference subject line using the previously defined function
        subject = clean_text(example[target_column])

        # Returns new columns that the training code will use
        return {
            # This is the input that T5 receives
            "input_text": config["task_prefix"] + email_body,

            # This is the target output T5 learns to generate
            "target_text": subject,

            # This stores the cleaned email body for later output inspection
            "raw_email": email_body,

            # This stores the cleaned original email subject for comparison
            "reference_subject": subject,
        }

    # Applies the previously defined formatting function to every example in the dataset
    dataset = dataset.map(format_example)

    # Defines the examples which are useful enough to keep
    def keep_example(example):
        # Keeps only examples with enough email text length and enough subject text length
        return (
            len(example["raw_email"]) >= config["min_email_chars"]
            and len(example["reference_subject"]) >= config["min_subject_chars"]
        )

    # Removes examples that are empty or don't meet our minimum length requirements
    dataset = dataset.filter(keep_example)

    # Confirms that each split still contains examples after filtering
    for split_name in ("train", "validation", "test"):
        if len(dataset[split_name]) == 0:
            raise ValueError(f"The {split_name} split is empty after filtering. "
                "Reduce min_email_chars or min_subject_chars.")

    # Creates the processed data folder in case it does not already exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Opens a small preview file which allows us to inspect the formatted data
    with open("data/processed/t5_preview.jsonl", "w", encoding="utf-8") as file:
        # Saves a maximum of 5 formatted training examples
        for example in dataset["train"].select(range(min(5, len(dataset["train"])))):
            # Converts the preview example into JSON text
            line = json.dumps({
                    # Saves the exact input text used by the model
                    "input_text": example["input_text"],

                    # Saves the exact target subject line which is used for training
                    "target_text": example["target_text"],},
                # Keeps the special characters readable instead of replacing them with a plain text code
                ensure_ascii=False,)

            # Outputs one JSON example per line
            file.write(line + "\n")

    # Prints dataset sizes after cleaning
    print(f"Train examples: {len(dataset['train'])}")

    # Prints validation size after cleaning
    print(f"Validation examples: {len(dataset['validation'])}")

    # Prints test size after cleaning
    print(f"Test examples: {len(dataset['test'])}")

    # Returns the full dataset cleaned and formatted
    return dataset


# Converts the text into token IDs that T5 can interpret
def tokenize_dataset(dataset, tokenizer, config):
    # Defines a function to tokenize a batch of examples
    def tokenize_batch(batch):
        # Tokenizes the input email text
        model_inputs = tokenizer(batch["input_text"], # Uses the formatted input text

            # Limits the email input length, according to the setting in the config file
            max_length=config["max_input_length"],

            # Cuts off inputs that are too long
            truncation=True,)

        # Tokenizes the target subject lines
        labels = tokenizer(text_target=batch["target_text"], # Tells Hugging Face these are output labels, not input text

            max_length=config["max_target_length"], # Limits the generated subject length

            truncation=True,) # Cuts off target subjects that are too long

        # Stores the target token IDs as "labels", to be used for supervised training
        model_inputs["labels"] = labels["input_ids"]

        # Returns tokenized model inputs and labels
        return model_inputs

    # Stores the original column names, to be removed after tokenization
    columns_to_remove = dataset["train"].column_names

    # Tokenizes the full dataset
    tokenized_dataset = dataset.map(
        # Uses the tokenize_batch function defined above
        tokenize_batch,

        # Processes the examples in batches for speed
        batched=True,

        # Removes old text columns, as the model only needs token IDs
        remove_columns=columns_to_remove,
    )

    # Returns the tokenized dataset
    return tokenized_dataset

# Cleans generated token IDs before decoding them into text
def clean_prediction_ids(predictions, tokenizer):
    # Converts predictions into a NumPy array
    predictions = np.asarray(predictions)

    # Replaces negative token IDs with the padding token ID
    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)

    # Replaces token IDs outside the tokenizer vocabulary with the padding token ID
    predictions = np.where(
        predictions >= len(tokenizer),
        tokenizer.pad_token_id,
        predictions,
    )

    # Converts token IDs into normal integer format
    return predictions.astype(np.int64)

# Defines the ROUGE metric function used during validation while training
# ROUGE is the metric used to compare generated text with a human written reference
def compute_metrics_builder(tokenizer):
    # Loads the ROUGE metric
    rouge = evaluate.load("rouge")

    # Defines the metric function that is called by the Trainer
    def compute_metrics(eval_prediction):
        # Splits the model predictions and reference labels
        predictions, labels = eval_prediction

        # Hugging Face versions can return predictions inside a tuple
        if isinstance(predictions, tuple):
            # Keeps only the actual prediction values
            predictions = predictions[0]

        # Replaces ignored label positions with the padding token
        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id,
        )

        # Cleans prediction token IDs before decoding
        predictions = clean_prediction_ids(predictions, tokenizer)

        # Converts labels to normal integer format
        labels = labels.astype(np.int64)

        # Decodes the generated token IDs into readable text
        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        # Decodes the reference label token IDs into readable text
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        # Removes extra spaces from generated subjects
        decoded_predictions = [text.strip() for text in decoded_predictions]

        # Removes extra spaces from the reference subjects
        decoded_labels = [text.strip() for text in decoded_labels]

        # Computes ROUGE scores between generated and reference subject lines
        result = rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )

        # Returns the ROUGE scores we want to track during training
        return {
            "rouge1": round(result["rouge1"], 4),
            "rouge2": round(result["rouge2"], 4),
            "rougeL": round(result["rougeL"], 4),
        }

    # Returns the metric function to the Trainer
    return compute_metrics

# Builds savable rows using predictions returned by trainer.predict()
def build_prediction_rows(prediction_output, tokenizer, raw_dataset):
    # Gets the generated token IDs
    predictions = prediction_output.predictions

    # Some Transformers versions return predictions inside a tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Decodes all generated subjects at once
    predictions = clean_prediction_ids(predictions, tokenizer)

    generated_subjects = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,)

    # Creates rows for saving and error analysis
    prediction_rows = []

    for i, generated_subject in enumerate(generated_subjects):
        # Gets the corresponding original test example
        example = raw_dataset["test"][i]

        prediction_rows.append(
            {
                "index": i,
                "email_body": example["raw_email"],
                "reference_subject": example["reference_subject"],
                "generated_subject": generated_subject.strip(),
            }
        )

    return prediction_rows


# Computes ROUGE, BLEU, BERTScore, and length statistics after generation has completed
def compute_final_metrics(prediction_rows, config):
    # Prints a progress message
    print("Computing the final metrics...")

    # Extracts the generated subject lines from the prediction rows
    generated_subjects = [row["generated_subject"] for row in prediction_rows]

    # Extracts the original reference subject lines from the prediction rows
    reference_subjects = [row["reference_subject"] for row in prediction_rows]

    # Creates a dictionary to store the results
    final_metrics = {}

    # Loads the ROUGE metric
    rouge = evaluate.load("rouge")

    # Computes the ROUGE scores
    rouge_result = rouge.compute(
        # Model-generated subject lines.
        predictions=generated_subjects,

        # Original subject lines for reference
        references=reference_subjects,

        # Using stemming for fairer word matching
        use_stemmer=True,
    )

    # Saves ROUGE-1
    final_metrics["rouge1"] = round(rouge_result["rouge1"], 4)

    # Saves ROUGE-2
    final_metrics["rouge2"] = round(rouge_result["rouge2"], 4)

    # Saves ROUGE-L
    final_metrics["rougeL"] = round(rouge_result["rougeL"], 4)

    # Loads SacreBLEU using Hugging Face's Evaluate.
    bleu = evaluate.load("sacrebleu")

    # Computes BLEU
    bleu_result = bleu.compute(
        # Model-generated subject lines
        predictions = generated_subjects,

        # SacreBLEU needs each reference to be inside its own list
        references = [[reference] for reference in reference_subjects],
    )

    # Saves BLEU score
    final_metrics["bleu"] = round(bleu_result["score"], 4)

    # Checks to confirm BERTScore is enabled in the config file
    if config.get("compute_bertscore", False):
        # Loads BERTScore metric
        bertscore = evaluate.load("bertscore")

        # Computes BERTScore
        bertscore_result = bertscore.compute(predictions = generated_subjects, # Model-generated subject lines

            # Original reference subject lines
            references = reference_subjects,

            # Tells BERTScore that the language is English
            lang = "en",)

        # Saves the average BERTScore precision
        final_metrics["bertscore_precision"] = round(float(np.mean(bertscore_result["precision"])), 4)

        # Saves the average BERTScore recall
        final_metrics["bertscore_recall"] = round(float(np.mean(bertscore_result["recall"])), 4)

        # Saves the average BERTScore F1
        final_metrics["bertscore_f1"] = round(float(np.mean(bertscore_result["f1"])), 4)

    # Counts the number of words in each generated subject
    generated_lengths = [len(text.split()) for text in generated_subjects]

    # Saves the average generated subject length
    final_metrics["average_generated_subject_words"] = round(float(np.mean(generated_lengths)), 2)

    # Returns all final metrics
    return final_metrics


# Calculates the longest common subsequence length, used for simple error analysis
# A longer common subsequence means the generated subject is more similar to the reference
def lcs_length(words_a, words_b):
    # Creates a table with one extra row and column
    table = [[0] * (len(words_b) + 1) for _ in range(len(words_a) + 1)]

    # Loops through words in the generated subject
    for i in range(1, len(words_a) + 1):
        # Loops through words in the reference subject
        for j in range(1, len(words_b) + 1):
            # Checks if the two current words match
            if words_a[i - 1] == words_b[j - 1]:
                # Extends the previous matching sequence
                table[i][j] = table[i - 1][j - 1] + 1

            # If the words do not match, keeps the best previous score
            else:
                # Chooses the better score from the left or above
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    # Returns the final longest common subsequence length
    return table[-1][-1]

# Converts text into normalized words for rough error analysis
def normalize_words(text):
    # Converts text to lowercase
    text = text.lower()

    # Replaces punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Replaces repeated whitespace with one space
    text = re.sub(r"\s+", " ", text)

    # Splits the cleaned text into individual words
    return text.strip().split()

# Computes a rough per-example ROUGE-L-style F1 score
def rough_rouge_l_f1(prediction, reference):

    # Normalizes generated and reference subjects
    pred_words = normalize_words(prediction)
    ref_words = normalize_words(reference)

    # Returns zero if either text is empty
    if len(pred_words) == 0 or len(ref_words) == 0:
        return 0.0

    # Computes the longest common subsequence length
    lcs = lcs_length(pred_words, ref_words)

    # Computes the precision-like overlap
    precision = lcs / len(pred_words)

    # Computes the recall-like overlap
    recall = lcs / len(ref_words)

    # Avoids division by zero.
    if precision + recall == 0:
        return 0.0

    # Returns an F1-style score
    return 2 * precision * recall / (precision + recall)


# Assigns a simple error category to each generated subject
# This helps identify errors and their frequency
def classify_error(generated_subject, reference_subject, rough_score):
    # Removes surrounding spaces for reliable comparisons
    generated_clean = generated_subject.strip()
    reference_clean = reference_subject.strip()

    # Identifies empty generations
    if generated_clean == "":
        return "Generated output is empty"

    # Identifies outputs that exactly match the reference subject
    if generated_clean.lower() == reference_clean.lower():
        return "Generated output is an exact match with the reference text"

    # Counts words in the generated subject
    word_count = len(generated_clean.split())

    # Identifies outputs that are likely too short
    if word_count <= 1:
        return "Generated output too short"

    # Identifies outputs that are likely too long for a subject line
    if word_count > 12:
        return "Generated output too long"

    # Identifies outputs that have very low overlap with the reference subject
    if rough_score < 0.15:
        return "Generated output has low overlap with reference"

    # Otherwise, marks the example for further review
    return "Generated output needs manual review"

# Saves predictions, readable samples, metrics, and error examples
def save_outputs(prediction_rows, final_metrics, test_results, config):
    # Defines the output folder
    output_dir = Path("outputs/t5_subject_generation")

    # Creates the output folder if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adds rough error-analysis fields to each prediction row
    for row in prediction_rows:
        # Compute rough ROUGE-L score for this example.
        score = rough_rouge_l_f1(row["generated_subject"], row["reference_subject"])

        # Stores the rounded rough score
        row["rough_rouge_l_f1"] = round(score, 4)

        # Stores the simple error category
        row["error_type"] = classify_error(row["generated_subject"], row["reference_subject"], score)

    # Defines the CSV file path for all predictions
    predictions_csv_path = output_dir / "t5_subject_predictions.csv"

    # Opens the predictions CSV file
    with open(predictions_csv_path, "w", encoding="utf-8", newline="") as file:
        # Defines the columns in the CSV file
        fieldnames = [
            "index",
            "email_body",
            "reference_subject",
            "generated_subject",
            "rough_rouge_l_f1",
            "error_type",
        ]

        # Creates the CSV writer
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Writes the CSV header row
        writer.writeheader()

        # Writes all the prediction rows
        writer.writerows(prediction_rows)

    # Defines the CSV file path for error examples
    errors_csv_path = output_dir / "t5_subject_error_examples.csv"

    # Sorts examples from lowest rough score to highest rough score
    worst_rows = sorted(prediction_rows, key=lambda row: row["rough_rouge_l_f1"])[:10]

    # Opens the error examples CSV file
    with open(errors_csv_path, "w", encoding="utf-8", newline="") as file:
        # Define the columns in the error CSV file.
        fieldnames = [
            "index",
            "email_body",
            "reference_subject",
            "generated_subject",
            "rough_rouge_l_f1",
            "error_type",
        ]

        # Creates the CSV writer
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Writes the CSV header row
        writer.writeheader()

        # Writes the worst examples
        writer.writerows(worst_rows)

    # Defines the readable text sample file path
    samples_txt_path = output_dir / "samples_subject.txt"

    # Decides how many readable samples to save
    num_samples = min(config["num_output_samples"], len(prediction_rows))

    # Opens the readable sample text file
    with open(samples_txt_path, "w", encoding="utf-8") as file:
        # Loops through the first few prediction rows
        for i, row in enumerate(prediction_rows[:num_samples], start=1):
            # Writes a separator line
            file.write("=" * 80 + "\n")

            # Writes the example number
            file.write(f"Example {i}\n\n")

            # Writes the email body heading
            file.write("EMAIL BODY:\n")

            # Writes the first 1100 characters of the email body
            file.write(row["email_body"][:1100] + "\n\n")

            # Writes the original subject
            file.write(f"REFERENCE SUBJECT: {row['reference_subject']}\n")

            # Writes the model-generated subject
            file.write(f"GENERATED SUBJECT: {row['generated_subject']}\n")

            # Writes the simple error category
            file.write(f"ERROR TYPE: {row['error_type']}\n")

            # Writes the rough per-example ROUGE-L score
            file.write(f"ROUGH ROUGE-L F1: {row['rough_rouge_l_f1']}\n\n")

    # Defines the JSON sample file path
    samples_json_path = output_dir / "samples_subject.json"

    # Opens the JSON sample file
    with open(samples_json_path, "w", encoding="utf-8") as file:

        # Save selected prediction rows as formatted JSON.
        json.dump(prediction_rows[:num_samples],
        file,
        indent=2,
        ensure_ascii=False,)

    # Retrieves test loss from trainer.predict()
    test_loss = test_results.get("test_loss",
        test_results.get("eval_loss"),)

    # Saves test loss only when it is available
    if test_loss is not None:
        final_metrics["test_loss"] = round(float(test_loss), 4,)

    # Defines the metrics file path
    metrics_path = output_dir / "test_metrics.json"

    # Opens the metrics JSON file
    with open(metrics_path, "w", encoding="utf-8") as file:
        # Save metrics as formatted JSON.
        # Keeps non-English text readable
        json.dump(final_metrics, file, indent=2, ensure_ascii=False,)

    # Prints where predictions were saved
    print(f"Saved predictions to {predictions_csv_path}")

    # Prints where error examples were saved
    print(f"Saved error examples to {errors_csv_path}")

    # Prints where readable samples were saved
    print(f"Saved readable samples to {samples_txt_path}")

    # Prints where metrics were saved
    print(f"Saved metrics to {metrics_path}")


# Main function which controls the full training script
def main():
    # Load project settings from the YAML config file.
    config = load_config("configs/t5_config.yaml")

    # Creates logs folder if needed
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)

    # Creates model output folder if needed
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Sets the random seed for reproducibility
    set_seed(config["seed"])

    # Loads, cleans, filters, and formats the full dataset
    raw_dataset = prepare_dataset(config)

    # Prints progress before loading the model
    print("Loading tokenizer and model...")

    # Loads the tokenizer for T5-small
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])

    # Loads the pretrained T5-small model
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_checkpoint"])

    # Tokenizes the full dataset
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, config)

    # Creates a data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        # Use the tokenizer for padding rules.
        tokenizer=tokenizer,

        # Uses the model to handle sequence-to-sequence labels correctly
        model=model,
    )

    # Defines the training settings
    training_args = Seq2SeqTrainingArguments(
        # Folder where model checkpoints and final weights are saved.
        output_dir=config["output_dir"],

        # Evaluates once after every epoch
        eval_strategy="epoch",

        # Saves a checkpoint once after every epoch
        save_strategy="epoch",

        # Retrieves learning rate from the config file.
        learning_rate=config["learning_rate"],

        # Retrieves training batch size from the config file.
        per_device_train_batch_size=config["train_batch_size"],

        # Retrieves evaluation batch size from the config file.
        per_device_eval_batch_size=config["eval_batch_size"],

        # Retrieves weight decay from the config file.
        weight_decay=config["weight_decay"],

        # Retrieves number of full passes through the training dataset.
        num_train_epochs=config["num_train_epochs"],

        # Generates text during evaluation so ROUGE can be calculated
        predict_with_generate=True,

        # Retrieves maximum generated subject length during evaluation
        generation_max_length=config["max_target_length"],

        # Beam search setting during evaluation.
        generation_num_beams=config["num_beams"],

        # Uses mixed precision only if enabled and CUDA is available
        fp16=config["fp16"] and torch.cuda.is_available(),

        # Folder where logs are saved.
        logging_dir=config["logging_dir"],

        # Prints training logs every 25 steps
        logging_steps=25,

        # Disables external logging services such as Weights & Biases
        report_to=config["report_to"],

        # Keeps only two checkpoints to save disk space
        save_total_limit=2,

        # Reloads the best checkpoint when training finishes
        load_best_model_at_end=True,

        # Uses ROUGE-L to decide which checkpoint is best
        metric_for_best_model="rougeL",

        # Higher ROUGE-L is better.
        greater_is_better=True,
    )

    # Creates the Hugging Face Trainer
    trainer = Seq2SeqTrainer(
        # Model to fine-tune
        model=model,

        # Training settings
        args=training_args,

        # Full tokenized training split
        train_dataset=tokenized_dataset["train"],

        # Full tokenized validation split
        eval_dataset=tokenized_dataset["validation"],

        # Supplies the tokenizer as the Trainer's processing class
         processing_class = tokenizer,

        # Data collator used for batch padding
        data_collator=data_collator,

        # ROUGE metric function used during validation
        compute_metrics=compute_metrics_builder(tokenizer),
    )

    # Prints a progress message before the training starts
    print("Starting the training...")

    # Fine-tune the model on the full training dataset.
    trainer.train()

    # Generates predictions and evaluates the full test set in one operation
    print("Generating predictions and evaluating the full test set...")

    prediction_output = trainer.predict(tokenized_dataset["test"])

    # Stores the test loss and Trainer metrics
    test_results = prediction_output.metrics

    # Prints progress before saving model weights
    print("Saving final model weights...")

    # Saves the fine-tuned model weights to models/t5_subject_model/
    trainer.save_model(config["output_dir"])

    # Saves tokenizer files needed to reload the model later
    tokenizer.save_pretrained(config["output_dir"])

    # Combines the generated predictions with the original test examples
    prediction_rows = build_prediction_rows(prediction_output,
        tokenizer,
        raw_dataset,)

    # Computes the final ROUGE, BLEU, BERTScore, and length metrics.
    final_metrics = compute_final_metrics(prediction_rows, config)

    # Saves predictions, metrics, samples, and error examples.
    save_outputs(prediction_rows, final_metrics, test_results, config)

    # Prints a final completion message.
    print("All tasks completed.")


# This block runs main() only when this file is executed directly.
if __name__ == "__main__":
    # Start the full training workflow.
    main()