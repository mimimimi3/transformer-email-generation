import json
import re
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Finds the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

# Files and folders locations
CONFIG_FILE = PROJECT_FOLDER / "configs/flan_t5_config.yaml"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "flan_t5_subject_generation"

# Public Hugging Face model repository
MODEL_REPOSITORY = (
    "deva-penumaka/flan-t5-email-subject-generator"
)

# Number of examples
NUMBER_OF_SAMPLES = 10

def clean_text(text):
    """
    Removes line breaks and extra spaces from text
    """

    if text is None:
        return ""

    text = str(text)

    # Replaces line breaks with spaces
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Replaces repeated spaces with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    # Loads the YAML configuration file

    print("Loading the configuration file...")

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Creates the outputs folder

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Downloads and loads the fine-tuned model

    print("Loading the tokenizer from Hugging Face...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPOSITORY
    )

    print("Loading fine-tuned model from Hugging Face...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_REPOSITORY
    )

    # Uses the GPU when one is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    # Evaluation mode ensures the model is not training
    model.eval()

    print(f"Using device: {device}")

    # Streams the test dataset

    print("Loading test examples from Hugging Face...")

    test_dataset = load_dataset(
        config["dataset_name"],
        split="test",
        streaming=True,
    )

    # Extracts names come from the YAML configuration
    body_column = config["body_column"]
    target_column = config["target_column"]

    selected_samples = []

    # Cleans examples until there are 10 valid samples

    print("Selecting 10 valid test examples...")

    for dataset_index, example in enumerate(test_dataset):
        email_body = clean_text(
            example[body_column]
        )

        reference_subject = clean_text(
            example[target_column]
        )

        # Uses the same filtering rules used during training
        if len(email_body) < config["min_email_chars"]:
            continue

        if len(reference_subject) < config["min_subject_chars"]:
            continue

        # Adds the same prefix that was used during training
        input_text = (
            config["task_prefix"]
            + email_body
        )

        selected_samples.append(
            {
                "dataset_index": dataset_index,
                "email_body": email_body,
                "reference_subject": reference_subject,
                "input_text": input_text,
            }
        )

        # Stops as soon as 10 valid examples are collected
        if len(selected_samples) == NUMBER_OF_SAMPLES:
            break

    print(
        f"Selected {len(selected_samples)} examples."
    )

    # Converts the 10 inputs into model tokens

    input_texts = [
        sample["input_text"]
        for sample in selected_samples
    ]

    model_inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_input_length"],
    )

    # Move every input tensor to the CPU or GPU
    model_inputs = {
        name: tensor.to(device)
        for name, tensor in model_inputs.items()
    }

    # Generates the email subjects

    print("Generating email subject lines...")

    # inference_mode prevents training calculations.
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=config["max_target_length"],
            num_beams=config["num_beams"],
        )

    generated_subjects = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # Builds the final result records

    results = []

    for index in range(NUMBER_OF_SAMPLES):
        sample = selected_samples[index]

        result = {
            "sample_number": index + 1,
            "dataset_index": sample["dataset_index"],
            "email_body": sample["email_body"],
            "reference_subject": sample[
                "reference_subject"
            ],
            "generated_subject": generated_subjects[
                index
            ].strip(),
        }

        results.append(result)

    # Saves each generated example separately

    print("Saving individual sample files...")

    for result in results:
        sample_number = result["sample_number"]

        output_file = (
            OUTPUT_FOLDER
            / f"flan_t5_sample_{sample_number:03d}.txt"
        )

        output_text = (
            f"SAMPLE {sample_number}\n"
            f"{'=' * 70}\n\n"
            f"EMAIL BODY:\n"
            f"{result['email_body']}\n\n"
            f"REFERENCE SUBJECT:\n"
            f"{result['reference_subject']}\n\n"
            f"GENERATED SUBJECT:\n"
            f"{result['generated_subject']}\n"
        )

        output_file.write_text(
            output_text,
            encoding="utf-8",
        )

    # Saves all results in one JSON file

    json_file = (
        OUTPUT_FOLDER
        / "flan_t5_generated_samples.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Saves the required description file

    description_file = (
        OUTPUT_FOLDER
        / "README_model_outputs.txt"
    )

    description = f"""
FLAN-T5 Email Subject Generation Demonstration

Model: {MODEL_REPOSITORY}

Dataset: {config["dataset_name"]}

Dataset split: test

Number of generated samples: {NUMBER_OF_SAMPLES}

Description:
This script downloads the fine-tuned FLAN-T5 model from
Hugging Face. It also streams the test split of the AESLC
dataset and selects 10 valid email examples for the model to process.

The model then generates one email subject line for each
email body.

Files:
- flan_t5_sample_001.txt to flan_t5_sample_010.txt
  contain the individual generated examples.

- flan_t5_generated_samples.json contains all
  generated examples in one JSON file.

- README_model_outputs.txt contains this description.

This script does not train the model.
""".strip()

    description_file.write_text(
        description,
        encoding="utf-8",
    )

    # Prints completion information

    print()
    print("Inference completed successfully.")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
