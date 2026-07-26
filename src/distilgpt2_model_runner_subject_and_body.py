import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import load_config
from utils.continuation import (
    build_prompt_text,
    extract_generated_continuation,
    split_email_body,
)
from utils.text_cleaning import clean_text

# Finds the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

# Files and folders locations
CONFIG_FILE = PROJECT_FOLDER / "configs/distilgpt2_config.yaml"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "distilgpt2_email_continuation"

# Prompt condition to demonstrate:
# - "body_only": incomplete email body only
# - "subject_and_body": subject line plus incomplete email body
CONDITION = "subject_and_body"

# Public Hugging Face model repository
MODEL_ID = "deva-penumaka/distilgpt2-email-continuation-subject-and-body"

# Number of examples
NUMBER_OF_SAMPLES = 10

# Sampling settings reduce the phrase-looping that greedy decoding often causes
# with DistilGPT-2 email continuations.
DO_SAMPLE = True
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3
SEED = 42


def main():
    # Loads the YAML configuration file

    print("Loading the configuration file...")

    config = load_config(CONFIG_FILE)

    # Creates the outputs folder for this condition

    condition_output_folder = OUTPUT_FOLDER / CONDITION / "model_runner"
    condition_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading the tokenizer from Hugging Face...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # GPT-2 tokenizers do not ship with a pad token; reuse EOS for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading fine-tuned DistilGPT-2 model from Hugging Face...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Uses the GPU when one is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    # Evaluation mode ensures the model is not training
    model.eval()

    # Makes sampling-based generation reproducible across runs
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Using device: {device}")
    print(f"Using condition: {CONDITION}")
    print(
        "Decoding: "
        f"do_sample={DO_SAMPLE}, temperature={TEMPERATURE}, "
        f"top_p={TOP_P}, repetition_penalty={REPETITION_PENALTY}, "
        f"no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}"
    )
    # Streams the test dataset

    print("Loading test examples from Hugging Face...")

    test_dataset = load_dataset(
        config["dataset_name"],
        split="test",
        streaming=True,
    )

    # Column names come from the YAML configuration
    body_column = config["body_column"]
    subject_column = config["subject_column"]

    selected_samples = []

    # Cleans examples until there are 10 valid samples

    print(f"Selecting {NUMBER_OF_SAMPLES} valid test examples...")

    for dataset_index, example in enumerate(test_dataset):
        email_body = clean_text(
            example[body_column],
            strip_artifacts=True,
            preserve_newlines=True,
        )
        subject = clean_text(
            example[subject_column],
            strip_artifacts=True,
            preserve_newlines=True,
        )
        email_prompt, reference_continuation = split_email_body(email_body)

        # Uses the same filtering rules used during training
        if len(email_body) < config["min_email_chars"]:
            continue

        if len(subject) < config["min_subject_chars"]:
            continue

        if len(email_prompt) < config["min_prompt_chars"]:
            continue

        if len(reference_continuation) < config["min_continuation_chars"]:
            continue

        prompt_text = build_prompt_text(
            CONDITION,
            subject,
            email_prompt,
        )

        selected_samples.append(
            {
                "dataset_index": dataset_index,
                "subject": subject,
                "email_prompt": email_prompt,
                "reference_continuation": reference_continuation,
                "prompt_text": prompt_text,
            }
        )

        # Stops as soon as enough valid examples are collected
        if len(selected_samples) == NUMBER_OF_SAMPLES:
            break

    print(f"Selected {len(selected_samples)} examples.")

    # Generates one continuation for each selected prompt

    print("Generating email continuations...")

    results = []

    for index, sample in enumerate(selected_samples):
        prompt_text = sample["prompt_text"]

        model_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=config["max_length"],
        )

        # Move every input tensor to the CPU or GPU
        model_inputs = {
            name: tensor.to(device)
            for name, tensor in model_inputs.items()
        }

        # inference_mode prevents training calculations
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        generated_continuation = extract_generated_continuation(
            decoded,
            prompt_text,
        )

        results.append(
            {
                "sample_number": index + 1,
                "dataset_index": sample["dataset_index"],
                "condition": CONDITION,
                "subject": sample["subject"],
                "email_prompt": sample["email_prompt"],
                "prompt_text": prompt_text,
                "reference_continuation": sample["reference_continuation"],
                "generated_continuation": generated_continuation,
                "decoding": {
                    "do_sample": DO_SAMPLE,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "repetition_penalty": REPETITION_PENALTY,
                    "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                    "seed": SEED,
                },
            }
        )

    # Saves each generated example separately

    print("Saving individual sample files...")

    for result in results:
        sample_number = result["sample_number"]

        output_file = (
            condition_output_folder
            / f"distilgpt2_sample_{sample_number:03d}.txt"
        )

        output_text = (
            f"SAMPLE {sample_number}\n"
            f"{'=' * 70}\n\n"
            f"CONDITION:\n"
            f"{result['condition']}\n\n"
            f"SUBJECT:\n"
            f"{result['subject']}\n\n"
            f"EMAIL PROMPT:\n"
            f"{result['email_prompt']}\n\n"
            f"REFERENCE CONTINUATION:\n"
            f"{result['reference_continuation']}\n\n"
            f"GENERATED CONTINUATION:\n"
            f"{result['generated_continuation']}\n"
        )

        output_file.write_text(
            output_text,
            encoding="utf-8",
        )

    # Saves all results in one JSON file

    json_file = (
        condition_output_folder
        / "distilgpt2_generated_samples.json"
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
        condition_output_folder
        / "README_model_outputs.txt"
    )

    description = f"""
DistilGPT-2 Email Continuation Demonstration

Model: {MODEL_ID}

Condition: {CONDITION}

Dataset: {config["dataset_name"]}

Dataset split: test

Number of generated samples: {NUMBER_OF_SAMPLES}

Decoding:
- do_sample={DO_SAMPLE}
- temperature={TEMPERATURE}
- top_p={TOP_P}
- repetition_penalty={REPETITION_PENALTY}
- no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}
- seed={SEED}

Description:
This script loads the fine-tuned DistilGPT-2 model from the Hugging Face Hub. It also streams the test split of the AESLC
dataset and selects {NUMBER_OF_SAMPLES} valid email examples for the
model to process.

Each email body is split into an incomplete prompt and a held-out
continuation. The model then generates the missing continuation text.

Files:
- distilgpt2_sample_001.txt to distilgpt2_sample_{NUMBER_OF_SAMPLES:03d}.txt
  contain the individual generated examples.

- distilgpt2_generated_samples.json contains all
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
    print(f"Outputs saved in: {condition_output_folder}")


if __name__ == "__main__":
    main()
