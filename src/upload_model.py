from pathlib import Path

from huggingface_hub import HfApi
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# Folder created for the training script.
LOCAL_MODEL_DIRECTORY = Path(
    "models/flan_t5_subject_model"
)

# The model path on Hugging Face
MODEL_REPOSITORY = (
    "deva-penumaka/flan-t5-email-subject-generator"
)


def main() -> None:
    if not LOCAL_MODEL_DIRECTORY.exists():
        raise FileNotFoundError(
            "The saved model folder was not found: "
            f"{LOCAL_MODEL_DIRECTORY}"
        )

    print("Creating the public Hugging Face repository for the trained model...")

    api = HfApi()

    api.create_repo(
        repo_id=MODEL_REPOSITORY,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    print("Loading the saved tokenizer for the trained model...")

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIRECTORY
    )

    print("Loading the saved fine-tuned model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        LOCAL_MODEL_DIRECTORY
    )

    print("Uploading the model weights...")

    model.push_to_hub(
        MODEL_REPOSITORY,
        private=False,
    )

    print("Uploading the tokenizer...")

    tokenizer.push_to_hub(
        MODEL_REPOSITORY,
    )

    print("Upload completed.")
    print(f"Repository: {MODEL_REPOSITORY}")


if __name__ == "__main__":
    main()
