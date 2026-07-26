DistilGPT-2 Email Continuation Demonstration

Model: deva-penumaka/distilgpt2-email-continuation-subject-and-body

Condition: subject_and_body

Dataset: postbot/aeslc_kw

Dataset split: test

Number of generated samples: 10

Decoding:
- do_sample=True
- temperature=0.8
- top_p=0.9
- repetition_penalty=1.2
- no_repeat_ngram_size=3
- seed=42

Description:
This script loads the fine-tuned DistilGPT-2 model from the Hugging Face Hub. It also streams the test split of the AESLC
dataset and selects 10 valid email examples for the
model to process.

Each email body is split into an incomplete prompt and a held-out
continuation. The model then generates the missing continuation text.

Files:
- distilgpt2_sample_001.txt to distilgpt2_sample_010.txt
  contain the individual generated examples.

- distilgpt2_generated_samples.json contains all
  generated examples in one JSON file.

- README_model_outputs.txt contains this description.

This script does not train the model.