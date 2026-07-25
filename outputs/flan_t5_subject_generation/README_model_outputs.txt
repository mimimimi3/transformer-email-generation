FLAN-T5 Email Subject Generation Demonstration

Model: deva-penumaka/flan-t5-email-subject-generator

Dataset: postbot/aeslc_kw

Dataset split: test

Number of generated samples: 10

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