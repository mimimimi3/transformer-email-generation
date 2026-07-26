DistilGPT-2 Email Continuation Demonstration

Model: C:\Users\Galen\Documents\GitHub\transformer-email-generation\models\distilgpt2_continuation\body_only

Condition: body_only

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

Files:
- distilgpt2_sample_001.txt to distilgpt2_sample_010.txt
  contain the individual generated examples.

- distilgpt2_generated_samples.json contains all
  generated examples in one JSON file.

- README_model_outputs.txt contains this description.
