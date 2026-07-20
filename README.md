# Email Generation: Transformer Model Adaptation

## Project Overview

### Project Objects
This project focuses on fine-tuning pretrained transformer models for email generation tasks. Two generation tasks are explored:

**1. Email Subject Generation**

*Task:*  
Generate an appropriate email subject line based on the given email body.

*Models:*
- T5-small
- FLAN-T5-small

**2. Email Continuation Generation**

*Task:*
Generate a continuation of an email based on the provided email body.

*Model:*
- DistilGPT-2


### Dataset:
[postbot/aeslc_kw](https://huggingface.co/datasets/postbot/aeslc_kw)

**Dataset Information:**

| Attribute | Value |
|-----------|-------|
| Number of rows | 18,302 |
| Total file size | 30.6 MB |

This Hugging Face dataset is used because it is well-suited for this project, as it contains both email subject lines and email bodies.

### Research and Selection of Methods
T5-small and FLAN-T5-small are both T5-style transformer models used in this project and are well suited for email subject generation because they follow a text-to-text framework, enabling them to generate appropriate subject lines based on the content of an email body (Raffel et al., 2019; Chung et al., 2022).

DistilGPT-2, another transformer model used in this project, is suitable for email continuation generation because it is an autoregressive language model designed to predict the next tokens in a sequence, making it effective for generating coherent text continuations (Hugging Face, 2023).

The selected models meet the goals of this project by providing lightweight and reproducible models that specialize in email subject generation and email continuation generation.

### Preliminary Experiments
See the 'docs' fold for detailed information.

---

## Setup Instructions and Usage Examples:
### Requirements

We recommend using Docker to reproduce the project environment. The Docker image provides a consistent Python environment with all required dependencies installed.

Before running the project, make sure the following software is installed:

- Git
- Docker

### Setup Instructions

*1. Clone the Repository*

```bash
git clone https://github.com/mimimimi3/transformer-email-generation

cd transformer-email-generation
```

*2. Build the Docker Image*

Build the Docker image and assign a name to it:

```bash
docker build -t <docker-image-name> .
```

Example:

```bash
docker build -t email-generation .
```

*3. Run the Docker Container*

Run the Docker container using the image name:

```bash
docker run -it <docker-image-name>
```

Example:

```bash
docker run -it email-generation
```

*4. Run Project Scripts*

All scripts are located in the `src/` directory.

- Fine-tune, implement, evaluate the T5-small Model and save the output locally

```bash
docker run -it -v $(pwd):/workdir <docker-image-name> python src/train_t5.py
```

Example:

```bash
docker run -it -v $(pwd):/workdir email-generation python src/train_t5.py
```

- Fine-tune, implement, evaluate the FLAN-T5-small Model and save the output locally

```bash
docker run -it -v $(pwd):/workdir <docker-image-name> python src/train_flan_t5.py
```

Example:

```bash
docker run -it -v $(pwd):/workdir email-generation python src/train_flan_t5.py
```

- Fine-tune, implement, evaluate the DistilGPT2-small Model and save the output locally

```bash
docker run -it -v $(pwd):/workdir <docker-image-name> python src/train_distilgpt2.py
```

Example:

```bash
docker run -it -v $(pwd):/workdir email-generation python src/train_distilgpt2.py
```


---
## Model Implementation

### Dataset Preparation: 
Uninformative patterns (e.g., text enclosed in angle brackets such as << >>), attachment filename lines, and extra whitespace were removed during data cleaning. The dataset is already split into training, validation, and test sets, making it ready for model traininig and evaluation.

### Email subject generation: 

T5-small was chosen for this task because it is a lightweight sequence-to-sequence model, well suited for email subject generation (Hugging Face, 2024).

A baseline experiment was conducted using the original pretrained T5-small and FLAN-T5-small models, this experiment did not do any fine-tuning on the models using the email dataset, before testing them on a subset of the dataset. The results are shown below:

The baseline was evaluated on 500 test examples from the dataset.

| Model                    | ROUGE-1 | ROUGE-2 | ROUGE-L |   BLEU | BERTScore F1 | Avg. Generated Words | Test Loss | Perplexity |
| ------------------------ | --------| -------| ---------| -------| -------------| ---------------------| ----------| -----------|
| Pretrained T5-small      |  0.1071 |  0.0413 |  0.0963 | 0.9168 |       0.8315 |                27.29 |    6.1733 |   479.7625 |
| Pretrained FLAN-T5-small |  0.2721 |  0.1409 |  0.2588 | 3.3766 |       0.8662 |                 8.14 |    2.9279 |    18.6875 |

We can see from the above results that, the pretrained FLAN-T5-small model performed relatively better than the pretrained T5-small model. However, fine-tuning the models on the email dataset helped improve T5-small's performance greatly.

Preliminary experiments were also conducted on a small subset of the dataset to compare T5-small to FLAN-T5-small. The Preliminary experiment fine-tuned the models on a subset of the dataset, results showed that FLAN-T5-small could perform better compared to T5-small. 

Both models were therefore trained on the full training dataset and evaluated on the full test dataset, FLAN-T5-small outperformed T5-small across ROUGE, BLEU, BERTScore, test loss and perplexity. This boost in performance of FLAN-T5-small could be due to instruction tuning done on the model (“Google/Flan-T5-Small · Hugging Face,” 2024).

The final metrics of both the models are as follows:


| Model        | ROUGE-1  | ROUGE-2 | ROUGE-L | BLEU    | BERTScore F1 | Test Loss | Perplexity |
|--------------|----------|---------|---------|---------|--------------|-----------|------------|
| T5-small     | 0.3173   | 0.1755  | 0.3091  | 10.3159   | 0.8798     | 3.0606    | 21.3394    |
| FLAN-T5-small| 0.3328   | 0.1811  | 0.3241  | 12.0934   | 0.8819     | 2.8305    | 16.9544    |

These results suggest that FLAN-T5-small is better suited to the task of email subject generation.

We can also see that fine-tuning on the email dataset helped the models improve their perplexity, BLEU, BERTScore F1 and ROUGE-L scores. The BLEU score for FLAN-T5-small improved from 3.3766 to 12.0934 after fine tuning, while perplexity decreased from 18.6875 to 16.9544.

### Email continuation generation:

DistilGPT-2 was selected for email continuation because it is a lightweight decoder-only language model that is well suited to next-token prediction. The continuation setup splits each email body into an incomplete prompt and a held-out continuation, and compares prompt conditions such as body-only versus subject-and-body.

The pretrained row below is the untouched DistilGPT-2 baseline on `body_only`. The fine-tuned row is from a shorter preliminary run (1 epoch, 2048 training examples).

| Model | Stage | Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 | Test Loss | Perplexity |
|---|---|---|--:|---:|---:|---:|---:|---:|---:|
| DistilGPT-2 | Pretrained | body_only | 0.1434 | 0.0219 | 0.1220 | 2.0576 | 0.8201 | 3.7616 | 43.0177 |
| DistilGPT-2 | Fine-tuned | body_only | 0.1543 | 0.0301 | 0.1279 | 2.4121 | 0.8155 | 3.1273 | 22.8127 |

Fine-tuning lowered test loss and perplexity substantially (about 47% perplexity reduction versus the pretrained baseline), with small gains in ROUGE and BLEU. BERTScore stayed similar. These are encouraging preliminary continuation results; a full-data / multi-epoch run and the `subject_and_body` condition will be added next.


---
Group 12:
- L. Penumaka
- G. Wu
- M. Zhang

References:
google-t5/t5-small · Hugging Face. (2024, March 5). Huggingface.co. https://huggingface.co/google-t5/t5-small
google/flan-t5-small · Hugging Face. (2024). In huggingface.co. https://huggingface.co/google/flan-t5-small
