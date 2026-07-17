# transformer-email-generation
Fine-tuning transformer models for email generation, including email continuation and subject generation.

Email subject generation: 

T5-small was first implemented as a baseline because it is a lightweight sequence-to-sequence model. Preliminary experiments were conducted on a small subset of the dataset to compare T5-small to FLAN-T5-small. The Preliminary experiment results showed that FLAN-T5-small could perform better compared to T5-small. Both models were therefore trained on the full training dataset and evaluated on the test dataset, FLAN-T5-small outperformed T5-small across ROUGE, BLEU, BERTScore, test loss and perplexity. This boost in performance of FLAN-T5-small could be due to instruction tuning done on the model.

The final metrics of both the models are as follows:


| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 | Test Loss | Perplexity |
|---|--:|---:|---:|---:|---:|---:|---:|
| T5-small | 0.3173 | 0.1755 | 0.3091 | 10.3159 | 0.8798 | 3.0606 | 21.3394 |
| FLAN-T5-small | 0.3328 | 0.1811 | 0.3241 | 12.0934 | 0.8819 | 2.8305 | 16.9544 |

These results suggest that FLAN-T5-small is better suited to the task of email subject generation.
