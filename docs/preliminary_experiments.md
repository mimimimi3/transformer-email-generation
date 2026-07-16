# Preliminary Experiments


In order to finalize the training workflow, and choose the correct model, smaller implementation experiments were conducted. These tests showed that the dataset could be loaded correctly from Hugging Face, the expected columns were present, the email bodies and subject lines could be cleaned and formatted. The inputs and targets were also checked to see if they could be tokenized for T5. The models that were experimented with showed that they could train without any GPU memory issues, their predictions could be generated and decoded, ROUGE, BLEU, BERTScore, and test loss could be calculated. The outputs were saved and used for further analysis.

T5-small and FLAN-T5-small were evaluated on a small subset of the dataset, and their performances were compared. These results were preliminary rather than final because the experiment used a smaller dataset and fewer training epochs.

During experimentation, a decoding error occurred where prediction token IDs contained invalid values. This caused the decoding process to fail during metric calculation. In order to fix this issue, the prediction IDs had to be cleaned before decoding them into text. The final scripts now include this safeguard.

The preliminary experiments helped confirm that the models selected could be trained and evaluated on the chosen dataset, and that the pipeline was feasible. Therefore, full training runs were completed for both T5-small and FLAN-T5-small. The final results showed that FLAN-T5-small achieved better results across the main evaluation metrics.
