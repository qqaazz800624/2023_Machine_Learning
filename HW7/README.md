# 2023 Machine Learning HW7: BERT

I use hfl/chinese-macbert-large from Hugging Face pretrain model.

## Qucik start
    To run my code, simply first run the following three .py files
    - d11948002_hw7_model1.py
    - d11948002_hw7_model2.py
    - d11948002_hw7_model3.py

These will generate model configurations and checkpoints in folders saved_model/model1, saved_model/model2 and saved_model/model3 that will be used in d11948002_hw7_ensemble.py

The method I adopted to ensemble is to average three models' output logits.
    - d11948002_hw7_ensemble.py

References: https://huggingface.co/hfl/chinese-macbert-large
