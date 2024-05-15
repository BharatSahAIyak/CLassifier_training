# NER_training

The repository can be used to train an NER models. 

The training data must be kept in the data folder combined into a single file - train.csv. The csv must follow the same structure as [this](https://huggingface.co/datasets/GautamR/akai_flow_classifier_seed_pest_scheme) HF dataset.

Another csv test.csv can be added for evaluation metrics which will be populated in the logs folder 

One just needs to run the command : 

``` python3 '/trainer/tain_bert.py' ```
