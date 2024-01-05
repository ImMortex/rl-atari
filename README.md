# stcngurs-proof-of-concept

## Setup .env variables (train config):
- There is a config that ensures the reproducibility of experiments and serves to control the training, e.g. Hyper parameters: ``train_config.py`` 
- Create your ``.env`` file in the project root, to set parameters of ``train_config.py`` 
- ``.env-example`` shows all needed variables as example
- You need to generate your own secrets for wandb and minio server
- HTTPS using BEARER_TOKEN


## Setup python
A python environment with cuda support is recommended. See https://pytorch.org/
````shell
pip install -r ./requirements.txt
````