import wandb
import json
api = wandb.Api()

runs = api.runs("stcngurs/stable-gym")

for run in runs:

    config = json.loads(run.json_config)

    pretrain_mode = "no pretraining"
    pretrained = False
    unfrozen = False
    if 'pretrained_vision_encoder_model_path' in config:

        if config['pretrained_vision_encoder_model_path']['value'] is not None and config['pretrained_vision_encoder_model_path']['value'] != "None":
            pretrained = True

    if 'unfreeze_pretrained_vision_encoder_weights' in config:
        if config['unfreeze_pretrained_vision_encoder_weights']['value']:
            unfrozen = True

    if pretrained and unfrozen:
        pretrain_mode = "finetuning"

    if pretrained and not unfrozen:
        pretrain_mode = "linear probing"
    run.config["pretrain_mode"] = pretrain_mode
    run.update()