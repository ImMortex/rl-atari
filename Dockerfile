FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

ARG WANDB_KEY

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv --no-install-recommends

# all below copied files will be inside dir /app
WORKDIR /app

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

ARG CACHEBUST="$(date)"

# copy data
COPY ./README.md ./README.md
COPY ./.env-example ./.env-example
COPY ./config ./config
COPY ./src/common ./src/common
COPY ./src/overrides ./src/overrides
COPY ./pretrained_models/resnet18_breakout_out512_seq1/model ./pretrained_models/resnet18_breakout_out512_seq1/model
COPY ./pretrained_models/resnet18_breakout_out512_seq2/model ./pretrained_models/resnet18_breakout_out512_seq2/model
COPY ./pretrained_models/resnet18_breakout_out512_seq4/model ./pretrained_models/resnet18_breakout_out512_seq4/model
COPY ./pretrained_models/resnet18_breakout_out512_seq8/model ./pretrained_models/resnet18_breakout_out512_seq8/model
COPY ./pretrained_models/resnet50_breakout_out512_seq1/model ./pretrained_models/resnet50_breakout_out512_seq1/model
COPY ./pretrained_models/resnet50_breakout_out512_seq2/model ./pretrained_models/resnet50_breakout_out512_seq2/model
COPY ./pretrained_models/resnet50_breakout_out512_seq4/model ./pretrained_models/resnet50_breakout_out512_seq4/model
COPY ./pretrained_models/resnet50_breakout_out512_seq8/model ./pretrained_models/resnet50_breakout_out512_seq8/model
COPY ./pretrained_models/naturecnn_breakout_out512_seq1/model ./pretrained_models/naturecnn_breakout_out512_seq1/model
COPY ./pretrained_models/naturecnn_breakout_out512_seq2/model ./pretrained_models/naturecnn_breakout_out512_seq2/model
COPY ./pretrained_models/naturecnn_breakout_out512_seq4/model ./pretrained_models/naturecnn_breakout_out512_seq4/model
COPY ./pretrained_models/naturecnn_breakout_out512_seq8/model ./pretrained_models/naturecnn_breakout_out512_seq8/model
COPY minio_test.py minio_test.py
COPY main.py main.py
COPY wandb_login.py wandb_login.py

ARG CACHEBUST

# execute experiment
CMD python3 main.py
