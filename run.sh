#!/usr/bin/env bash

python Train_Transformer-MixSpeech_with_ADV.py \
    hparams/train_transformer-MixSpeech_with_ADV.yaml \
    --device=cuda:0 \
    --data_folder=/mnt/nas5/Jacky/ASR_Dataset/AISHELL
