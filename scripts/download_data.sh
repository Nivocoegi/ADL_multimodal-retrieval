#!/bin/bash

mkdir -p /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/data

curl -L -o data/raw/flickr8k.zip \
  https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k

unzip /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/data/raw/flickr8k.zip -d /cfs/earth/scratch/vognic01/ADL_multimodal-retrieval/data

chmod +x scripts/download_data.sh

