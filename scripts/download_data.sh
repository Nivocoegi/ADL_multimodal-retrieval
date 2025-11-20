#!/bin/bash

mkdir -p data/raw
curl -L -o data/raw/flickr8k.zip \
  https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
unzip data/raw/flickr8k.zip -d data/raw/

chmod +x scripts/download_data.sh

