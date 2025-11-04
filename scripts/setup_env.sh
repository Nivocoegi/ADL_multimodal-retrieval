#!/bin/bash
# Erstellt Environment und installiert alle Dependencies im Cluster

python3 -m venv ~/envs/multimodal
source ~/envs/multimodal/bin/activate

pip install --upgrade pip
pip install -r requirements.txt