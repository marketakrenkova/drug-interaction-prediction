#!/bin/bash

./kg_model.py -m complex -s ogb_settings -e 100 -emb 1000 -o adam -lr 0.001 -b 512 -nn 128 -d drugbank
./kg_model.py -m complex -s ogb_settings -e 50 -emb 1000 -o adam -lr 0.001 -b 512 -nn 128 -d biokg
./kg_model.py -m complex -s ogb_settings -e 50 -emb 1000 -o adam -lr 0.001 -b 512 -nn 128 -d hetionet
./kg_model.py -m distmul -s ogb_settings -e 50 -emb 2000 -o adam -lr 0.001 -b 512 -nn 128 -d drugbank
./kg_model.py -m transe -s ogb_settings -e 50 -emb 2000 -o adam -lr 0.0001 -b 512 -nn 128 -d drugbank
./kg_model.py -m transe -s ogb_settings -e 50 -emb 2000 -o adam -lr 0.0001 -b 512 -nn 128 -d biokg

