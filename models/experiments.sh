#!/bin/bash


# this sript runs kg_model.py where KGs are trained and evaluated
# here, we can run several experiments at once with predifedned hyper-parameters 

# run 1 - rotate 
./kg_model.py -m rotate -s best_pipeline4 -r run1 -e 50 -emb 500 -o adam -lm 1.09 -lr 0.002 -b 512 -nn 6 -d interactions
./kg_model.py -m rotate -s best_pipeline4 -r run1 -e 50 -emb 1000 -o adam -lm 2.89 -lr 0.005 -b 512 -nn 79 -d drugbank 
# ./kg_model.py -m rotate -s best_pipeline4 -r run1 -e 50 -emb 2500 -o adam -lr 0.008 -b 512 -nn 5 -d biokg
./kg_model.py -m rotate -s best_pipeline4 -r run1 -e 50 -emb 1500 -o adam -lr 0.002 -b 512 -nn 14 -d hetionet 

# run 1 - complex
./kg_model.py -m complex -s best_pipeline4 -r run1 -e 50 -emb 2000 -o adam -lm 2.4 -lr 0.001 -b 512 -nn 8 -d interactions
./kg_model.py -m complex -s best_pipeline4 -r run1 -e 50 -emb 3000 -o adam -lm 0.59 -lr 0.002 -b 512 -nn 32 -d drugbank 
./kg_model.py -m complex -s best_pipeline4 -r run1 -e 50 -emb 1500 -o adam -lm 0.37 -lr 0.009 -b 512 -nn 26 -d biokg 
./kg_model.py -m complex -s best_pipeline4 -r run1 -e 50 -emb 500 -o adam -lm 2.3 -lr 0.003 -b 512 -nn 14 -d hetionet 




