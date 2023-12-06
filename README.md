# Drug interaction prediction

This repository complements a master thesis __Using Knowledge Graphs for Predicting Food Supplement Interactions with Drugs__. It contains code for knowledge graphs (KG)construction, training and evaluation.

## Description

This project addresses the issue of drug interactions, particularly in the context of cancer treatment and the simulatenous use of pharmaceutical drugs, alternative medicines, and dietary supplements. Drug interactions can significantly impact treatment outcomes, making their understanding crucial in healthcare. The project focuses on utilizing knowledge graphs, popular in modeling complex biological systems, to predict potential drug interactions.

In this project we construct a complex knowledge graph encompassing pharmaceutical drugs, foods, herbs, chemical compounds, and known drug interactions. Knowledge graph embeddings are employed to predict novel interactions.

This repository contains all necessary code, starting from data processing, then knowledge graphs construction and training and then evaluation of novel drug predictions.


#### Methodology

The project involves constructing a comprehensive knowledge graph from various data sources, training knowledge graph embeddings (specifically ComplEx and RotatE), and predicting novel drug interactions. The databases utilized include DrugBank, FooDB, About Herbs (MSKCC), OGB - BioKG, and Hetionet.


## Getting Started

### Dependencies

The core of this project relies on the PyKEEN Python library for knowledge graph training. All required libraries are listed in `docker_dir/requirements.txt`. We also provide a `Dockerfile` for building a container with all necessary dependencies.

### How to Run

1. **Download Data:**
    - Download data from the specified sources and save it in the `data/<data_source>/` directory (e.g., `data/drugbank/`).
    
2. **Data Preprocessing:**
    - Preprocess the downloaded data using scripts in the `data_processing` folder. Generate *tsv* files with triplets *(head, relation, tail)* in the `triplets` directory.

3. **Knowledge Graph Construction:**
    - Construct a knowledge graph by creating train, validation, and test triplets using the `create_kg_data.sh` script.

4. **Training and Evaluation:**
    - Run `kg_model.py` to train and evaluate the knowledge graph. Example:
        ```bash
        ./kg_model.py -d interactions -m ComplEx -s test_run -emb 1000 -e 10 -o Adam -lr 0.0001
        ```

In the `data` folder, a *ready-to-use* sample of data for training and evaluation is provided.


<!-- -----
__experiments-hpo__ folder contains code for hyper-parameter optimization (HPO). It is devided into several experiments. Each experiment contains a __config.json__ file with tested hyperparameters. Also a __run_experiments.sh__ script is provided to run the HPO of an experiment. -->

## Results

In the `results` folder, we provide predictions obtained from the trained knowledge graphs. The predictions offer insights into potential drug interactions based on the employed models and datasets.


## Authors

* Markéta Křenková
    - a master student at Faculty of Informatics at Masaryk University Brno

