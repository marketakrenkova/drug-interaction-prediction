#!/bin/sh

cd ../data/triplets/
cp train_interactions.tsv train_interactions.txt
cp valid_interactions.tsv valid_interactions.txt
cp test_interactions.tsv test_interactions.txt

cp train_drugbank.tsv train_drugbank.txt
cp valid_drugbank.tsv valid_drugbank.txt
cp test_drugbank.tsv test_drugbank.txt

cp train_biokg.tsv train_biokg.txt
cp valid_biokg.tsv valid_biokg.txt
cp test_biokg.tsv test_biokg.txt

cp train_hetionet.tsv train_hetionet.txt
cp valid_hetionet.tsv valid_hetionet.txt
cp test_hetionet.tsv test_hetionet.txt
