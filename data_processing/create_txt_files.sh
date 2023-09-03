#!/bin/sh

cd ../data/triplets/
cp train_$1.tsv train_$1.txt
cp valid_$1.tsv valid_$1.txt
cp test_$1.tsv test_$1.txt