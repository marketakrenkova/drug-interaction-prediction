#!/usr/bin/env python3

import pandas as pd

food_comouds_path = "../data/triplets/compounds_names.tsv"
herbs_path = "../data/triplets/herbs-di.tsv"


def main():
    food_compounds_df = pd.read_csv(food_comouds_path, delimiter='\t', index_col=[0])
    herbs_df = pd.read_csv(herbs_path, delimiter='\t', index_col=[0])
   
    food_comouds = set(food_compounds_df.compound_id)
    herbs = set(herbs_df.drug2)

    with open("../data/foods4predictions.txt", "w") as f:
        for food in food_comouds:
            f.write(food + '\n')
        
        for h in herbs:
            f.write(h + '\n')
        

if __name__ == "__main__":
    main()