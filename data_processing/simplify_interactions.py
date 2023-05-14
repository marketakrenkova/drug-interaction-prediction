#!/usr/bin/python3

import pandas as pd

data_dir = '../data/triplets/'

def main():
    
    ddi_df = pd.read_csv(data_dir + 'ddi.tsv', sep='\t', index_col=[0])
    dfi_df = pd.read_csv(data_dir + 'dfi_processed.tsv', sep='\t', index_col=[0])

if __name__ == "__main__":
    main() 