#!/usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
from os import listdir

def read_interactions_data(data_dir):
    ddi_df = pd.read_csv(data_dir + 'ddi.tsv', sep='\t', index_col=[0])
    drug_supplement_df = pd.read_csv(data_dir + 'ds_relations.tsv', sep='\t', index_col=[0])
    dfi_df = pd.read_csv(data_dir + 'dfi.tsv', sep='\t', index_col=[0])
    
    return ddi_df, drug_supplement_df, dfi_df

# compute sizes of train and test sets if the number of exaples is <= 7
def compute_size(n):
    if n == 2:
        return 1, 1
    if n == 3:
        return 1, 2
    if n == 4:
        return 2, 3
    if n == 5:
        return 3, 4
    if n == 6:
        return 4, 5
    # n == 7
    return 4, 6 

# train : valid : test = 80 : 10 : 10
def split_data_relation(df_relation):
    # too few triplets with the realtion
    if df_relation.shape[0] <= 7:
        train_size, valid_size = compute_size(df_relation.shape[0])
        df_relation = df_relation.sample(frac=1, random_state=42)  
        X_train = df_relation.iloc[:train_size]
        X_valid = df_relation.iloc[train_size:valid_size]
        X_test = df_relation.iloc[valid_size:]

    else:
        X_train, X_rem = train_test_split(df_relation, train_size=0.8, random_state=42)
        X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=42)
        
    return X_train, X_valid, X_test

# split drug-supplements relation dataset
def split_drug_supplements_dataset(drug_supplement_df):
    relations = set(drug_supplement_df.REL)

    print('Ds relation count:', len(relations))
    
    train_triplets = pd.DataFrame(columns=['CUI1', 'REL', 'CUI2'])
    valid_triplets = pd.DataFrame(columns=['CUI1', 'REL', 'CUI2'])
    test_triplets = pd.DataFrame(columns=['CUI1', 'REL', 'CUI2'])

    for rel in relations:
        train, valid, test = split_data_relation(drug_supplement_df[drug_supplement_df['REL'] == rel])
        train_triplets = pd.concat([train_triplets, train])
        valid_triplets = pd.concat([valid_triplets, valid])
        test_triplets = pd.concat([test_triplets, test])

    train_triplets = train_triplets.rename(columns={'CUI1': 'head', 'REL': 'relation', 'CUI2': 'tail'})
    valid_triplets = valid_triplets.rename(columns={'CUI1': 'head', 'REL': 'relation', 'CUI2': 'tail'})
    test_triplets = test_triplets.rename(columns={'CUI1': 'head', 'REL': 'relation', 'CUI2': 'tail'})    

    print('Drug Supplement database - drug-suplement interactions')
    print('train dataset size:', train_triplets.shape[0])
    print('validation dataset size:',valid_triplets.shape[0])
    print('test dataset size:',test_triplets.shape[0])
    
    return train_triplets, valid_triplets, test_triplets

# split drug-drug interaction dataset (from DrugBank)
def split_ddi_dataset(ddi_df):
    interactions = set(ddi_df.interaction)
    
    print('DDI/DFI interactions count:', len(interactions))
    
    train_triplets = pd.DataFrame(columns=['drug1', 'interaction', 'drug2'])
    valid_triplets = pd.DataFrame(columns=['drug1', 'interaction', 'drug2'])
    test_triplets = pd.DataFrame(columns=['drug1', 'interaction', 'drug2'])
    
    for inter in interactions:
        train, valid, test = split_data_relation(ddi_df[ddi_df['interaction'] == inter])
        train_triplets = pd.concat([train_triplets, train])
        valid_triplets = pd.concat([valid_triplets, valid])
        test_triplets = pd.concat([test_triplets, test])
        
    train_triplets = train_triplets.rename(columns={'drug1': 'head', 'interaction': 'relation', 'drug2': 'tail'})
    valid_triplets = valid_triplets.rename(columns={'drug1': 'head', 'interaction': 'relation', 'drug2': 'tail'})
    test_triplets = test_triplets.rename(columns={'drug1': 'head', 'interaction': 'relation', 'drug2': 'tail'})    

    print('DrugBank drug-drug interactions')
    print('train dataset size:', train_triplets.shape[0])
    print('validation dataset size:',valid_triplets.shape[0])
    print('test dataset size:',test_triplets.shape[0])
    
    return train_triplets, valid_triplets, test_triplets 

def split_interactions_data(ddi_df, drug_supplement_df, dfi_df, use_interaction_data):
    train_triplets, valid_triplets, test_triplets = split_ddi_dataset(ddi_df)

    if use_interaction_data[0]:    
        train_triplets_ds, valid_triplets_ds, test_triplets_ds = split_drug_supplements_dataset(drug_supplement_df)
        train_triplets = pd.concat([train_triplets, train_triplets_ds])
        valid_triplets = pd.concat([valid_triplets, valid_triplets_ds])
        test_triplets = pd.concat([test_triplets, test_triplets_ds])

    if use_interaction_data[1]:    
        train_triplets_dfi, valid_triplets_dfi, test_triplets_dfi = split_ddi_dataset(dfi_df)
        train_triplets = pd.concat([train_triplets, train_triplets_dfi])
        valid_triplets = pd.concat([valid_triplets, valid_triplets_dfi])
        test_triplets = pd.concat([test_triplets, test_triplets_dfi])

    print('All interactions:')
    print('train dataset size:', train_triplets.shape[0])
    print('validation dataset size:',valid_triplets.shape[0])
    print('test dataset size:',test_triplets.shape[0])
    
    return train_triplets, valid_triplets, test_triplets

def add_other_info_to_train(data_dir, train_triplets, use_interaction_data):
    files = listdir(data_dir)
    
    files2skip = ['ddi.tsv', 'dfi.tsv', 'ds_relations.tsv', 'ds_atoms_concept_map.tsv', 'ds_concept_type.tsv', '.ipynb_checkpoints', 'drug_atc_codes.tsv', 'drugs_inchi_key.tsv', 'salts_salts_inchi_key.tsv', 'ingredients.tsv']

    for file in files:
        if file in files2skip:
            continue
        # name -> don't add names of elements, just ids
        if 'train' in file or 'valid' in file or 'test' in file or 'name' in file:
            continue
        # use food and drug supplements in KG iff use_interaction_data[i]=True
        if not use_interaction_data[0] and file == 'ds_ingredients.tsv': # drug supplements
            continue
        if not use_interaction_data[1] and 'compound' in file: # foods
            continue

        df = pd.read_csv(data_dir + file, sep='\t', index_col=[0])

        df = df.set_axis(['head', 'relation', 'tail'], axis=1) 
        train_triplets = pd.concat([train_triplets, df])

    print('Final size of train dataset (with other relations):', train_triplets.shape[0]) 
    return train_triplets


def main():
    data_dir = '../data/triplets/'

    # which interactions use - drug-drug_supplement, drug-food
    # drug-drug is used always
    use_interactions_data = [False, True]
    
    ddi_df, drug_supplement_df, dfi_df = read_interactions_data(data_dir)
    train, valid, test = split_interactions_data(ddi_df, drug_supplement_df, dfi_df, use_interactions_data)
    train = add_other_info_to_train(data_dir, train, use_interactions_data)
    
    train = train.astype(str)
    valid = valid.astype(str)
    test = test.astype(str)
    
    train.to_csv(data_dir + 'train.tsv', sep='\t', index=False)
    valid.to_csv(data_dir + 'valid.tsv', sep='\t', index=False)
    test.to_csv(data_dir + 'test.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main() 
