#!/usr/bin/python3

import pandas as pd
import random

# corrupting triplets by changing a tail 
def generate_corupted_triplets(interactions, num):
    corrupted_triplets = []
    entities = set(interactions.drug1.unique())
    entities = list(entities.union(interactions.drug2.unique()))
    num_ent = len(entities)

    interactions = interactions.sample(frac=1)

    for _, interaction in interactions.iterrows():
        if len(corrupted_triplets) >= num:
            break

        rand_idx = random.randint(0, num_ent-1)
        new_tail = entities[rand_idx]
        new_triplet = (interaction['drug1'], interaction['interaction'], new_tail)

        # check if the created triplet exists -> if not, add it to the corrupted triplets
        check_df = interactions.loc[(interactions['drug1'] == new_triplet[0]) & (interactions['interaction'] == new_triplet[1]) & (interactions['drug2'] == new_triplet[2])]

        if check_df.shape[0] == 0:
            corrupted_triplets.append(new_triplet) 
    
    return corrupted_triplets

if __name__ == '__main__':
    # ddi, dfi
    data_dir = '../data/triplets/'
    ddi = pd.read_csv(data_dir + 'ddi.tsv', sep='/t', index_col=[0])
    dfi = pd.read_csv(data_dir + 'dfi.tsv', sep='/t', index_col=[0])

    negative_ddi = generate_corupted_triplets(ddi, 500)   
    negative_dfi = generate_corupted_triplets(dfi, 200)

    negative_triplets = pd.concat([negative_ddi, negative_dfi])
    negative_triplets.to_csv(data_dir + 'negative_triplets.tsv', spe='\t', index=False)
