#!/usr/bin/python3

"""
This script combines all data necessary for knowledge graphs construction - creates train, validation and test triplets.
A KG from interactions, drugbank, biokg or hetionet can be created.
It supports cross-validation - creates different splits of the data (run1, run2, ...) 

Run seperately, e.g.: ./create_train-test_data.py interactions run1 creates KG only from drug interactions data.
Or run using create_kg_data.sh 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from os import listdir, path, makedirs
import itertools
import sys

seen_interactions = set()


def read_interactions_data(data_dir):
    ddi_df = pd.read_csv(data_dir + "ddi.tsv", sep="\t", index_col=[0])
    dfi_df = pd.read_csv(data_dir + "dfi_processed.tsv", sep="\t", index_col=[0])
    herbs_df = pd.read_csv(data_dir + "herbs-di.tsv", sep="\t", index_col=[0])

    return ddi_df, dfi_df, herbs_df


# all types of interactions are transformed to two types - 'positive'/'negative'
def simplify_interactions(interaction_df, interaction_label_file):
    labels = pd.read_csv(interaction_label_file, sep=";")
    label_map = labels.set_index("relation").to_dict()["positive/negative"]
    interaction_df.interaction = interaction_df.interaction.map(label_map)

    return interaction_df


# Function to check and update the set
def check_and_update_set(row):
    pair = frozenset([row["drug1"], row["drug2"]])
    if pair not in seen_interactions:
        seen_interactions.add(pair)
        return True
    return False


# all types of interactions are transformed to one type = 'interacts'
def simplify_interactions2(interaction_df):
    interaction_df.interaction = list(
        itertools.repeat("interacts", interaction_df.shape[0])
    )

    # Apply the function to filter duplicate (also inversed) triplets
    interaction_df = interaction_df[interaction_df.apply(check_and_update_set, axis=1)]

    return interaction_df


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
def split_data_relation(df_relation, rs):
    # too few triplets with the realtion
    if df_relation.shape[0] <= 7:
        train_size, valid_size = compute_size(df_relation.shape[0])
        df_relation = df_relation.sample(frac=1, random_state=rs)
        X_train = df_relation.iloc[:train_size]
        X_valid = df_relation.iloc[train_size:valid_size]
        X_test = df_relation.iloc[valid_size:]

    else:
        X_train, X_rem = train_test_split(df_relation, train_size=0.8, random_state=rs)
        X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=rs)

    return X_train, X_valid, X_test


# split drug interactions (from DrugBank)
def split_ddi_dataset(ddi_df, rs):
    interactions = set(ddi_df.interaction)

    print("DDI/DFI interactions count:", len(interactions))

    train_triplets = pd.DataFrame(columns=["drug1", "interaction", "drug2"])
    valid_triplets = pd.DataFrame(columns=["drug1", "interaction", "drug2"])
    test_triplets = pd.DataFrame(columns=["drug1", "interaction", "drug2"])

    for inter in interactions:
        train, valid, test = split_data_relation(
            ddi_df[ddi_df["interaction"] == inter], rs
        )
        train_triplets = pd.concat([train_triplets, train])
        valid_triplets = pd.concat([valid_triplets, valid])
        test_triplets = pd.concat([test_triplets, test])

    train_triplets = train_triplets.rename(
        columns={"drug1": "head", "interaction": "relation", "drug2": "tail"}
    )
    valid_triplets = valid_triplets.rename(
        columns={"drug1": "head", "interaction": "relation", "drug2": "tail"}
    )
    test_triplets = test_triplets.rename(
        columns={"drug1": "head", "interaction": "relation", "drug2": "tail"}
    )

    print("DrugBank drug-drug interactions")
    print("train dataset size:", train_triplets.shape[0])
    print("validation dataset size:", valid_triplets.shape[0])
    print("test dataset size:", test_triplets.shape[0])

    return train_triplets, valid_triplets, test_triplets


def split_interactions_data(
    ddi_df, dfi_df, herbs_df, use_food_interaction, random_seed
):
    train_triplets, valid_triplets, test_triplets = split_ddi_dataset(
        ddi_df, random_seed
    )

    if use_food_interaction:
        train_triplets_dfi, valid_triplets_dfi, test_triplets_dfi = split_ddi_dataset(
            dfi_df, random_seed
        )
        (
            train_triplets_herbs,
            valid_triplets_herbs,
            test_triplets_herbs,
        ) = split_ddi_dataset(herbs_df, random_seed)
        train_triplets = pd.concat(
            [train_triplets, train_triplets_dfi, train_triplets_herbs]
        )
        valid_triplets = pd.concat(
            [valid_triplets, valid_triplets_dfi, valid_triplets_herbs]
        )
        test_triplets = pd.concat(
            [test_triplets, test_triplets_dfi, test_triplets_herbs]
        )

    print("All interactions:")
    print("train dataset size:", train_triplets.shape[0])
    print("validation dataset size:", valid_triplets.shape[0])
    print("test dataset size:", test_triplets.shape[0])

    return train_triplets, valid_triplets, test_triplets


def add_other_info_to_train(
    data_dir, train_triplets, use_food_interaction, specification
):
    files = listdir(data_dir)

    files2skip = [
        "ddi.tsv",
        "dfi.tsv",
        "herbs-di.tsv",
        "herbs-di-old.tsv",
        "dfi_processed.tsv",
        ".ipynb_checkpoints",
        "drug_atc_codes.tsv",
        "drugs_inchi_key.tsv",
        "salts_salts_inchi_key.tsv",
        "ingredients.tsv",
    ]

    for file in files:
        # skip defined files
        if file in files2skip:
            continue
        # name -> don't add names of elements, just ids
        if (
            "train" in file
            or "valid" in file
            or "test" in file
            or "name" in file
            or "run" in file
        ):
            continue

        if not use_food_interaction and "compound" in file:
            continue

        # additinal data sources
        if file == "biokg_subgraph.tsv" and specification != "biokg":
            continue

        if file == "hetionet.tsv" and specification != "hetionet":
            continue

        # DrugBank interations only
        if specification == "interactions":
            if (
                "pathway" in file
                or "salt" in file
                or "subclass" in file
                or "category" in file
                or "compound" in file
            ):
                continue

        df = pd.read_csv(data_dir + file, sep="\t", index_col=[0])

        df = df.set_axis(["head", "relation", "tail"], axis=1)
        train_triplets = pd.concat([train_triplets, df])

    print(
        "Final size of train dataset (with other relations):", train_triplets.shape[0]
    )

    return train_triplets


def main(name, run):
    data_dir = "../data/triplets/"
    # interaction_label_file = '../data/unique_relations-labeled.csv'

    # use drug-food interactions
    use_food_interaction = True

    ddi_df, dfi_df, herbs_df = read_interactions_data(data_dir)

    # simplify interaction names -> just positive/negative interactions
    # ddi_df_simple = simplify_interactions(ddi_df, interaction_label_file)
    # dfi_df_simple = simplify_interactions(dfi_df, interaction_label_file)

    # simplify interaction names -> just interaction
    ddi_df_simple = simplify_interactions2(ddi_df)
    dfi_df_simple = simplify_interactions2(dfi_df)

    random_seed = int(run[3])
    if not isinstance(random_seed, int):
        random_seed = 42

    train, valid, test = split_interactions_data(
        ddi_df_simple, dfi_df_simple, herbs_df, use_food_interaction, random_seed
    )
    train = add_other_info_to_train(data_dir, train, use_food_interaction, name)

    print(
        "Size of the whole KG (num triplets):",
        train.shape[0] + valid.shape[0] + test.shape[0],
    )
    train = train.astype(str)
    valid = valid.astype(str)
    test = test.astype(str)

    run += "/"

    if not path.exists(data_dir + run):
        makedirs(data_dir + run)

    train.to_csv(data_dir + run + "train_" + name + ".tsv", sep="\t", index=False)
    valid.to_csv(data_dir + run + "valid_" + name + ".tsv", sep="\t", index=False)
    test.to_csv(data_dir + run + "test_" + name + ".tsv", sep="\t", index=False)


# name: interactions, drugbank, biokg, hetionet
# run: run1, run2, ...
if __name__ == "__main__":
    name = ""
    run = "run1"

    if len(sys.argv) > 1:
        name = sys.argv[1]
        run = sys.argv[2]

    main(name, run)
