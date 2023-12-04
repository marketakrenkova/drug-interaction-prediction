#!/usr/bin/env python3

"""
This script prepares files with triplets in form (head, relation, tail) that are essential for knowledge graphs.
These files are subsequently used in create_train_test_triplets.py

Each file contains one type of relation, e.g. (drug, in_category, category).

How to run: ./prepare_triplets.py
"""

import pandas as pd
import itertools
import json

triplets_dir = "../data/triplets/"
drug_dir = "../data/drugbank/"
food_dir = "../data/fooDB/"


# ---------------------------------------------------------------------------------
# DrugBank
def prepare_triplets_drugbank():
    # drug_id - name
    drugs_map = pd.read_csv(drug_dir + "drug_id_name_map.csv", index_col=[0])

    # drug name - id map
    drug_ids = drugs_map.id
    drug_names = drugs_map.drug_name
    drug_id_map_dict = dict(zip(drug_names, drug_ids))
    # ---------------------------------------------------------------------------------

    # drug_id - subclass
    drug_class = pd.read_csv(drug_dir + "drug_classification.csv", index_col=[0])
    drug_subclass = drug_class[["id", "subclass"]]
    drug_subclass["relation"] = list(
        itertools.repeat("in_subclass", drug_subclass.shape[0])
    )
    drug_subclass = drug_subclass.iloc[:, [0, 2, 1]]
    drug_subclass = drug_subclass.dropna()

    # create subclass mapping
    classes = drug_subclass["subclass"].unique()
    class_map = {c: "class_" + str(i) for i, c in enumerate(classes)}
    drug_subclass["subclass"] = drug_subclass["subclass"].map(class_map)

    with open(drug_dir + "drug_class_map.json", "w") as f:
        json.dump(class_map, f)

    print("Number of drug subclasses:", len(classes))
    print(drug_subclass.head())
    print()
    drug_subclass.to_csv(triplets_dir + "drug_subclass.tsv", sep="\t")

    # ---------------------------------------------------------------------------------
    # drug category
    drug_category = pd.read_csv(drug_dir + "drug_category.csv", index_col=[0])
    drug_category["relation"] = list(
        itertools.repeat("in_category", drug_category.shape[0])
    )
    drug_category = drug_category.iloc[:, [0, 2, 1]]
    drug_category = drug_category.dropna()

    print("Number of drug categories:", len(drug_category["category"].unique()))
    print(drug_category.head())
    drug_category.to_csv(triplets_dir + "drug_category.tsv", sep="\t")

    # ---------------------------------------------------------------------------------
    # drug_name - ingredients
    mixtures = pd.read_csv(
        drug_dir + "drug_mixtures.csv", index_col=[0]
    )  # TODO: inspect data (ingredients - drugbank ID)
    drugs = []
    ingredients = []
    for row in mixtures.itertuples():
        ingreds = row[2].split("+")
        for ingredient in ingreds:
            drugs.append(row[1].lower())
            ingredients.append(ingredient)

    # mapping ingredient names to its ids
    ingredients_s = pd.Series(ingredients)
    ingredients_ids = ingredients_s.map(drug_id_map_dict)

    relations = list(itertools.repeat("contains", len(drugs)))
    ingredient_df = pd.DataFrame(
        {"drug_name": drugs, "relation": relations, "ingredient": ingredients_ids}
    )
    ingredient_df = ingredient_df.dropna()
    ingredient_df = ingredient_df.drop_duplicates()
    print(ingredient_df.head())
    print()
    ingredient_df.to_csv(triplets_dir + "ingredients.tsv", sep="\t")
    # ---------------------------------------------------------------------------------
    # drug_id - inchi_key
    inchi_key = pd.read_csv(drug_dir + "drug_inchi_key.csv", index_col=[0])
    inchi_key["relation"] = list(itertools.repeat("has_inchi_key", inchi_key.shape[0]))
    inchi_key = inchi_key.iloc[:, [0, 2, 1]]
    inchi_key = inchi_key.dropna()
    print(inchi_key.head())
    print()
    inchi_key.to_csv(triplets_dir + "drugs_inchi_key.tsv", sep="\t")
    # ---------------------------------------------------------------------------------
    # drug_id - ATC code
    atc_codes = pd.read_csv(drug_dir + "drug_atc_code.csv", index_col=[0])
    atc_codes["relation"] = list(itertools.repeat("has_atc_code", atc_codes.shape[0]))
    atc_codes = atc_codes.iloc[:, [0, 2, 1]]
    atc_codes = atc_codes.dropna()
    print(atc_codes.head())
    print()
    atc_codes.to_csv(triplets_dir + "drug_atc_codes.tsv", sep="\t")

    # ---------------------------------------------------------------------------------

    # salt_id - salt name
    salts = pd.read_csv(drug_dir + "drug_salts.csv", index_col=[0])
    salts_name = salts[["id", "name"]]

    salts_names_dic = salts_name.set_index("id")["name"].to_dict()

    # drug_id - salt_id
    drug_salts = salts[["id", "drug"]].dropna()
    drug_salts["relation"] = list(itertools.repeat("contains", drug_salts.shape[0]))
    drug_salts = drug_salts.iloc[:, [1, 2, 0]]
    drug_salts.rename(columns={"id": "salt_id"}, inplace=True)
    print(drug_salts.head())
    print()
    drug_salts.to_csv(triplets_dir + "drug_salts.tsv", sep="\t")
    # ---------------------------------------------------------------------------------

    # salt_id - cas_number
    salts_cas_numebr = salts[["id", "cas_number"]].dropna()
    salts_cas_numebr["relation"] = list(
        itertools.repeat("has_cas_number", salts_cas_numebr.shape[0])
    )
    salts_cas_numebr = salts_cas_numebr.iloc[:, [0, 2, 1]]
    salts_cas_numebr.rename(columns={"id": "salt_id"}, inplace=True)
    print(salts_cas_numebr.head())
    print()
    salts_cas_numebr.to_csv(triplets_dir + "salts_cas_numebr.tsv", sep="\t")
    # ---------------------------------------------------------------------------------

    # salt_id - inchi_key
    salts_inchi = salts[["id", "inchi_key"]].dropna()
    salts_inchi["relation"] = list(
        itertools.repeat("has_inchi_key", salts_inchi.shape[0])
    )
    salts_inchi = salts_inchi.iloc[:, [0, 2, 1]]
    salts_inchi.rename(columns={"id": "salt_id"}, inplace=True)
    print(salts_inchi.head())
    print()
    salts_inchi.to_csv(triplets_dir + "salts_salts_inchi_key.tsv", sep="\t")
    # ---------------------------------------------------------------------------------

    # pathways
    pathway_df = pd.read_csv(drug_dir + "pathways.csv", index_col=[0])
    pathway_id_name = pathway_df[["smpdb_id", "pathway_name"]].dropna()
    pathway_id_name_dic = pathway_id_name.set_index("smpdb_id")[
        "pathway_name"
    ].to_dict()

    # pathway - drug
    pathway_drug = pathway_df[["smpdb_id", "drug_id"]].dropna()
    pathway_drug["relation"] = list(
        itertools.repeat("involved_in_pathway", pathway_drug.shape[0])
    )
    pathway_drug = pathway_drug.iloc[:, [0, 2, 1]]
    print(pathway_drug.head())
    print()
    pathway_drug.to_csv(triplets_dir + "pathway_drug.tsv", sep="\t")

    # pathway - category
    pathway_category = pathway_df[["smpdb_id", "category"]].dropna()
    pathway_category["relation"] = list(
        itertools.repeat("is_category", pathway_category.shape[0])
    )
    pathway_category = pathway_category.iloc[:, [0, 2, 1]]

    # create pathway_categor mapping
    categories = pathway_category["category"].unique()
    pathway_cat_map = {c: "pathway_cat_" + str(i) for i, c in enumerate(categories)}
    pathway_category["category"] = pathway_category["category"].map(pathway_cat_map)

    with open(drug_dir + "pathway_category__map.json", "w") as f:
        json.dump(pathway_cat_map, f)

    print(pathway_category.head())
    print()
    pathway_category.to_csv(triplets_dir + "pathway_category.tsv", sep="\t")

    # pathway - enzymes
    pathway_enzyme_df = pd.read_csv(drug_dir + "pathways_enzym.csv", index_col=[0])
    pathway_enzyme = pathway_enzyme_df[["smpdb_id", "enzyme_id"]]
    pathway_enzyme["relation"] = list(
        itertools.repeat("involved_in_pathway", pathway_enzyme.shape[0])
    )
    pathway_enzyme = pathway_enzyme.iloc[:, [0, 2, 1]]
    print(pathway_enzyme.head())
    print()
    pathway_enzyme.to_csv(triplets_dir + "pathway_enzyme.tsv", sep="\t")


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# FooDB
def prepare_triplets_foodb():
    # food_id - name
    food = pd.read_csv(food_dir + "food.csv")
    food_df = food[["public_id", "name"]].dropna()
    food_df["relation"] = list(itertools.repeat("has_name", food_df.shape[0]))
    food_df = food_df.iloc[:, [0, 2, 1]]

    print("Number of food:", food_df.shape[0])
    print(food_df.head())
    print()
    food_df.to_csv(triplets_dir + "food_name.tsv", sep="\t")

    food_names_dic = food_df.set_index("public_id")["name"].to_dict()

    with open(food_dir + "most_contributing_food_compounds.json", "r") as f:
        food_compounds = json.load(f)

    food_ids = list(food_df["public_id"].values)
    compound_public_ids = []
    compound_names = set()
    compound_cas_numbers = set()

    for food_id in food_ids:
        food_info = food_compounds[str(food_id)]
        for compound in food_info:
            # food_id - compound public_id
            compound_public_ids.append((food_id, "contains", compound["public_id"]))
            # compound publid_id - compound name
            compound_names.add((compound["public_id"], "has_name", compound["name"]))
            # compound publid_id - cas_number
            compound_cas_numbers.add(
                (compound["public_id"], "has_cas_number", compound["cas_number"])
            )

    food_compound = pd.DataFrame(
        compound_public_ids, columns=["food_id", "relation", "compound_id"]
    )
    compounds_names_df = pd.DataFrame(
        compound_names, columns=["compound_id", "relation", "name"]
    )
    compound_cas_numbers_df = pd.DataFrame(
        compound_cas_numbers, columns=["compound_id", "relation", "cas_number"]
    )

    food_compound["food_id"] = food_compound["food_id"].astype("string")

    compounds_names_dic = compounds_names_df.set_index("compound_id")["name"].to_dict()

    food_compound = food_compound.dropna()
    compounds_names_df = compounds_names_df.dropna()
    compound_cas_numbers_df = compound_cas_numbers_df.dropna()

    print("Number of food compounds:", food_compound.shape[0])
    print(food_compound.head())
    print()
    print(compounds_names_df.head())
    print()
    print(compound_cas_numbers_df.head())

    food_compound.to_csv(triplets_dir + "food_compound.tsv", sep="\t")
    compounds_names_df.to_csv(triplets_dir + "compounds_names.tsv", sep="\t")
    compound_cas_numbers_df.to_csv(triplets_dir + "compounds_cas_number.tsv", sep="\t")


# ---------------------------------------------------------------------------------


prepare_triplets_drugbank()
prepare_triplets_foodb()
