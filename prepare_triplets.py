#!/usr/bin/env python3

import pandas as pd
import itertools
import json
# from parser import parse_drug_interactions

# DrugBank
drug_dir = 'data/drugbank/'
triplets_dir = 'data/triplets/'

# drug_id - name
drugs = pd.read_csv(drug_dir + 'drug_id_name_map.csv', index_col=[0])
drugs['relation'] = list(itertools.repeat('has_name', drugs.shape[0]))
drugs = drugs.iloc[:,[0,2,1]]
print(drugs.head())
print()
drugs.to_csv(triplets_dir + 'drugs_names.tsv', sep='\t')

# category - mesh_id ?

# drug_id - subclass
drug_class = pd.read_csv(drug_dir + 'drug_classification.csv', index_col=[0])
drug_subclass = drug_class[['id', 'subclass']]
drug_subclass['relation'] = list(itertools.repeat('in_subclass', drug_subclass.shape[0]))
drug_subclass = drug_subclass.iloc[:,[0,2,1]]
print(drug_subclass.head())
print()
drug_subclass.to_csv(triplets_dir + 'drug_subclass.tsv', sep='\t')

# subclass - class ?
# TODO: to lower case
# sub_class = drug_class[['class', 'subclass']].drop_duplicates()
# sub_class['relation'] = list(itertools.repeat('in_class', sub_class.shape[0]))
# sub_class = sub_class.iloc[:,[0,2,1]]
# print(sub_class.head())

# class - superclass ?
# superclass - kinkdom ?

# drug_id - interaction - drug_id 
# parse_drug_interactions()

# drug_id - interaction - food
# TODO: parser 

# drug_name - ingredients
mixtures = pd.read_csv(drug_dir + 'drug_mixtures.csv', index_col=[0])
drugs = []
ingredients = []
relations = []
for drug in mixtures.itertuples():
    ingreds = drug[2].split('+')
    for ingredient in ingreds:
        drugs.append(drug[1])
        relations.append('contains')
        ingredients.append(ingredient)
ingredient_df = pd.DataFrame({'drug_name': drugs, 'relation': relations, 'ingredient': ingredients})        
print(ingredient_df.head())   
print()    
ingredient_df.to_csv(triplets_dir + 'ingredients.tsv', sep='\t')

# drug_id - inchi_key
inchi_key = pd.read_csv(drug_dir + 'drug_inchi_key.csv', index_col=[0])
inchi_key['relation'] = list(itertools.repeat('has_inchi_key', inchi_key.shape[0]))
inchi_key = inchi_key.iloc[:,[0,2,1]]
print(inchi_key.head())
print()
inchi_key.to_csv(triplets_dir + 'drugs_inchi_key.tsv', sep='\t')

# drug_id - molecule
molecules = pd.read_csv(drug_dir + 'drug_molecul.csv', index_col=[0])
molecules['relation'] = list(itertools.repeat('has_molecule', molecules.shape[0]))
molecules = molecules.iloc[:,[0,2,1]]
print(molecules.head())
print()
molecules.to_csv(triplets_dir + 'drugs_molecule.tsv', sep='\t')


# salt_id - salt name
salts = pd.read_csv(drug_dir + 'drug_salts.csv', index_col=[0])
salts_name = salts[['id', 'name']]
salts_name['relation'] = list(itertools.repeat('has_name', salts_name.shape[0]))
salts_name = salts_name.iloc[:,[0,2,1]]
print(salts_name.head())
print()
salts_name.to_csv(triplets_dir + 'salts_name.tsv', sep='\t')

# drug_id - salt_id
drug_slats = salts[['id', 'drug']]
drug_slats['relation'] = list(itertools.repeat('contains_salt', drug_slats.shape[0]))
drug_slats = drug_slats.iloc[:,[1,2,0]]
print(drug_slats.head())
print()
drug_slats.to_csv(triplets_dir + 'drug_slats.tsv', sep='\t')

# salt_id - cas_number
salts_cas_numebr = salts[['id', 'cas_number']].dropna()
salts_cas_numebr['relation'] = list(itertools.repeat('has_cas_number', salts_cas_numebr.shape[0]))
salts_cas_numebr = salts_cas_numebr.iloc[:,[0,2,1]]
print(salts_cas_numebr.head())
print()
salts_cas_numebr.to_csv(triplets_dir + 'salts_cas_numebr.tsv', sep='\t')

# salt_id - inchi_key
salts_inchi = salts[['id', 'inchi_key']].dropna()
salts_inchi['relation'] = list(itertools.repeat('has_inchi_key', salts_inchi.shape[0]))
salts_inchi = salts_inchi.iloc[:,[0,2,1]]
print(salts_inchi.head())
print()
salts_inchi.to_csv(triplets_dir + 'salts_salts_inchi_key.tsv', sep='\t')

# -----------------------------------------------------------
# FooDB

# compound publid_id - source_id ?
# compound publid_id - contribution score

# food_id - name
food = pd.read_csv('../data/food.csv')
food_df = food[['id', 'name']]
food_df['relation'] = list(itertools.repeat('has_name', food_df.shape[0]))
food_df = food_df.iloc[:,[0,2,1]]
print(food_df.head())
print()
food_df.to_csv(triplets_dir + 'food_name.tsv', sep='\t')

with open('../data/most_contributing_food_compounds.json', 'r') as f:
    food_compounds = json.load(f)

food_ids = list(food_df.id.values)    
compound_public_ids = []
compound_names = set()
compound_cas_numbers = set()
# food_compound_contibution_scores = [] # ??

for food_id in food_ids:
    food_info = food_compounds[str(food_id)] 
    for compound in food_info:
        # food_id - compound publid_id
        compound_public_ids.append((food_id, 'has_compound', compound['public_id']))
        # compound publid_id - compound name
        compound_names.add((compound['public_id'], 'has_name', compound['name']))
        # compound publid_id - cas_number
        compound_cas_numbers.add((compound['public_id'], 'has_cas_number', compound['cas_number']))

food_compound = pd.DataFrame(compound_public_ids, columns=['food_id', 'relation', 'compound_id'])
compounds_names_df = pd.DataFrame(compound_names, columns=['compound_id', 'relation', 'name'])
compound_cas_numbers_df = pd.DataFrame(compound_cas_numbers, columns=['compound_id', 'relation', 'cas_number'])

print(food_compound.head())
print()
print(compounds_names_df.head())
print()
print(compound_cas_numbers_df.head())

food_compound.to_csv(triplets_dir + 'food_compound.tsv', sep='\t') 
compounds_names_df.to_csv(triplets_dir + 'compounds_names.tsv', sep='\t') 
compound_cas_numbers_df.to_csv(triplets_dir + 'compounds_cas_number.tsv', sep='\t') 

# -----------------------------------------------------------
# iDISK
ds_data_dir = '../data/idisk-rrf/'

# concept id - type
concept_type = pd.read_csv(ds_data_dir + 'MRSTY.csv', sep='|')
concept_type['relation'] = list(itertools.repeat('is_type', concept_type.shape[0]))
concept_type = concept_type.iloc[:,[0,2,1]]
print(concept_type.head())
print()
concept_type.to_csv(triplets_dir + 'ds_concept_type.tsv', sep='\t')

# atom id - concept id
atoms = pd.read_csv(ds_data_dir + 'MRCONSO.csv', sep='|')
atoms_concept = atoms[['CUI', 'AUI']]
atoms_concept['relation'] = list(itertools.repeat('maps_to', atoms_concept.shape[0]))
atoms_concept = atoms_concept.iloc[:,[0,2,1]]
print(atoms_concept.head())
print()
atoms_concept.to_csv(triplets_dir + 'ds_atoms_concept_map.tsv', sep='\t')

# atom id -name
atoms_name = atoms[['AUI', 'STR']]
atoms_name['relation'] = list(itertools.repeat('has_name', atoms_name.shape[0]))
atoms_name = atoms_name.iloc[:,[0,2,1]]
print(atoms_name.head())
print()
atoms_name.to_csv(triplets_dir + 'ds_atoms_names.tsv', sep='\t')

# atom id - source db ?
# atom id - id in source db ?

# TODO: parser
# attribute id - attribute name
# attributes = pd.read_csv(ds_data_dir + 'MRSAT.csv', sep='|')
# attr_name = attributes[['ATUI', 'ATN']]
# attr_name['relation'] = list(itertools.repeat('has_name', attr_name.shape[0]))
# attr_name = attr_name.iloc[:,[0,2,1]]
# print(attr_name.head())
# print()
# attr_name.to_csv(triplets_dir + 'ds_attribute_names.tsv', sep='\t')

# # attribute id - type (concept/relation) ?

# # attribute id - concept/relation id
# attr_id_map = attributes[['ATUI', 'UI']]
# attr_id_map['relation'] = list(itertools.repeat('maps_to', attr_id_map.shape[0]))
# attr_id_map = attr_id_map.iloc[:,[0,2,1]]
# print(attr_id_map.head())
# print()
# attr_id_map.to_csv(triplets_dir + 'ds_attr_id_map.tsv', sep='\t')

# attribute id - value (decription)
# attr_val = attributes[['ATUI', 'UI']]
# attr_val['relation'] = list(itertools.repeat('descritpion', attr_val.shape[0]))
# attr_val = attr_val.iloc[:,[0,2,1]]
# print(attr_val.head())
# print()
# attr_val.to_csv(triplets_dir + 'ds_attr_description.tsv', sep='\t')

# relation: concept1 id - relation id - concept2 id
relations = pd.read_csv(ds_data_dir + 'MRREL.csv', sep='|')
relations = relations[['CUI1', 'REL', 'CUI2']]
print(relations.head())
relations.to_csv(triplets_dir + 'ds_relations,tsv', sep='\t')

# relation id - description (TODO: create relation id + description map)