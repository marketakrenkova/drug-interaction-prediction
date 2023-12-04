import pandas as pd

# map drug names to just ingredients in possible-dfi.csv

# How to run: python3 map_ingretients2drugs.py

possible_dfi = pd.read_csv('possible_dfi_simple_dfi_predictor.csv')
drug_ingredients = pd.read_csv('../data/triplets/ingredients.tsv', sep='\t', index_col=[0])

drug_id_name_map = pd.read_csv('../data/drugbank/drug_id_name_map.csv', sep=',') 
drug_id_name_map_dict = dict(zip(drug_id_name_map.drug_name, drug_id_name_map.id))


ingredients = list(set(drug_ingredients['ingredient']))
ingredients = [x.strip() for x in ingredients]

corrected_drugs = []
ingredient_drugs_map = dict()

for row in possible_dfi.iterrows():
    drug = row[1]['drug']

    if drug in corrected_drugs:
        continue

    corrected_drugs.append(drug)
    
    drug_id = drug_id_name_map_dict.get(drug)

    if drug_id is not None and drug_id in ingredients:
        drugs = list(drug_ingredients[drug_ingredients['ingredient'] == drug_id]['drug_name'].values)
        if len(drugs) > 0:
            ingredient_drugs_map[drug] = drugs
        # print(drug, ":", len(drugs))

with open('possible_dfi-ingredients_drug_map.txt', 'w') as out:
    for ingredient, drugs in ingredient_drugs_map.items():
        out.write(ingredient + ":\n")
        for i, d in enumerate(drugs):
            out.write(d)
            if i == len(drugs) - 1:
                out.write("\n")
            else:
                out.write(", ")    
        out.write("\n")
