#!/usr/bin/python3

import pandas as pd
import json

"""
Simple drug-food predictor: predicts possible drug-food interactions according to common compounds.
E.g. drug1 interacts_with drug2 -> find all foods with common compound as drug2 -> possible interaction between drug1 and the found foods

Necessary files (triplets in tsv file): drug-drug interactions, drug ingredients, food ingredients

Returns: pandas.DataFrame containing drug, type of interaction, food and the common compound
"""
class SimpleDFIPredictor():
    def __init__(self):
        self.ddi_path = '../data/triplets/ddi.tsv'
        self.drug_ingredients_path = '../data/triplets/ingredients.tsv'
        self.food_ingredients_path = '../data/triplets/food_compound.tsv'

        self.load_data()
        self.food_compound_dict = self.create_food_compound_dict()
        self.drug_compound_dict = self.create_drug_compound_dict()

    def load_data(self):
        self.ddi = pd.read_csv(self.ddi_path, sep='\t', index_col=[0])
        self.drug_ingredients = pd.read_csv(self.drug_ingredients_path, sep='\t', index_col=[0])
        self.food_ingredients = pd.read_csv(self.food_ingredients_path, sep='\t', index_col=[0])    

    # food_compound_dict = {compound : [list of foods which contain this compound]}
    def create_food_compound_dict(self):
        compounds = set(self.food_ingredients.compound)
        food_compound_dict = dict()

        for c in compounds:
            food_compound_dict[c] = list(self.food_ingredients[self.food_ingredients['compound'] == c].food)

        return food_compound_dict

    # drug_compound_dict = {drug : [list of compounds which contain this drug]}
    def create_drug_compound_dict(self):
        drug_compound_dict = dict()
        interacting_drugs = set(self.ddi.drug1)
        interacting_drugs.union(set(self.ddi.drug2))  # doesn't have to be drug, it can be a compound too

        for d in interacting_drugs:
            drug_compound_dict[d] = list(self.drug_ingredients[self.drug_ingredients['drug_name'] == d].ingredient)
         
        return drug_compound_dict

    # for each drug ingredient/compound (drug_ingredients), find a list of foods which contain this ingredient 
    # return dictionary {ingredient: [list of foods  which contain this ingredient]}
    def get_similar_food(self, drug_ingredients):
        sim_food = {}

        for ingredient in drug_ingredients:
            # print(ingredient)
            foods = self.food_compound_dict.get(ingredient)
            if foods is not None:
                sim_food[ingredient] = foods    

        if len(sim_food) < 1:
            return {}

        return sim_food          

    def predict_dfi(self):     
        inspected_drugs = []
        possible_interactions = []

        for row in self.ddi.itertuples():
            drug1 = row[1]
            effect = row[2]
            drug2 = row[3]

            if drug2 in inspected_drugs or drug1 in inspected_drugs:
                continue

            inspected_drugs.append(drug2)
        
            drug2_ingredients = self.drug_compound_dict.get(drug2)
            drug1_ingredients = self.drug_compound_dict.get(drug1)
            
            if drug2_ingredients is not None:
                sim_food = self.get_similar_food(drug2_ingredients)
                if len(sim_food) > 0 and len(sim_food) <= 20:  # if sim food > 20 -> probably no interaction
                    for compound, food in sim_food.items():
                        for f in food:
                            possible_interactions.append([drug1, effect, f, compound])

            if drug1_ingredients is not None:
                sim_food = self.get_similar_food(drug1_ingredients)
                if len(sim_food) > 0 and len(sim_food) <= 20:  # if sim food > 20 -> probably no interaction
                    for compound, food in sim_food.items():
                        for f in food:
                            possible_interactions.append([drug2, effect, f, compound])

        self.possible_dfi  = pd.DataFrame(possible_interactions, columns=['drug', 'interaction', 'food', 'common_compound'])
        
    def save(self, output_file):
        df = self.possible_dfi.sort_values(by=['drug', 'food'])
        df.to_csv(output_file)


if __name__ == '__main__':
    dfi_predictor = SimpleDFIPredictor()
    dfi_predictor.predict_dfi()
    dfi_predictor.save('../data/possible_dfi.csv')

    print(dfi_predictor.possible_dfi.head())
