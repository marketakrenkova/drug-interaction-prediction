#!/usr/bin/env python3

# some code from:
# https://github.com/dsi-bdi/biokg
# biokg/biokg/processing/parsers.py 

import re
import pandas as pd

# drug interactions
DDI_SIDE_EFFECT_1 = re.compile('The risk or severity of (?P<se>.*) can be (?P<mode>\S+)d when (?P<drug1>\w*\s*\w*) is combined with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_2 = re.compile('(?P<drug1>\w*\s*\w*) may (?P<mode>\S+) (?P<se>\S+\s?\w*\s?\w*) of (?P<drug2>\w*\s*\w*) as a diagnostic agent.')
DDI_SIDE_EFFECT_3 = re.compile('The (?P<se>\S+\s?\w*\s?\w*) of (?P<drug1>\w*\s*\w*) can be (?P<mode>\S+)d when used in combination with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_4 = re.compile('The (?P<se>\S+\s?\w*\s?\w*) of (?P<drug1>\w*\s*\w*) can be (?P<mode>\S+)d when it is combined with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_5 = re.compile('(?P<drug1>\w*\s*\w*) can cause a decrease in the absorption of (?P<drug2>\w*\s*\w*) resulting in a (?P<mode>\S+) (?P<se>\S+\s?\w*\s?\w*) and potentially a decrease in efficacy.')
DDI_SIDE_EFFECT_6 = re.compile('(?P<drug1>\w*\s*\w*) may decrease the excretion rate of (?P<drug2>\w*\s*\w*) which could result in a (?P<mode>\S+) (?P<se>\S+\s?\w*\s?\w*).')
DDI_SIDE_EFFECT_7 = re.compile('(?P<drug1>\w*\s*\w*) may increase the excretion rate of (?P<drug2>\w*\s*\w*) which could result in a (?P<mode>\S+) (?P<se>\S+\s?\w*\s?\w*) and potentially a reduction in efficacy.')
DDI_SIDE_EFFECT_8 = re.compile('The (?P<se>\S+\s?\w*\s?\w*) of (?P<drug1>\w*\s*\w*) can be (?P<mode>\S+)d when combined with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_9 = re.compile('(?P<drug1>\w*\s*\w*) can cause an increase in the absorption of (?P<drug2>\w*\s*\w*) resulting in an (?P<mode>\S+)d (?P<se>\S+\s?\w*\s?\w*) and potentially a worsening of adverse effects.')
DDI_SIDE_EFFECT_10 = re.compile('The risk of a (?P<se>\S+\s?\w*\s?\w*) to (?P<drug1>\w*\s*\w*) is (?P<mode>\S+)d when it is combined with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_11 = re.compile('The (?P<se>\S+\s?\w*\s?\w*) of (?P<drug1>\w*\s*\w*) can be (?P<mode>\S+)d when combined with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_12 = re.compile('The (?P<se>\S+\s?\w*\s?\w*) of the active metabolites of .* can be (?P<mode>\S+)d when (?P<drug1>\w*\s*\w*) is used in combination with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_13 = re.compile('The (?P<se>\S+\s?\w*\s?\w*) of (?P<drug1>\w*\s*\w*), an active metabolite of .* can be (?P<mode>\S+)d when used in combination with (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_14 = re.compile('(?P<drug1>\w*\s*\w*) may (?P<mode>\S+) the (?P<se>.*) of (?P<drug2>\w*\s*\w*)')
DDI_SIDE_EFFECT_15 = re.compile('(?P<drug1>\w*\s*\w*) may (?P<mode>\S+) the central nervous system depressant (?P<se>\S+\s?\S*\s?\S*) of (?P<drug2>\w*\s*\w*)')

DDI_SIDE_EFFECTS = [
    DDI_SIDE_EFFECT_1, DDI_SIDE_EFFECT_2, DDI_SIDE_EFFECT_3, DDI_SIDE_EFFECT_4,
    DDI_SIDE_EFFECT_5, DDI_SIDE_EFFECT_6, DDI_SIDE_EFFECT_7, DDI_SIDE_EFFECT_8,
    DDI_SIDE_EFFECT_9, DDI_SIDE_EFFECT_10, DDI_SIDE_EFFECT_11, DDI_SIDE_EFFECT_12,
    DDI_SIDE_EFFECT_13, DDI_SIDE_EFFECT_14, DDI_SIDE_EFFECT_15
]

DDI_MODE_MAP = {
    'reduced': "decrease",
    'increase': "increase",
    'higher': "increase",
    'decrease': "decrease",
    'reduce': "decrease",
    'lower': "decrease"
}

DDI_SE_NAME_MAP = {
    "central_nervous_system_depressant_(cns_depressant)_activities": 'cns_depression_activities',
    "(cns_depressant)_activities": 'cns_depression_activities',
    "cns_depression": 'cns_depression_activities',
    "cardiotoxic_activities": 'cardiotoxicity',
    "constipating_activities": 'constipation',
    "excretion": 'excretion_rate',
    "hyperkalemic_activities": 'hyperkalemia',
    "hypertensive_activities": 'hypertension',
    "qtc-prolonging_activities": "qtc_prolongation",
    "tachycardic_activities": "tachycardia",
    "hypokalemic_activities": "hypokalemia",
    "hypoglycemic_activities": "hypoglycemia",
    "hypercalcemic_activities": "hypercalcemia",
    "bradycardic_activities": "bradycardia",
    "neutropenic_activities": "neutropenia",
    "orthostatic_hypotensive_activities": "orthostatic_hypotension",
    "neutropenic_activities": "neutropenia",
    "pseudotumor_cerebri_activities": "pseudotumor_cerebri",
    "sedative_activities": "sedation",
    "ototoxic_activities": "ototoxicity",
    "neuromuscular_blocking_activities": "neuromuscular_blockade",
    "nephrotoxic_activities": "nephrotoxicity",
    "myelosuppressive_activities": "myelosuppression",
    "hypotensive_activities": "hypotension",
    "serum_level": "serum_concentration"
}

# food interactions
DFI_1 = re.compile('Avoid (?P<food>.*)$')
DFI_2 = re.compile('Administer (?P<food>.*)')
DFI_3 = re.compile('Drink plenty of (?P<food>.*)')
DFI_4 = re.compile('Limit (?P<food>.*) intake')
DFI_5 = re.compile('Take with (?P<food>.*)')
DFI_6 = re.compile('Exercise caution with (?P<food>.*)')
DFI_7 = re.compile('Take separate from (?P<food>.*)')

patterns = [DFI_1, DFI_2, DFI_3, DFI_4, DFI_5, DFI_6, DFI_7]
patterns_dict = { DFI_1: 'avoid', DFI_2: 'increase_intake', DFI_3: 'increase_intake', DFI_4: 'limit', 
                 DFI_5: 'take_with', DFI_6: 'avoid', DFI_7: 'avoid'}

pattern_interaction_map = {'avoid': 'decrease_effectiveness', 
                            'limit': 'decrease_effectiveness', 
                            'take_with': 'increase_effectiveness',
                            'increase_intake': 'decrease_adverse_effects',
                            'increase_antiplatelet_activities': 'increase_antiplatelet_activities'}


def sanatize_se_txt(txt):
    return txt.strip().replace(" ", "_").lower()


def extract_side_effects(desc, drug_orig):
    """
    Extracts side effects from drug drug interaction descriptions
    Parameters
    ----------
    desc : str
        The interaction description
    Returns
    -------
    side_effects : list
        The list of side effects of the interaction
    """
    side_effects = []
    drug = ''
    for pattern_index, pattern in enumerate(DDI_SIDE_EFFECTS):
        pg = re.match(pattern, desc)
        if pg is not None:
            se_name_list = []
            se_name = pg.group("se").lower()
            mode = pg.group("mode")
            drug = pg.group("drug1")
            if len(pg.groups()) > 3:
                drug2 = pg.group("drug2")
                if drug_orig == drug:
                    drug = drug2

            # Handle the case of multiple activities eg x, y and z activities
            has_word_activities = ("activities" in se_name)
            if has_word_activities:
                se_name = se_name.replace(" activities", "")
            mode_name = DDI_MODE_MAP[mode]
            if ", and" in se_name:
                se_name_list = [sanatize_se_txt(se) for se in se_name.replace("and", "").split(", ")]
            elif "and" in se_name:
                se_name_list = [sanatize_se_txt(se) for se in se_name.split(" and ")]
            else:
                se_name_list = [sanatize_se_txt(se_name)]

            
            if has_word_activities:
                se_name_list = [txt+"_activities" for txt in se_name_list]

            for side_effect in se_name_list:
                if side_effect in DDI_SE_NAME_MAP:
                    side_effect = DDI_SE_NAME_MAP[side_effect]
                side_effects.append(f'{mode_name}_{side_effect}')

            # decrease_excretion_rate
            if pattern_index == 5:
                side_effects.append('decrease_excretion_rate')
            elif pattern_index == 6:
                side_effects.append('increase_excretion_rate')

            break
    return side_effects, drug


def parse_drug_interactions():
    interactions_df = pd.read_csv('data/drugbank/drug_interactions.csv', index_col=[0])

    drug_id_map = pd.read_csv('data/drugbank/drug_id_name_map.csv', index_col=[0])
    drug_ids = drug_id_map.id
    drug_names = drug_id_map.drug_name
    drug_id_map_dict = dict(zip(drug_names, drug_ids))

    drugs1 = []
    drugs2 = []
    interactions = []

    for inter in interactions_df.itertuples():
        # drug_id = inter[1]
        drug_name1 = inter[2]
        description = inter[3]
        side_effects, drug_name2 = extract_side_effects(description, drug_name1)
        # print(description)
        # print('drug1:', drug_name1)
        # print('drug2:', drug_name2)
        # print(side_effects)
        # print()

        for se in side_effects:
            drugs1.append(drug_name1)
            drugs2.append(drug_name2)
            interactions.append(se)

    drugs1 = pd.Series(drugs1)
    drugs1_ids = drugs1.map(drug_id_map_dict)
    drugs2 = pd.Series(drugs2)
    drugs2_ids = drugs2.map(drug_id_map_dict) 

    interactions_triplets = pd.DataFrame({'drug1': drugs1_ids, 'interaction': interactions, 'drug2': drugs2_ids})
    interactions_triplets = interactions_triplets.dropna()
    interactions_triplets.to_csv('data/triplets/ddi.tsv', sep='\t')
    
def extract_food_interaction(desc):
    interaction_type = ''
    food = ''
   
    for pattern in patterns:
        desc = desc.split('. ')[0]
        desc = re.sub('\.', '', desc)
        desc = re.sub('\ \(eg', '', desc)
        pg = re.match(pattern, desc)
        if pg is not None:
            interaction_type = patterns_dict[pattern]
            food = pg.group("food").lower() 
            continue
   
    return interaction_type, food                
    
def parse_food_interactions():
    food_interactions = pd.read_csv('data/drugbank/drug_food_interactions.csv', index_col=[0])
    drugs = pd.read_csv('data/drugbank/drug_id_name_map.csv', index_col=[0])
            
    drugs_list = []
    food_list = []
    interactions = []

    for inter in food_interactions.itertuples():
        drug = drugs[drugs['id'] == inter[1]].values[0][1]
        desc = inter[2]

        interaction, food = extract_food_interaction(desc)

        if len(interaction) > 0 and interaction != 'nothing' and food != 'or without food':
            if 'anticoagulant/antiplatelet activity' in food:
                interaction = 'increase_antiplatelet_activities' 

            elif food == 'st':
                food = "St. John's Wort"

            elif 'alcohol' in food:
                food = "alcohol"

            elif 'water' in food or 'fluids' in food:
                food = "water"

            elif 'grapefruit' in food:
                food = "grapefruit"

            elif 'potassium' in food:
                food = 'potassium'

            elif 'calcium' in food:
                food = 'calcium'

            elif 'dairy' in food:
                food = 'dairy products'    

            interaction = pattern_interaction_map[interaction]

            drugs_list.append(drug)
            food_list.append(food)  
            interactions.append(interaction)

    food_interactions_triplets = pd.DataFrame({'drug1': drugs_list, 'interaction': interactions, 'drug2': food_list})  # drug1/2 -> better parsing later
    food_interactions_triplets.to_csv('data/triplets/dfi.tsv', sep='\t')        


parse_drug_interactions()
# parse_food_interactions()
