import json
import re

# scraped data (herbs_scraping.json) from https://www.mskcc.org/cancer-care/diagnosis-treatment/symptom-management/integrative-medicine/herbs/search
# for each herb, a list of interacting drugs and link to a web page extracted
def extract_herbs_drugs_interaction():
    with open('herbs_scraping.json', 'r') as f:
        herbs = json.load(f)

    herbs_interactions = dict()

    for i in range(len(herbs['herbs'])):
        h = list(herbs['herbs'][i])
        k = h[0]
        drug_interaction = herbs['herbs'][i][k]['herb-drug_interactions']
        if drug_interaction != "":
            if '\n' in drug_interaction[0]:
                drug_interaction = drug_interaction[0].split('\n')
            
            pattern = r'([^:]+):'
            drugs = []
            for inter in drug_interaction:
                match = re.match(r'^(.*?):', inter)
                if match is not None:
                    drugs.append(match.group(1))

            herbs_interactions[k] = dict()
            herbs_interactions[k]['drugs'] = drugs
            herbs_interactions[k]['url'] = herbs['herbs'][i][k]['url']

    # print(herbs_interactions)

    with open('herb-drug_interactions.json', 'w') as out:
        json.dump(herbs_interactions, out, indent=4)

# map extracted drugs to their corresponding ids (ATC codes)
# information from DrugBank (drug category, synonyms, ATC codes)
# some extracted drugs are just drug categories -> unpack to drugs (DrugBank - drug category) 
# mapping a drug name to ATC code
def drugs2ids():
    pass

if __name__ == '__main__':
    extract_herbs_drugs_interaction()

