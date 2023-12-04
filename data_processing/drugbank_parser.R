## load dbparser package
library(dbparser)
library(dplyr)
library(ggplot2)
library(XML)

data_dir <- "/Users/marketa/work/school_staff/diplomka/data/DrugBank/"

## parse data from XML and save it to memory
read_drugbank_xml_db(paste(data_dir, "drugbank.xml", sep = ""))

## load drugs data
drugs <- drugs()

## load drug groups data
drug_groups <- drug_groups()

## load drug targets actions data
drug_targets_actions <- targets_actions()


# Exploring data
drugs_info <- drugs$general_information
drugs_info %>% 
  select(type) %>% 
  ggplot(aes(x = type, fill = type)) + 
  geom_bar() + 
  guides(fill = FALSE)     ## removes legend for the bar colors


## get counts of the different target actions in the data
targetActionCounts <- 
  drug_targets_actions %>% 
  group_by(action) %>% 
  summarise(count = n()) %>% 
  arrange(desc(count))

## get bar chart of the 10 most occurring target actions in the data
p <- ggplot(targetActionCounts[1:10,], 
         aes(x = reorder(action,count), y = count, fill = letters[1:10])) + 
  geom_bar(stat = 'identity') +
  labs(fill = 'action', 
       x = 'Target Action', 
       y = 'Quantity', 
       title = 'Target Actions Distribution', 
       subtitle = 'Distribution of Target Actions in the Data',
       caption = 'created by ggplot') + 
  #guides(fill = FALSE) +    ## removes legend for the bar colors
  coord_flip()              ## switches the X and Y axes

## display plot
p

## ---------------------------------------------------------------
## Create tables only with needed information

approved_drugs <- drug_groups[drug_groups$group == "approved", ]

# primary key (Drugbank ID) + name of drug
drug_id_name_map <- tibble(
  id = drugs$general_information$primary_key,
  drug_name = drugs$general_information$name
) |> subset(id %in% approved_drugs$`drugbank-id`)

write.csv(drug_id_name_map, paste(data_dir, "drug_id_name_map.csv", sep = ""))

# id + subclass + class + … (drug classification) 
drug_class <- tibble(
  id = drugs$drug_classification$drugbank_id,
  kingdom = drugs$drug_classification$kingdom,
  superclass = drugs$drug_classification$superclass,
  class = drugs$drug_classification$class,
  subclass = drugs$drug_classification$subclass,
  substituent = drugs$drug_classification$substituents
) |> subset(id %in% approved_drugs$`drugbank-id`)
write.csv(drug_class, paste(data_dir, "drug_classification.csv", sep = ""))

# pharmacology ??

# ingredients (mixtures)
drug_mix <- tibble(
  drug_name = drugs$mixtures$name,
  ingredients = drugs$mixtures$ingredients,
  ingredients_id = drugs$mixtures$parent_key
) |> subset(ingredients_id %in% approved_drugs$`drugbank-id`)
write.csv(drug_mix, paste(data_dir, "drug_mixtures.csv", sep = ""))


# category + id (categories) 
drug_category <- tibble(
  drug_id = drugs$categories$parent_key, 
  category = drugs$categories$category
)
write.csv(drug_category, paste(data_dir, "drug_category.csv", sep = ""))

# synonyms + id (synonyms) 
drug_synonyms <- tibble(
  drug_id = drugs$synonyms$`drugbank-id`, 
  synonym = drugs$synonyms$synonym
)
write.csv(drug_synonyms, paste(data_dir, "drug_synonyms.csv", sep = ""))

# id + name + description (interactions)
drug_interact <- tibble(
  id = drugs$interactions$`drugbank-id`,
  drug_name = drugs$interactions$name,
  description = drugs$interactions$description
) |> subset(id %in% approved_drugs$`drugbank-id`)
write.csv(drug_interact, paste(data_dir, "drug_interactions.csv", sep = ""))


# food interaction + id (food interactions)
drug_food_inter <- tibble(
  id = drugs$food_interactions$drugbank_id,
  description = drugs$food_interactions$food_interaction
) |> subset(id %in% approved_drugs$`drugbank-id`)
write.csv(drug_food_inter, paste(data_dir, "drug_food_interactions.csv", sep = ""))



# clac_prop - kind = InChi(key) value + parent_key
inchikey <- drugs$calc_prop[drugs$calc_prop$kind == 'InChIKey', ]
inchikey_drugs <- tibble(
  id = inchikey$parent_key,
  inchi_key = inchikey$value
) |> subset(id %in% approved_drugs$`drugbank-id`)
write.csv(inchikey_drugs, paste(data_dir, "drug_inchi_key.csv", sep = ""))

# exp_prop -  kind = Molecular Formula value + parent_key
molec <- drugs$exp_prop[drugs$exp_prop$kind == 'Molecular Formula', ]
molecul_drugs <- tibble(
  id = molec$parent_key,
  molecule = molec$value
) |> subset(id %in% approved_drugs$`drugbank-id`)
write.csv(molecul_drugs, paste(data_dir, "drug_molecul.csv", sep = ""))
# don't use -> too little

# salts - cas-number + name + parent_key (drug id) 
salts <- tibble(
  id = drugs$salts$`drugbank-id`,
  name = drugs$salts$name,
  cas_number = drugs$salts$`cas-number`,
  inchi_key = drugs$salts$inchikey,
  drug = drugs$salts$parent_key
) |> subset(drug %in% approved_drugs$`drugbank-id`)
write.csv(salts, paste(data_dir, "drug_salts.csv", sep = ""))

# ATC code
drug_atc_code <- tibble(
  id = drugs$atc_codes$`drugbank-id`,
  atc_code = drugs$atc_codes$atc_code
) |> subset(id %in% approved_drugs$`drugbank-id`)
write.csv(drug_atc_code, paste(data_dir, "drug_atc_code.csv", sep = ""))

# pathway 
pathways <- tibble(
  smpdb_id = drugs$pathway$smpdb_id,
  pathway_name = drugs$pathway$name,
  category = drugs$pathway$category,
  drug_id = drugs$pathway$parent_key
) |> subset(drug_id %in% approved_drugs$`drugbank-id`)
write.csv(pathways, paste(data_dir, "pathways.csv", sep = ""))

pathways_ids_list <- pathways$smpdb_id

pathways_enzym <- tibble(
  smpdb_id = drugs$pathway_enzyme$pathway_id,
  enzyme_id = drugs$pathway_enzyme$enzyme
) |> subset(smpdb_id %in% pathways_ids_list)
write.csv(pathways_enzym, paste(data_dir, "pathways_enzym.csv", sep = ""))

# reactions ?
# structure



#--------------------------------------------------
# plot count of interaction of each drug
agg_df_interactions <- drug_interact |> group_by(id) |> summarise(total_count=n(),
                                        .groups = 'drop')
agg_df_interactions |> ggplot(aes(x=id, y=total_count)) +
  geom_bar(stat = "identity")




