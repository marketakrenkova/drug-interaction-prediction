#!/usr/bin/env python3

import pandas as pd

# DrugBank

# drug_id - name
# category - mesh_id ?
# drug_id - subclass
# subclass - class ?
# class - superclass ?
# superclass - kinkdom ?
# drug_id - interaction - drug_id (TODO: create interaction id + description map)
# drug_id - interaction - food
# drug_name - ingredients
# drug_id - inchi_key
# drug_id - molecule
# drug_id - salt_id
# salt_id - salt name
# salt_id - cas_number
# sal_id - inchi_key

# -----------------------------------------------------------
# FooDB

# food_id - name
# food_id - compound publid_id
# compound publid_id - source_id ??
# compound publid_id - compound name
# compound publid_id - cas_number
# compound publid_id - contribution score

# -----------------------------------------------------------
# iDISK

# concept id - type
# atom id - name
# atom id - source db
# atom id - id in source db
# attribute id - attribute name
# attribute id - type (concept/relation)
# attribute id - concept/relation id
# attribute id - value (decription)
# relation: concept1 id - relation id - concept2 id
# relation id - description (TODO: create relation id + description map)