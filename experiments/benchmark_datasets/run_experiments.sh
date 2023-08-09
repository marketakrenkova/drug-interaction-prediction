#!/bin/sh

cd BioKG
pykeen experiments ablation config.json -d results
cd ..

cd Hetionet
pykeen experiments ablation config.json -d results
cd ..

# cd OGBBioKG
# pykeen experiments ablation config.json -d results
# cd ..

cd openBioLink
pykeen experiments ablation config.json -d results
cd ..

cd PharmKG
pykeen experiments ablation config.json -d results
cd ..
