#!/bin/sh


#cd experiment-drugbank
#pykeen experiments ablation config.json -d results
#cd ..

# cd experiment-biokg
# pykeen experiments ablation config.json -d results
# cd ..

cd experiment-hetionet
pykeen experiments ablation config.json -d results
cd ..

# cd experiment-interactions
# pykeen experiments ablation config.json -d results
# cd ..

# cd experiment-compounds
# pykeen experiments ablation config.json -d results
# cd ..
