#!/bin/sh

# DDI dataset

#cd experiment1
#pykeen experiments ablation config.json -d results
#cd ..

#cd experiment2
#pykeen experiments ablation config.json -d results
#cd ..

#cd experiment3
#pykeen experiments ablation config.json -d results
#cd ..

# runs several days -> maybe less epochs ??
# cd experiment4
# pykeen experiments ablation config.json -d results
# cd ..

# cd experiment5
# pykeen experiments ablation config.json -d results
# cd ..

# --------------------------------------------
# BioKG dataset

# cd experiment-biokg-1
# pykeen experiments ablation config.json -d results
# cd ..

#cd experiment-biokg-2
#pykeen experiments ablation config.json -d results
#cd ..

cd experiment-biokg-3
pykeen experiments ablation config.json -d results
cd ..
