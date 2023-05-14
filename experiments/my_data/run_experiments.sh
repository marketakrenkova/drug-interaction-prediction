#!/bin/sh

cd experiment1
pykeen experiments ablation config.json -d results
cd ..

cd experiment2
pykeen experiments ablation config.json -d results
cd ..

cd experiment3
pykeen experiments ablation config.json -d results
cd ..

cd experiment4
pykeen experiments ablation config.json -d results
cd ..

# cd experiment5
# pykeen experiments ablation config.json -d results
# cd ..

cd experiment6
pykeen experiments ablation config.json -d results
cd ..

