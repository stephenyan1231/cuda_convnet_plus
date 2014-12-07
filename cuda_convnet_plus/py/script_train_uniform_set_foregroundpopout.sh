#!/bin/bash

inDataDir="${PROJ_DIR}dl-image-enhance/data/uniform_set"
enhDataDir="${PROJ_DIR}dl-image-enhance/data/uniform_set_foregroundpopout"
layerDir=${inDataDir}/layers
trainData=uniform_set_foregroundpopout_7K_10_batch_seg_voting_k0.10

layerDef=${layerDir}/layers-mitfivek-imgglobal-pixlocalcontext-Lab-fc192.cfg
layerParams=${layerDir}/layer-params-mitfivek-imgglobal-pixlocalcontext-Lab-fc192.cfg

python convnet.py --data-path=${enhDataDir}/${trainData} --save-path=${enhDataDir}/convnet_checkpoints --train-range=1-10 --test-range=8-10  --layer-def=${layerDef} --layer-params=${layerParams} --data-provider=mitfivek_4  --test-freq=1000 --test-one=1 --gpu=0 --layer-verbose=0 --regress-L-channel-only=0 --use-local-context-ftr=1 --mini=512 --epochs=4000
