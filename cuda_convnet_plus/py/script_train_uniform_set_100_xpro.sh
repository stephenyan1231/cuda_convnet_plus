#!/bin/bash

dataDir=/home/yzc/yzc/proj/cnn-image-enhance/data/uniform_set_100_xpro
layerDir=${dataDir}/layers
trainData=uniform_set_100_7K_10_batch_seg_voting_k0.10

#set layerDef=%layerDir%\layers-mitfivek-imgglobal-Lab.cfg
#set layerParams=%layerDir%\layer-params-mitfivek-imgglobal-Lab.cfg

layerDef=${layerDir}/layers-mitfivek-imgglobal-pixlocalcontext-Lab-fc192.cfg
layerParams=${layerDir}/layer-params-mitfivek-imgglobal-pixlocalcontext-Lab-fc192.cfg

#set layerDef=%layerDir%\layers-mitfivek-imgglobal-contextSem-contextColor-Lab-fc192.cfg
#set layerParams=%layerDir%\layer-params-mitfivek-imgglobal-contextSem-contextColor-Lab-fc192.cfg

python convnet.py --data-path=${dataDir}/${trainData} --save-path=${dataDir}/convnet_checkpoints --train-range=1-10 --test-range=8-10  --layer-def=${layerDef} --layer-params=${layerParams} --data-provider=mitfivek_4  --test-freq=100 --test-one=1 --gpu=0 --layer-verbose=0 --regress-L-channel-only=0 --use-local-context-ftr=1 --use-local-context-color-ftr=0 --use-position-ftr=0 --mini=512 --epochs=5000
