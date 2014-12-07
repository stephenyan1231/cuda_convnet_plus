#!/bin/bash

for checkpoint in "ConvNet__2014-12-05_21.53.05zyan3-server2"
do
	python shownet.py -f  ${PROJ_DIR}dl-image-enhance/data/uniform_set_foregroundpopout/convnet_checkpoints/${checkpoint}  --show-cost=costlayer --cost-idx=0  --show-layer-weights-biases=0

done



