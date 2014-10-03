#!/bin/bash
CHECKPOINT=ConvNet__2014-10-03_09.01.52yzyu-server4
python shownet.py -f  ${PROJ_DIR}dl-image-enhance/data/uniform_set_100_xpro/convnet_checkpoints/${CHECKPOINT}  --show-cost=costlayer --cost-idx=0  --show-layer-weights-biases=0
