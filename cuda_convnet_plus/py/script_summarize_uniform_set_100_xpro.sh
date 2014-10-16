#!/bin/bash
CHECKPOINT=ConvNet__2014-10-14_21.10.10zyan3-server
python shownet.py -f  ${PROJ_DIR}dl-image-enhance/data/uniform_set_100_xpro/convnet_checkpoints/${CHECKPOINT}  --show-cost=costlayer --cost-idx=0  --show-layer-weights-biases=0
