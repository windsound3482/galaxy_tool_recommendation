#!/bin/bash

python scripts/main.py -wf data/worflow-connection-subset-04-20.tsv -tu data/tool-popularity-20-03.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 5 -oe 5 -me 5 -ts 0.2 -bs '32,256' -ut '32,512' -es '32,512' -dt '0.0,0.5' -sd '0.0,0.5' -lr '0.00001,0.1' -cpus 4

