#!/bin/bash

python scripts/main.py -wf data/worflow-connection-subset-20-04.tsv -tu data/tool-popularity-20-04.tsv -om data/tool_recommendation_model.hdf5 -tm data/model.joblib -cd '2017-12-01' -pl 25 -me 20 -ts 0.05 -ne '1,20' -ct 'gini,entropy' -md '1,10' -mss '0.0001,1.0' -mf 'auto,sqrt,log2' -cpus 4
