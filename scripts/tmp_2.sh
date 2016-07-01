#!/bin/bash

m=roc-c1-tune_h@500_lr@0.5_d@0.2_l@2_epoch1.00_60.20.t7

th beam.lua -gpuid 1 -model checkpoints/${m} -targ_file data/targ-val.txt -src_file data/src-val.txt -src_dict data/demo.src.dict -targ_dict data/demo.targ.dict -output_file predictions/pred_c1_${m}.txt 

#th beam.lua -gpuid 1 -model checkpoints/${m} -targ_file /home/rcf-40/al_227/nlg-05/DATA/tasks/task2/roc/test_T_c_1.txt -src_file /home/rcf-40/al_227/nlg-05/DATA/tasks/task2/roc/test_S_c_1.txt -output_file predictions/pred_c1_${m}.txt -src_dict data/roc-c1.src.dict -targ_dict data/roc-c1.targ.dict





