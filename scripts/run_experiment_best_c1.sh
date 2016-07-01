#!/bin/bash
#PBS -q isi
#PBS -l gpus=1
#PBS -l walltime=336:00:00
#PBS -e /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn_new/qsub_logs/err.$JOB_ID
#PBS -o /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn_new/qsub_logs/out.$JOB_ID
#PBS -d /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn_new/


h=500
a=0.5
d=0.2
l=2

th train.lua -data_file data/roc-c1-train.hdf5 -gpuid 1 -attn 0 -val_data_file data/roc-c1-val.hdf5 -savefile checkpoints/roc-c1-tune_h@${h}_lr@${a}_d@${d}_l@${l} -epochs 100 -num_layers $l -dropout $d -learning_rate $a -rnn_size $h > logs/log-c1_att@0_h@${h}_lr@${a}_d@${d}_l@${l}.txt
