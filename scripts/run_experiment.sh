#!/bin/bash
#PBS -q isi
#PBS -l gpus=1
#PBS -l walltime=336:00:00
#PBS -e /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/qsub_logs/err.$JOB_ID
#PBS -o /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/qsub_logs/out.$JOB_ID
#PBS -d /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/


h=$1
a=$2
d=$3
l=$4

th train.lua -data_file data/roc-train.hdf5 -gpuid 1 -val_data_file data/roc-val.hdf5 -savefile checkpoints/roc-tune_h@${h}_lr@${a}_d@${d}_l@${l} -epochs 100 -num_layers $l -dropout $d -learning_rate $a -rnn_size $h > logs/log_h@${h}_lr@${a}_d@${d}_l@${l}.txt
