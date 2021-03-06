#!/bin/bash
#PBS -q isi
#PBS -l gpus=1
#PBS -l walltime=336:00:00
#PBS -e /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/qsub_logs/err.$JOB_ID
#PBS -o /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/qsub_logs/out.$JOB_ID
#PBS -d /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/


m=$1

th beam.lua -gpuid 1 -model checkpoints/roc-c1-tune_${m}_final.t7 -src_file /home/rcf-40/al_227/nlg-05/DATA/tasks/task2/roc/test_S_c_1.txt  -targ_file /home/rcf-40/al_227/nlg-05/DATA/tasks/task2/roc/test_T_c_1.txt -output_file predictions/pred_c1_${m}.txt > predictions/out_c1_${m}.txt




