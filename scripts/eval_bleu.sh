echo $1

awk 'NR%2==1{print $0}' ~/nlg-05/DATA/tasks/task2/roc/test_T_c_1.txt > t.tmp
awk 'NR%2==1{print $0}' $1 > t2.tmp

perl /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/scripts/multi-bleu.perl t.tmp < t2.tmp


awk 'NR%2==0{print $0}' ~/nlg-05/DATA/tasks/task2/roc/test_T_c_1.txt > t3.tmp
awk 'NR%2==0{print $0}' $1 > t4.tmp

perl /home/rcf-40/al_227/nlg-05/gits/seq2seq-attn/scripts/multi-bleu.perl t3.tmp < t4.tmp

rm t3.tmp
rm t4.tmp



