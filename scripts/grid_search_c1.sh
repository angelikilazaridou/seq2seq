#!/bin/bash
echo "hello"

hiddensize=( 500 1000 ) #1500 ) 
learningrate=( 0.3 0.5 1 ) 
dropout=( 0.2 ) #0.5 0.6  )
layers=( 1 2 3 )

for h in ${hiddensize[@]}; do
	for a in ${learningrate[@]}; do
		for d in ${dropout[@]}; do
			for l in ${layers[@]}; do										                		
				qsub  run_experiment_c1.sh -F "$h $a $d $l"
			done
		done																								
	done
done
