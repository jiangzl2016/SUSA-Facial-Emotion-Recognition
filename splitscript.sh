#!/bin/bash

# Runs python script that saves a new model to disk
# Saves it to a corresponding directory

for split in {0..4}
do
	mkdir exp_$split
	for num in {0..3}
	do
		mkdir exp_$split/model_$num
		echo "doing split $split number $x"
		python3 model_train.py --splitnum $split --number $num
	done
done
# sudo halt
