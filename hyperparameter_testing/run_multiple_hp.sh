#!/bin/sh


if [ -z "$1" ]; then
	echo "Error: missing first parameter (learning rate).\nUsage: $0 learn_rate batch_size"
	exit 1
fi
if [ -z "$2" ]; then
	echo "Error: missing second parameter (batch size).\nUsage: $0 learn_rate batch_size"
	exit 1
fi


learning_rate=$1 # default 0.0001
batch_size=$2 # default 2048

for i in 0 1 2 3 # these are seed values
do
	./run_wnet_prior_hp.sh $learning_rate $batch_size $i
done

