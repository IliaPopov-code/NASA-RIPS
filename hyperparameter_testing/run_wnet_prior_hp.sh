#!/bin/sh

if [ -z "$1" ]; then
	echo "Error: missing first parameter (learning rate).\nSpecify learning_rate batch_size seed."
	exit 1
fi
if [ -z "$2" ]; then
	echo "Error: missing second parameter (batch size).\nSpecify learning_rate batch_size seed."
	exit 1
fi
if [ -z "$3" ]; then
	echo "Error: missing third parameter (batch size).\nSpecify learning_rate batch_size seed."
	exit 1
fi


# hp_version.txt should contain version=<version number>
versionfile="./hp_version.txt"
trainfile="./run_wnet_prior_hp.py"

. $versionfile # loads version
learning_rate=$1 # default 0.0001
batch_size=$2 # default 2048
seed=$3 # default 0

logfile="wnet_prior_v$version.log"
dir="hp_v$version"

# directory hp_tests should exist
cd ./hp_tests
mkdir $dir
cd $dir

echo "Training model $version with learning rate $learning_rate, batch_size $batch_size, seed $seed."
nohup python $trainfile $version $learning_rate $batch_size $seed > $logfile 2>&1

printf 'next version will be %s (saving in hp_version.txt)\n' "$((version+1))"
printf 'version=%s' "$((version+1))" > $versionfile
