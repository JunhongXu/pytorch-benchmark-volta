#! /bin/bash
count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

echo 'start'
for (( c=count; c>=1; c-- ))
do
      python3 benchmark_models.py -g $c&&
done
echo 'end'
