#!/bin/bash
# to rerun at full accuracy, do

# ./karmed.sh <dist_option> 300 300 2 1


python eval_mab.py $1 $2 $3 $4  --bayesucb --pgi_N 10 50 100 --pgi_T 1 10 50 --parallel $5
# python eval_mab.py $1 $2 $3 $4  --bayesucb --pfhgi --fhgi --gi --fhgi --parallel 5$
python plotkarmed.py $1 $2 $3 $4

