#!/bin/sh
# /!\ need to modify the python file a bit to accept external arguments (with sys.argv)

params="1.00e-05 1.40e-05 1.95e-05 2.72e-05 3.80e-05 ..."

for i in $params
do
    echo "$i"
    nohup python -u main.py "$i" 1> log.log 2> logerr.log &
#    python treat_HRB.py "$i"  # Not nohup to avoid running all at once and hogging all the processors
    echo $!
done
