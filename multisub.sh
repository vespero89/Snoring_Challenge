#!/bin/bash
for entry in Configs/*.cfg
do
	echo "$entry"
	#qsub -v CFGNAME=$entry pbs_run.pbs
done
 		 

