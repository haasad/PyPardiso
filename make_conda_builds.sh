#!/bin/bash

export LANG=en_US.UTF-8
pyversions=( "2.7" "3.4" "3.5" )
conda build purge

for pyversion in "${pyversions[@]}"
do
	echo $pyversion
	conda build . -q --python $pyversion --numpy '1.11'
	fname=$(conda build . --python $pyversion --numpy '1.11' --output)
	echo $fname
	conda convert --platform all $fname -o "$HOME/Desktop/conda_builds"
	echo "done with $pyversion"
done

