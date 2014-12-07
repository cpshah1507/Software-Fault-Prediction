#!/bin/bash
if [ -f features_selected ] ; then
	rm features_selected filtered.arff 
fi

if [ "$#" -ne 2 ]; then
  echo "Usage: sh $0 <dataset name> <Number of times feature selection is to be done>" >&2
  exit 1
fi

for i in `seq 1 10`;
do
        python featureSelection.py $1 $2 >> temp1
        # echo "$i"
done  

while read line           
do
	temp=$(awk -F"{" '{print $2}' | awk -F"}" '{print $1}')
	if [ -n "$temp" ]; then
		echo "$temp" > temp2
    fi
done < temp1
sed '/^$/d' temp2 > selected_features_for_all_trials
rm temp1 temp2

python sig_features.py > extracted_features
read line < extracted_features
temp2=$(echo "$line" | tr "," " " | tr "[" " "| tr "]" " ")

python selectFeatures.py $1 $temp2

python testCrossVal.py