#!/bin/bash

cd bmDCA_2_states
##./bmDCA\_compile\_v2.1.sh
for name_file in $@
do
    echo $name_file
    outputfolder="OutputFolder_${name_file%%*(.fasta)}"
    echo $outputfolder
    mkdir $outputfolder
    ./bmDCA\_preprocessing.sh -rw $name_file
    ./bmDCA\_v2.1.sh Processed/msa\_numerical.txt Processed/weights.txt $outputfolder
done
exit 1
