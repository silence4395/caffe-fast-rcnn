#!/bin/bash
#!/bin/sh
# time: 9/5/2017
# author: zhihui.luo@ingenic.com
#  AUTO run script
# 
####################################################
GPU_ID=0

bit_width=(8 16 24)
fl=('-1 0 1 2' '2 5 6 7' '2 12')

lrn_type=(POWER AREAS LUT_198 LUT_400)
lrn_length=${#lrn_type[@]}

lrn_index=0
bit_index=0
top_1=100
top_5=100

function CkeckUseGPU()
{
    grep "Use GPU with device ID" log
    BINGO=$?
    if [ $BINGO -eq 1 ]
    then
	echo "ERROR: Please use GPU mode."
	lrn_index=100
    fi
}

function CheckUseDynamicFixedPoint()
{
    grep "LRNRistretto" log
    BINGO=$?
    if [ $BINGO -eq 1 ]
    then
	echo "ERROR: Please use DynamicFixedPoint Net."
	lrn_index=100
	break
    fi
}

function LookResult()
{
    grep "caffe.cpp:367] loss3/top-1 =" log
    BINGO=$?

    if [ $BINGO -eq 0 ]
    then
	var=$(ps -ef | grep "caffe.cpp:367] loss3/top-1 =" log)
	top_1=${var#*=}
    else
	top_1=NAN
    fi

    grep "caffe.cpp:367] loss3/top-5 =" log
    BINGO=$?
    if [ $BINGO -eq 0 ]
    then
	var=$(ps -ef | grep "caffe.cpp:367] loss3/top-5 =" log)
	top_5=${var#*=}
    else
	top_5=NAN
    fi
    
    if [[ $(echo "$top_1 == 100" | bc) -eq 1 || $(echo "$top_5 == 100" | bc) -eq 1 ]]
    then
	lrn_index=100
	echo "ERROR: Please check your program."
    fi
}

function RecoverPowerLayer()
{
    awk -F ' ' '{if(($1 == "op_type") && ($2 == "="))
                    {print " " " " $1 " " $2 " " "POWER;"}
                  else if (($1 == "int") && ($2 == "fixed_point") && ($3 == "="))
                     {print " " " " $1 " " $2 " " $3 " " "0;"}
                 else
                    {print $0;} }' power_layer.cu  >| tmp.cu
    mv tmp.cu power_layer.cu
}

function RecoverLRNLayer()
{
    awk -F ' ' '{ if(($1 == "op_type") && ($2 == "="))
                     {print " " " " $1 " " $2 " " "POWER;"}
                  else if (($1 == "int") && ($2 == "fixed_point") && ($3 == "="))
                     {print " " " " $1 " " $2 " " $3 " " "0;"}
                  else
                     {print $0;} }' lrn_layer.cu >| tmp.cu
    mv tmp.cu lrn_layer.cu
}

function RecoverLRNRistrettOLayer()
{
    awk -F ' ' '{ if(($1 == "op_type") && ($2 == "="))
                      {print " " " " $1 " " $2 " " "POWER;"}
                  else if (($1 == "bool") && ($2 == "fixed_point") && ($3 == "="))
                     {print " " " " $1 " " $2 " " $3 " " "0;"}
                  else
                      {print $0;} }' lrn_ristretto_layer.cu >| tmp.cu
    mv tmp.cu lrn_ristretto_layer.cu
}

rm -i result.txt
printf "%-20s %-20s %-20s %-20s %-20s\n" LRN_TYPE BITWIDTH FRACTION TOP-1 TOP-5 >> result.txt
while(( $lrn_index<$lrn_length ))
do
    bit_index=0
    for row in "${fl[@]}"
    do
	row_value=($row)
	for fl_value in "${row_value[@]}"
	do
	    ./run_quan.sh $GPU_ID ${lrn_type[$lrn_index]} 1 ${bit_width[$bit_index]} $fl_value	    
	    LookResult
	    CkeckUseGPU
	    CheckUseDynamicFixedPoint
	    printf "%-20s %-20s %-20s %-20s %-20s\n" ${lrn_type[$lrn_index]} ${bit_width[$bit_index]} $fl_value $top_1 $top_5 >> result.txt
	    echo "=============================="
	    echo "LRN ${lrn_type[$lrn_index]} iteration done!"
	    echo "=============================="
	done
	CheckUseDynamicFixedPoint
	let "bit_index++"
    done
    let "lrn_index++"
done

# Recover CAFFE origin program config 
cd ../../../src/caffe/layers
RecoverPowerLayer
RecoverLRNLayer
cd ../quantization/layers
RecoverLRNRistrettOLayer

cd ../../../../
make -j