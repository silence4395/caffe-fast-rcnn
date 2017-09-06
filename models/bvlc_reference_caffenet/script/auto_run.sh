#!/bin/bash
#!/bin/sh
# time: 9/4/2017
# author: zhihui.luo@ingenic.com
#  AUTO run script
# 
####################################################
GPU_ID=0

lrn_type=(POWER AREAS LUT_198 LUT_400)

lrn_length=${#lrn_type[@]}
lrn_index=0
accuracy=100

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

function LookResult()
{
    grep "caffe.cpp:367] accuracy =" log
    BINGO=$?

    if [ $BINGO -eq 0 ]
    then
	var=$(ps -ef | grep "caffe.cpp:367] accuracy =" log)
	accuracy=${var#*=}
    else
	accuracy=NAN
    fi
    
    if [ $(echo "$accuracy == 100" | bc) -eq 1 ]
    then
	lrn_index=100
	echo "ERROR: Please check your program."
    fi
}

function RecoverPowerLayer()
{
    awk -F ' ' '{if(($1 == "op_type") && ($2 == "="))
                    {print " " " " $1 " " $2 " " "POWER;"}
                 else
                    {print $0;} }' power_layer.cu  >| tmp.cu
    mv tmp.cu power_layer.cu
}

function RecoverLRNLayer()
{
    awk -F ' ' '{ if(($1 == "op_type") && ($2 == "="))
                     {print " " " " $1 " " $2 " " "POWER;"}
                  else
                     {print $0;} }' lrn_layer.cu >| tmp.cu
    mv tmp.cu lrn_layer.cu
}

function RecoverLRNRistrettOLayer()
{
    awk -F ' ' '{ if(($1 == "op_type") && ($2 == "="))
                      {print " " " " $1 " " $2 " " "POWER;"}
                  else
                      {print $0;} }' lrn_ristretto_layer.cu >| tmp.cu
    mv tmp.cu lrn_ristretto_layer.cu
}

rm -i result.txt
printf "%-20s %-20s \n" LRN_TYPE ACCURACY >> result.txt
while(( $lrn_index<$lrn_length ))
do
    ./run.sh $GPU_ID ${lrn_type[$lrn_index]}
    LookResult
    CkeckUseGPU
    printf "%-20s %-20s \n" ${lrn_type[$lrn_index]} $accuracy >> result.txt
    echo "=============================="
    echo "LRN ${lrn_type[$lrn_index]} iteration done!"
    echo "=============================="
    let "lrn_index++"
done

# Recover CAFFE origin program config 
cd ../../../src/caffe/layers
RecoverPowerLayer
RecoverLRNLayer
cd ../quantization/layers
RecoverLRNRistrettOLayer