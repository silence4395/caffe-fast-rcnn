#!/bin/bash
#!/bin/sh
# time: 9/4/2017
# author: zhihui.luo@ingenic.com
# 
# Parameter config:
# LRN optional type: POWER AREAS LUT_198 LUT_400
#
#
###############################################################
GPU_ID=$1

LRN_TYPE=$2

fixed_point=$3
bit_width=$4
fraction_length=$5

function SetLRNType()
{
    awk -v type=$1 -F ' ' '{if (($1 == "op_type") && ($2 == "="))
                               {print " " " " $1 " " $2 " " type ";"}
                            else
                               {print $0;}}' lrn_ristretto_layer.cu >| tmp.cu
    mv tmp.cu lrn_ristretto_layer.cu
}

function SetBitWidth()
{
    awk -v fp=$1 -v bw=$2 -v fl=$3 -F ' ' '{if (($1 == "bool") && ($2 == "fixed_point") && ($3 == "="))
                                               {print " " " " $1 " " $2 " " $3 " " fp ";"}
                                            else if (($1 == "int") && ($2 == "bit_width") && ($3 == "="))
                                               {print " " " " $1 " " $2 " " $3 " " bw ";"}
                                            else if (($1 == "int") && ($2 == "fl") && ($3 == "="))
                                               {print " " " " $1 " " $2 " " $3 " " fl ";"}
                                            else
                                               {print $0;} }' lrn_ristretto_layer.cu >| tmp.cu
    mv tmp.cu lrn_ristretto_layer.cu
}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd ../../../
cd src/caffe/quantization/layers/
SetLRNType $LRN_TYPE
SetBitWidth $fixed_point $bit_width $fraction_length
cd ../../../../
make -j |& tee log

grep "error" log
ERROR=$?
if [ $ERROR -eq 0 ]; then
    exit
fi
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "LRN function type:"
cd src/caffe/quantization/layers/
grep "op_type =" lrn_ristretto_layer.cu
grep "bool fixed_point =" lrn_ristretto_layer.cu
grep "int bit_width =" lrn_ristretto_layer.cu
grep "int fl =" lrn_ristretto_layer.cu
echo "Compiler done!"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

cd ../../../../
./build/tools/caffe test --gpu=$GPU_ID --model=models/bvlc_alexnet/lrn_quan.prototxt --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel |& tee log
mv log models/bvlc_alexnet/script/.