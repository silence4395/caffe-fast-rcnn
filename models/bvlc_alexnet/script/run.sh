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

bit_width=24
fraction_length=20

function SetLRNType()
{
    awk -v type=$1 -F ' ' '{if (($1 == "op_type") && ($2 == "="))
                               {print " " " " $1 " " $2 " " type ";"}
                            else
                               {print $0;}}' lrn_layer.cu >| tmp.cu
    mv tmp.cu lrn_layer.cu
}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd ../../../
cd src/caffe/layers
SetLRNType $LRN_TYPE
cd ../../../
make -j |& tee log

grep "error" log
ERROR=$?
if [ $ERROR -eq 0 ]; then
    exit
fi
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "LRN function type:"
cd src/caffe/layers
grep "op_type =" lrn_layer.cu
echo "Compiler done!"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

cd ../../../
#./build/tools/caffe test --gpu=$GPU_ID --model=models/bvlc_alexnet/train_val.prototxt --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel |& tee log
./build/tools/caffe test --gpu=$GPU_ID --model=models/bvlc_alexnet/lrn_quan.prototxt --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel |& tee log
mv log models/bvlc_alexnet/script/.