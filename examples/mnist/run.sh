#!/bin/bash
#!/bin/sh
#  time: 4/14/2017
#  author: zhluo@ingenic.com
#  function:
#      shell for prunning, according to set error margin auto change prun ratio
##########################################################################################

set -e

PROTOTXT=$1
GPU_ID=$2
CAFFE_MODEL=$3
FLAG_FILE=$4
ERROR_MARGIN=$5

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXtRA_ARGS_SLUG=${EXTRA_ARGS// /_}

ITER_DONE=1
RATIO_CHANGE=1
CUT_FIELD_FC="--fc_ratio_0"
CUT_FIELD_CONV="--conv_ratio_0"
ORIGIN_ACCURACY=0.992
CAFFE_ROOT=/work/deep_compression/my_caffe_research/caffe
WEIGHT_ROOT=--weights=examples/mnist/lenet_iter_1000.caffemodel

if [ $# -lt 3 ]
then
    printf "\tUsage: \n"
    printf "\t\tprototxt: set prototxt file relative path \n"
    printf "\t\tset gpu id num: 0/1/all\n"
    printf "\t\tset error margin: eg. 0.01\n"
    printf "\tother parameters: \n"
    printf "\teg.\n"
    printf "\t\t--weights=\"weight file relative path\" \n"
    printf "\t\t--flagfile=\"google flag config file path\" \n"
    exit -1
fi

function Frun()
{
    ./build/tools/caffe train --solver=$PROTOTXT     \
			      --gpu=$GPU_ID          \
			      --weights=$CAFFE_MODEL \
			      --flagfile=$FLAG_FILE  \
			      ${EXTRA_ARGS}  2>&1 | tee log.txt
}
function Fflagfile_init()
{
    awk -F '=' '{
	 if (($1 == "--prun_conv") || ($1 == "--prun_fc") ||
	     ($1 == "--prun_retrain") || ($1 == "--sparse_csc") ||
	     ($1 == "--quan_enable") || ($1 == "--quan_retrain"))
	   {print $1 "=" "0";}
	 else
	   {print $0;}
      }' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    cp src/caffe/prun_cfg_1.cfg src/caffe/prun_cfg.cfg
}

# set prun option
function Fset_option_on()
{
    if [ $1 == 1 ]; then
	awk -F '=' '{if ($1 == "--prun_conv") {print $1 "=" "1"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 2 ]; then
	awk -F '=' '{if ($1 == "--prun_fc") {print $1 "=" "1"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 3 ]; then
	awk -F '=' '{if ($1 == "--prun_retrain") {print $1 "=" "1"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 4 ]; then
	awk -F '=' '{if ($1 == "--sparse_csc") {print $1 "=" "1"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 5 ]; then
	awk -F '=' '{if ($1 == "--quan_enable") {print $1 "=" "1"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 6 ]; then
	awk -F '=' '{if ($1 == "--quan_retrain") {print $1 "=" "1"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    else
	awk -F '=' '{print $0;}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    fi
    cp src/caffe/prun_cfg_1.cfg src/caffe/prun_cfg.cfg
}

function Fset_option_off()
{
    if [ $1 == 1 ]; then
	awk -F '=' '{if ($1 == "--prun_conv") {print $1 "=" "0"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 2 ]; then
	awk -F '=' '{if ($1 == "--prun_fc") {print $1 "=" "0"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 3 ]; then
	awk -F '=' '{if ($1 == "--prun_retrain") {print $1 "=" "0"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 4 ]; then
	awk -F '=' '{if ($1 == "--sparse_csc") {print $1 "=" "0"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 5 ]; then
	awk -F '=' '{if ($1 == "--quan_enable") {print $1 "=" "0"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    elif [ $1 == 6 ]; then
	awk -F '=' '{if ($1 == "--quan_retrain") {print $1 "=" "0"}
		     else {print $0;}}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    else
	awk -F '=' '{print $0;}' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    fi
    cp src/caffe/prun_cfg_1.cfg src/caffe/prun_cfg.cfg
}

function Ffc_prun_ratio()
{
    # TODO: one by one change FC layers' prun ratio
    awk -v CUT_FIELD_FC=$CUT_FIELD_FC -F '=' '{
	 if ($1 == CUT_FIELD_FC)
	   {{print $1 "=" $2+0.01}}
	 else
	   {print $0;}
      }' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    cp src/caffe/prun_cfg_1.cfg src/caffe/prun_cfg.cfg
}

function Ffc_ratio_recovery()
{
    awk -v CUT_FIELD_FC=$CUT_FIELD_FC -F '=' '{
	 if ($1 == CUT_FIELD_FC)
	   {{print $1 "=" $2-0.01}}
	 else
	   {print $0;}
      }' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    cp src/caffe/prun_cfg_1.cfg src/caffe/prun_cfg.cfg
}

function Fconv_prun_ratio()
{
    awk -v CUT_FIELD_CONV=$CUT_FIELD_CONV -F '=' '{
	 if ($1 == CUT_FIELD_CONV)
	   {{print $1 "=" $2+0.01}}
	 else
	   {print $0;}
      }' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    cp src/caffe/prun_cfg_1.cfg src/caffe/prun_cfg.cfg
}

function Fconv_ratio_recovery()
{
    awk -v CUT_FIELD_CONV=$CUT_FIELD_CONV -F '=' '{
	 if ($1 == CUT_FIELD_CONV)
	   {{print $1 "=" $2-0.01}}
	 else
	   {print $0;}
      }' src/caffe/prun_cfg.cfg >| src/caffe/prun_cfg_1.cfg
    cp src/caffe/prun_cfg_1.cfg src/caffe/prun_cfg.cfg
}

function Fcheck_accuracy()
{
    # traversal log, lookup
    ITER_DONE=`awk -v ERROR_MARGIN=$ERROR_MARGIN -v ORIGIN_ACCURACY=$ORIGIN_ACCURACY -F ' ' '{
      if (($4 == "solver.cpp:398]") && ($9 == "accuracy"))
      {
	if ((ORIGIN_ACCURACY-$11) < ERROR_MARGIN) {print "0";}
	else {print "1"};
      }}' log.txt`

    # get last compare result
    len=${#ITER_DONE}
    ITER_DONE=${ITER_DONE:$len-1:$len}
}

function Fmain()
{
    # initial flagfile
    Fflagfile_init
    Frun

    # prun FC layers
    Fset_option_on 2
    #Frun

    ## retrain FC layers
    ITER_DONE=1
    while (( $ITER_DONE == 1 ))
    do
    	Fset_option_on 3
    	Frun
    	Fcheck_accuracy
    done

    Fset_option_off 3
    echo " ^_^ Retrain done!"

    # explore new FC layers prun ratio
    while (( $ITER_DONE == 0 ))
    do
    	# change prun ratio
    	Ffc_prun_ratio
    	Frun
    	Fset_option_on 3
    	# retrain
    	ITER_DONE=1
    	while (( $ITER_DONE == 1 ))
    	do
    	    Frun
    	    Fcheck_accuracy
    	done
    	Fset_option_off 3
    done

    echo " >::< Recovery prun ratio."
    Ffc_ratio_recovery
    echo " ^-^ Optimize done!"
}

Fmain