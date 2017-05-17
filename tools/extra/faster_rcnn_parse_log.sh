#!/bin/bash
# Usage parse_log.sh caffe.log
# It creates the following two text files, each containing a table:
#     caffe.log.test (columns: '#Iters Seconds TestAccuracy TestLoss')
#     caffe.log.train (columns: '#Iters Seconds TrainingLoss LearningRate')


# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your.log"
exit
fi
LOG=`basename $1`
#sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > aux.txt
#sed -i '/Waiting for data/d' aux.txt
#sed -i '/prefetch queue empty/d' aux.txt
#sed -i '/Iteration .* loss/d' aux.txt
#sed -i '/Iteration .* lr/d' aux.txt
#sed -i '/Train net/d' aux.txt
#grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
#grep 'Test net output #0' aux.txt | awk '{print $11}' > aux1.txt
#grep 'Test net output #1' aux.txt | awk '{print $11}' > aux2.txt
#
## Extracting elapsed seconds
## For extraction of time since this line contains the start time
#grep 'Solving...' $1 > aux3.txt
#grep 'Testing net' $1 >> aux3.txt
#$DIR/faster_rcnn_extract_seconds.py aux3.txt aux4.txt
#
## Generating
#echo '#Iters Seconds TestAccuracy TestLoss'> $LOG.test
#paste aux0.txt aux4.txt aux1.txt aux2.txt | column -t >> $LOG.test
#rm aux.txt aux0.txt aux1.txt aux2.txt aux3.txt aux4.txt

# For extraction of time since this line contains the start time
grep 'Solving...' $1 > aux.txt
grep ', loss = ' $1 >> aux.txt
cp aux.txt aux_tmp.txt

awk -F ' ' '{if ($5 == "Iteration") {print $1 " " $2 " " $3 " " $4 " " $5 " " $6 ", " $11 " " $12 " " $13}
              else {print $0;}}' aux_tmp.txt >| aux.txt

grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ' #0: bbox_loss = ' $1 | awk '{print $11}' > aux1.txt
grep ' #1: cls_loss = ' $1 | awk '{print $11}' > aux2.txt
grep ' #2: rpn_cls_loss = ' $1 | awk '{print $11}' > aux3.txt
grep ' #3: rpn_loss_bbox = ' $1 | awk '{print $11}' > aux4.txt
grep ', lr = ' $1 | awk '{print $9}' > aux5.txt

# Extracting elapsed seconds
$DIR/faster_rcnn_extract_seconds.py aux.txt aux6.txt

# Generating
echo '#Iters Seconds Bbox_loss Cls_loss Rpn_cls_loss Rpn_loss_bbox LearningRate'> $LOG.train
paste aux0.txt aux6.txt aux1.txt aux2.txt aux3.txt aux4.txt aux5.txt | column -t >> $LOG.train
rm aux_tmp.txt  aux.txt aux0.txt aux6.txt aux1.txt aux2.txt aux3.txt aux4.txt aux5.txt
