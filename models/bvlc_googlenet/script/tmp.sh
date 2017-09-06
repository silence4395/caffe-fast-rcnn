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
    top_5s=NAN
fi

echo "top-1: $top_1, top-5: $top_1"

rm -i result.txt
printf "%-20s %-20s %-20s\n" LRN_TYPE TOP-1 TOP-5 >> result.txt
printf "%-20s %-20s %-20s\n" LUT_198 $top_1 $top_5 >> result.txt