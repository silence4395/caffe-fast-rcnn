grep "caffe.cpp:367] accuracy =" log
BINGO=$?

if [ $BINGO -eq 0 ]
then
    var=$(ps -ef | grep "caffe.cpp:367] accuracy =" log)
    accuracy=${var#*=}
else
    accuracy=NAN
fi

echo $accuracy

rm -i result.txt
printf "%-20s %-20s \n" LRN_TYPE ACCURACY >> result.txt
printf "%-20s %-20s \n" LUT_198 $accuracy >> result.txt