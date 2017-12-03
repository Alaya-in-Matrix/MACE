#!/bin/bash
./clear.sh
dir=`pwd`
echo workdir $dir > conf
cat ./tmp.conf >> conf
nohup mace_bo ./conf > log 2>&1 &
