#!/bin/bash

T=`date +%m%d%H%M%S`

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r loss exp/$T/code/loss
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log

python train.py 2>&1 | tee exp/$T/train.log/running.log
