#!/bin/bash

if [ $1 == 'pretext' ]; then # train pretext
    python pretext.py \
        --epochs 100 \
        --batch-size 512 --wd 5e-4 \
        --moco-dim 128 --moco-k 4096 --moco-m 0.99 --moco-t 0.1 \
        --results-dir ./results
elif [ $1 == 'poj_cls_a' ]; then # train downstream (POJ program classification)
    echo "starting poj-classify-a..."
    python poj_classify_a.py --epochs 15 --classes 293 --batch-size 64 \
        --results-dir ./results
elif [ $1 == 'poj_cls_b' ]; then
    python poj_classify_b.py --lr 0.06 --epochs 20 \
        --classes 293 --batch-size 64 --token-vocab 30002 --path-vocab 50002 --contexts 600 \
        --results-dir ./results
else
    echo "unknown command"
fi