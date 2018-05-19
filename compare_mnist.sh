#!/usr/bin/env bash

batch_sizes=( 256 512 768 1024 1280 1536 1792 2048 2304 2560 2816 3072 3328 3584 3840 4096)
layer_sizes=( 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000)

for BS in "${batch_sizes[@]}"
do
    :
    for LS in "${layer_sizes[@]}"
    do
        :
        FN="fp_${BS}_${LS}.log"
        XN="xg_${BS}_${LS}.log"

        if [ "${BS}" -gt "1500" ] && [ "${LS}" -gt "1500" ]; then
            if [ "${BS}" -gt "2200" ] && [ "${LS}" -gt "2200" ]; then
                STEPS=300
            else
                STEPS=600
            fi
        else
            STEPS=1000
        fi

        echo "Executing full-precision for BS = ${BS} and LS = ${LS} (${STEPS} steps) --> logs/bm_mnist/${FN}"
        python3 run_mnist.py --full_precision --learning_rate 0.005 --batch_size=${BS} --hidden_size=${LS} --steps ${STEPS} > logs/bm_mnist/${FN} 2>/dev/null

        # time to breathe for 10 seconds
        sleep 6

        echo "Executing xnor-network for BS = ${BS} and LS = ${LS} (${STEPS} steps) --> logs/bm_mnist/${FN}"
        python3 run_mnist.py --binary_xgemm --learning_rate 0.005 --batch_size=${BS} --hidden_size=${LS} --steps ${STEPS} > logs/bm_mnist/${XN} 2>/dev/null

        # time to breathe again for 10 seconds
        sleep 6
    done
done
