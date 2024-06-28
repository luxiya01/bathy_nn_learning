KP07_CONFIG="bathy_nn_learning/network_configs/kpthresh07.yaml"
KP07_MODEL_PATH="sin0627_19.pt/kpthresh_0.7"

TRAINED07_MODEL_PATH="finetuning/MBES_2023-09-15-11-23-09_epoch_29.pt/kpthresh_0.7"

NOISE="crop"
OVERLAP="0.2"
MODEL="kp07"
COMPUTE=true
EVAL=true

if [[ $MODEL == "kp07" ]]; then
    NETWORK_CONFIG=$KP07_CONFIG
    MODEL_PATH=$TRAINED07_MODEL_PATH
else
    echo "Unknown model $MODEL"
    exit 1
fi

CONFIG_FOLDER="bathy_nn_learning/mbes_data/configs/tests/meters"
MBES_CONFIG="$CONFIG_FOLDER/$NOISE/mbesdata_${NOISE}_meters_pairoverlap=$OVERLAP.yaml"
RESULTS_ROOT="20230711-$NOISE-meters-pairoverlap=$OVERLAP/${MODEL_PATH}"
mkdir -p $RESULTS_ROOT

logname="$RESULTS_ROOT/mbes_test-$NOISE-$OVERLAP-$(basename $NETWORK_CONFIG .yaml).log"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="

if [[ $COMPUTE == true ]]; then
    echo "Running mbes_test.py on noise=$NOISE overlap=$OVERLAP, network=$NETWORK_CONFIG..."
    echo "Using mbes_config=$MBES_CONFIG..."
    echo "logging to $logname..."

    python bathy_nn_learning/mbes_test.py \
        --mbes_config  $MBES_CONFIG\
        --network_config $NETWORK_CONFIG \
        | tee $logname
fi

if [[ $EVAL == true ]]; then
    echo "======================================="
    echo "Evaluating results at $RESULTS_ROOT..."
    python mbes-registration-data/src/evaluate_results.py \
        --results_root $RESULTS_ROOT \
        --use_transforms pred \
        | tee $RESULTS_ROOT/eval-res-$NOISE-$OVERLAP.log
fi
echo "Done!"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="