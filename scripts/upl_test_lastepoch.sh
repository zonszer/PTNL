#!/bin/bash

cd ..

# custom config
DATA=./data
TRAINER=UPLTrainer
exp_ID="10.26-DEBUG_test_cc_rc_ep100_normal"    #NOTE +time     log_10.20-test_cc_refine_ep100_refillpool_ssucf101.txt rn50_ep100_16shots-10.23-test_rc_refine_ep100_refillloop_1
# TODO: 
#1. change oonf clean threshold and set safe factor and range
#10.19-test_cc_refine_ep100_safe&clean2
# rememberi it use tfm_test now 10.19


# TAG=$(date +"%m-%d_%H-%M-%S")   # get current time stamp
TAG="${exp_ID}"      

declare -a PLL_partial_rates=(0.3 0.1)
# PLL_partial_rate=$1      #ssucf101 ssoxford_pets
CFG=$1  # config file
CTP=$2  # class token position (end or middle)
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
CSC=$5  # class-specific context (False or True)
CLASS_EQULE=$6  # CLASS_EQULE True or False
FP=$7 # number of false positive training samples per class
# SEED=$9
#---------------------------common settings: ---------------------------
# loss_type=$9 # loss used in the training
use_PLL=True    #$10
TAG=$TAG # log tag (multiple_models_random_init or rn50_random_init)
#---------------------------individual settings: ---------------------------

for PLL_partial_rate in "${PLL_partial_rates[@]}"
do

    # #---------------------------individual settings: ---------------------------
    USE_REGULAR=False     #add 2
    USE_LABEL_FILTER=False
    # declare -a BETAS=(0.0 0.1 0.2 0.3)
    BETA=0.0
    declare -a CONF_MOMNs=(0.95)
    declare -a TOP_POOLs=(2)
    # declare -a MAX_POOLNUMs=(14 16)
    declare -a DATASETs=('ssucf101')      #'ssoxford_flowers' 'ssucf101' 
    declare -a SAFT_FACTORs=(3.5 4.0 4.5 5.0)
    # declare -a SHRINK_FACTORs=(0.5 0.3 0.7)

    if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
        declare -a MAX_POOLNUMs=(16)  
    elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
        declare -a MAX_POOLNUMs=(16)  
    else 
        echo "Invalid rate for MAX_POOLNUMs"
    fi


    for SEED in {1..3}
    do
        for DATASET in "${DATASETs[@]}"
        do
            LOG_FILE="logs_scripts/log_${TAG}_${DATASET}--LastEpoch.txt"
            for loss_type in 'rc_cav' 'cc' 'rc_rc'
            do
                for TOP_POOL in "${TOP_POOLs[@]}"
                do
                    for CONF_MOMN in "${CONF_MOMNs[@]}"
                    do
                        for SAFT_FACTOR in "${SAFT_FACTORs[@]}"
                        do
                            for MAX_POOLNUM in "${MAX_POOLNUMs[@]}"
                            do
                                common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}_FILT-${USE_LABEL_FILTER}_cMomn-${CONF_MOMN}_topP-${TOP_POOL}_MAXPOOL-${MAX_POOLNUM}_safeF-${SAFT_FACTOR}"
                                DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots-${TAG}/SEED${SEED}/${common_id}
                                if [ -d "$DIR" ]; then
                                    echo "Results are available in ${DIR}. load model-last-0.pth.tar and test:"

                                    python upl_train.py \
                                    --root ${DATA} \
                                    --seed ${SEED} \
                                    --trainer ${TRAINER} \
                                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                                    --output-dir ${DIR} \
                                    --num-fp ${FP} \
                                    --loss_type ${loss_type} \
                                    TRAINER.UPLTrainer.N_CTX ${NCTX} \
                                    TRAINER.UPLTrainer.CSC ${CSC} \
                                    TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
                                    DATASET.NUM_SHOTS ${SHOTS} \
                                    DATASET.CLASS_EQULE ${CLASS_EQULE} \
                                    TEST.FINAL_MODEL last_step \
                                    TRAINER.PLL.BETA ${BETA} \
                                    TRAINER.PLL.USE_REGULAR ${USE_REGULAR} \
                                    TRAINER.PLL.USE_PLL ${use_PLL} \
                                    TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate} \
                                    TRAINER.PLL.USE_LABEL_FILTER ${USE_LABEL_FILTER} \
                                    TRAINER.PLL.CONF_MOMN ${CONF_MOMN} \
                                    TRAINER.PLL.MAX_POOLNUM ${MAX_POOLNUM} \
                                    TRAINER.PLL.SAFE_FACTOR ${SAFT_FACTOR} \
                                    TRAINER.PLL.TOP_POOLS ${TOP_POOL}
                                    
                                    # Get the latest log file based on the timestamp in the filename
                                    LATEST_LOG=$(ls ${DIR}/log.txt* | tail -n 1)
                                    # Use the latest log file in your script
                                    ACCURACY=$(grep -A4 'Do evaluation on test set' ${LATEST_LOG} | grep 'accuracy:' | awk -F' ' '{print $3}')
                                    RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
                                    echo "${RECORD}" | tee -a ${LOG_FILE}
                                    echo "${RECORD}" >> ${LATEST_LOG}
                                else
                                    echo "Results are not available for configure: ${DIR}. Skip this job"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done

    #---------------------------individual settings: ---------------------------
    USE_REGULAR=False     #add 2
    USE_LABEL_FILTER=True
    # declare -a BETAS=(0.0 0.1 0.2 0.3)
    BETA=0.0
    declare -a CONF_MOMNs=(0.95 0.97 0.99)
    declare -a TOP_POOLs=(1 2 3 4)
    # declare -a MAX_POOLNUMs=(14 16)
    declare -a DATASETs=('ssdtd' 'ssucf101')
    declare -a SAFT_FACTORs=(3.5 4.0 5.0)

    if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
        declare -a MAX_POOLNUMs=(16 14 12)  
    elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
        declare -a MAX_POOLNUMs=(16 14 12)  
    else 
        echo "Invalid rate for MAX_POOLNUMs"
    fi


    for SEED in {1..3}
    do
        for DATASET in "${DATASETs[@]}"
        do
            LOG_FILE="logs_scripts/log_${TAG}_${DATASET}--LastEpoch.txt"
            for loss_type in 'cc_refine'
            do
                for TOP_POOL in "${TOP_POOLs[@]}"
                do
                    for CONF_MOMN in "${CONF_MOMNs[@]}"
                    do
                        for SAFT_FACTOR in "${SAFT_FACTORs[@]}"
                        do
                            for MAX_POOLNUM in "${MAX_POOLNUMs[@]}"
                            do
                                common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}_FILT-${USE_LABEL_FILTER}_cMomn-${CONF_MOMN}_topP-${TOP_POOL}_MAXPOOL-${MAX_POOLNUM}_safeF-${SAFT_FACTOR}"
                                DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots-${TAG}/SEED${SEED}/${common_id}
                                if [ -d "$DIR" ]; then
                                    echo "Results are available in ${DIR}. load model-last-0.pth.tar and test:"

                                    python upl_train.py \
                                    --root ${DATA} \
                                    --seed ${SEED} \
                                    --trainer ${TRAINER} \
                                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                                    --output-dir ${DIR} \
                                    --num-fp ${FP} \
                                    --loss_type ${loss_type} \
                                    TRAINER.UPLTrainer.N_CTX ${NCTX} \
                                    TRAINER.UPLTrainer.CSC ${CSC} \
                                    TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
                                    DATASET.NUM_SHOTS ${SHOTS} \
                                    DATASET.CLASS_EQULE ${CLASS_EQULE} \
                                    TEST.FINAL_MODEL last_step \
                                    TRAINER.PLL.BETA ${BETA} \
                                    TRAINER.PLL.USE_REGULAR ${USE_REGULAR} \
                                    TRAINER.PLL.USE_PLL ${use_PLL} \
                                    TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate} \
                                    TRAINER.PLL.USE_LABEL_FILTER ${USE_LABEL_FILTER} \
                                    TRAINER.PLL.CONF_MOMN ${CONF_MOMN} \
                                    TRAINER.PLL.MAX_POOLNUM ${MAX_POOLNUM} \
                                    TRAINER.PLL.SAFE_FACTOR ${SAFT_FACTOR} \
                                    TRAINER.PLL.TOP_POOLS ${TOP_POOL}
                                    
                                    # Get the latest log file based on the timestamp in the filename
                                    LATEST_LOG=$(ls ${DIR}/log.txt* | tail -n 1)
                                    # Use the latest log file in your script
                                    ACCURACY=$(grep -A4 'Do evaluation on test set' ${LATEST_LOG} | grep 'accuracy:' | awk -F' ' '{print $3}')
                                    RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
                                    echo "${RECORD}" | tee -a ${LOG_FILE}
                                    echo "${RECORD}" >> ${LATEST_LOG}
                                else
                                    echo "Results are not available for configure: ${DIR}. Skip this job"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done

    # #---------------------------individual settings: ---------------------------
    USE_REGULAR=False     #add 2
    USE_LABEL_FILTER=True
    POOL_INITRATIO=0.4
    # declare -a BETAS=(0.0 0.1 0.2 0.3)
    BETA=0.0
    declare -a CONF_MOMNs=(1.0)
    declare -a TOP_POOLs=(1 2 3 4)
    # declare -a MAX_POOLNUMs=(16 19)     
    declare -a DATASETs=('ssdtd' 'ssucf101')
    declare -a SAFT_FACTORs=(2.5 3.0 3.5 4.0)
    declare -a HALF_USE_Ws=(0.4 0.5 0.6)

    if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
        declare -a MAX_POOLNUMs=(16 14 12)  
    elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
        declare -a MAX_POOLNUMs=(16 14 12)  
    else 
        echo "Invalid rate for MAX_POOLNUMs"
    fi


    # declare -a SHRINK_FACTORs=(0.5 0.3 0.7)


    for SEED in {1..3}
    do
        for DATASET in "${DATASETs[@]}"
        do
            LOG_FILE="logs_scripts/log_${TAG}_${DATASET}--LastEpoch.txt"
            for loss_type in 'rc_refine'
            do
                for TOP_POOL in "${TOP_POOLs[@]}"
                do
                    for CONF_MOMN in "${CONF_MOMNs[@]}"
                    do
                        for SAFT_FACTOR in "${SAFT_FACTORs[@]}"
                        do
                            for MAX_POOLNUM in "${MAX_POOLNUMs[@]}"
                            do
                                for HALF_USE_W in "${HALF_USE_Ws[@]}"
                                do
                                    common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}_FILT-${USE_LABEL_FILTER}_cMomn-${CONF_MOMN}_topP-${TOP_POOL}_MAXPOOL-${MAX_POOLNUM}_safeF-${SAFT_FACTOR}_halfW-${HALF_USE_W}"
                                    DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots-${TAG}/SEED${SEED}/${common_id}
                                    if [ -d "$DIR" ]; then
                                        echo "Results are available in ${DIR}. load model-last-0.pth.tar and test:"

                                        python upl_train.py \
                                        --root ${DATA} \
                                        --seed ${SEED} \
                                        --trainer ${TRAINER} \
                                        --dataset-config-file configs/datasets/${DATASET}.yaml \
                                        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                                        --output-dir ${DIR} \
                                        --num-fp ${FP} \
                                        --loss_type ${loss_type} \
                                        TRAINER.UPLTrainer.N_CTX ${NCTX} \
                                        TRAINER.UPLTrainer.CSC ${CSC} \
                                        TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
                                        DATASET.NUM_SHOTS ${SHOTS} \
                                        DATASET.CLASS_EQULE ${CLASS_EQULE} \
                                        TEST.FINAL_MODEL last_step \
                                        TRAINER.PLL.BETA ${BETA} \
                                        TRAINER.PLL.USE_REGULAR ${USE_REGULAR} \
                                        TRAINER.PLL.USE_PLL ${use_PLL} \
                                        TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate} \
                                        TRAINER.PLL.USE_LABEL_FILTER ${USE_LABEL_FILTER} \
                                        TRAINER.PLL.CONF_MOMN ${CONF_MOMN} \
                                        TRAINER.PLL.MAX_POOLNUM ${MAX_POOLNUM} \
                                        TRAINER.PLL.POOL_INITRATIO ${POOL_INITRATIO} \
                                        TRAINER.PLL.SAFE_FACTOR ${SAFT_FACTOR} \
                                        TRAINER.PLL.HALF_USE_W ${HALF_USE_W} \
                                        TRAINER.PLL.TOP_POOLS ${TOP_POOL}
                                        
                                        # Get the latest log file based on the timestamp in the filename
                                        LATEST_LOG=$(ls ${DIR}/log.txt* | tail -n 1)
                                        # Use the latest log file in your script
                                        ACCURACY=$(grep -A4 'Do evaluation on test set' ${LATEST_LOG} | grep 'accuracy:' | awk -F' ' '{print $3}')
                                        RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
                                        echo "${RECORD}" | tee -a ${LOG_FILE}
                                        echo "${RECORD}" >> ${LATEST_LOG}
                                    else
                                        echo "Results are not available for configure: ${DIR}. Skip this job"
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done