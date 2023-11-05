#!/bin/bash

cd ..

# custom config
DATA=./data
TRAINER=UPLTrainer
exp_ID="11.5-test_refine_ep100_All-0.8"    #NOTE +time 共72+27=99次
# TODO: 
#1. change oonf clean threshold and set safe factor and range
#10.19-test_cc_refine_ep100_safe&clean2
# rememberi it use tfm_test now 10.19


# TAG=$(date +"%m-%d_%H-%M-%S")   # get current time stamp
TAG="${exp_ID}"      

# PLL_partial_rates=$1      #ssucf101 ssoxford_pets
PLL_partial_rates="$1" 
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
CLASS_EQULE=$7  # CLASS_EQULE True or False
FP=$8 # number of false positive training samples per class
# SEED=$9
#---------------------------common settings: ---------------------------
# loss_type=$9 # loss used in the training
use_PLL=True    #$10
TAG=$TAG # log tag (multiple_models_random_init or rn50_random_init)
IFS=',' read -ra PLL_partial_rates <<< "$PLL_partial_rates"
echo "PLL_partial_rates: ${PLL_partial_rates[@]}"
#---------------------------individual settings: ---------------------------

# Function to run the job
run_job() {
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
    TRAINER.PLL.HALF_USE_W ${HALF_USE_W} \
    TRAINER.PLL.TOP_POOLS ${TOP_POOL}
}

USE_REGULAR=False
USE_LABEL_FILTER=True
BETA=0.0
SEEDs=(1 2 3)
declare -a DATASETs=('ssucf101')
declare -a POOL_INITRATIOs=(0.3 0.5)
declare -a loss_types=('rc_refine' 'cav_refine' 'cc_refine')

set_values() {
    local loss_type=$1
    local PLL_partial_rate=$2
    local CONF_MOMNs
    local HALF_USE_Ws

    if [ "$loss_type" == "cav_refine" ]; then
        if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
            CONF_MOMNs=(0.00)
            HALF_USE_Ws=(0.5 0.6 0.7)
        elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
            CONF_MOMNs=(0.00)
            HALF_USE_Ws=(0.2 0.3 0.4)
        fi

    elif [ "$loss_type" == "rc_refine" ]; then
        if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
            CONF_MOMNs=(0.3 0.4)
            HALF_USE_Ws=(0.3 0.4 0.5)
        elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
            CONF_MOMNs=(0.2 0.3)
            HALF_USE_Ws=(0.2 0.3 0.4)
        fi

    elif [ "$loss_type" == "lw_refine" ]; then
        if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
            CONF_MOMNs=(0.3 0.4)
            HALF_USE_Ws=(0.3 0.4 0.5)
        elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
            CONF_MOMNs=(0.2 0.3)
            HALF_USE_Ws=(0.2 0.3 0.4)
        fi

    elif [ "$loss_type" == "cc_refine" ]; then
        if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
            CONF_MOMNs=(0.03 0.05)
            HALF_USE_Ws=(0.3 0.4 0.5)
        elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
            CONF_MOMNs=(0.03 0.05)
            HALF_USE_Ws=(0.3 0.4 0.5)
        fi
    else
        echo "Invalid loss_type"
        exit 1
    fi

    echo "${CONF_MOMNs[*]} ; ${HALF_USE_Ws[*]}"
}


for SEED in "${SEEDs[@]}"; do
    for DATASET in "${DATASETs[@]}"; do
        LOG_FILE="logs_scripts/log_${TAG}_${DATASET}.txt"
        for loss_type in "${loss_types[@]}"; do
            for PLL_partial_rate in "${PLL_partial_rates[@]}"; do

                if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
                    MAX_POOLNUMs=(16)
                    TOP_POOLs=(3 4)
                elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
                    MAX_POOLNUMs=(16)
                    TOP_POOLs=(3 4)
                else
                    echo "Invalid rate for MAX_POOLNUMs"
                    exit 1
                fi
                
                set_values_output=$(set_values $loss_type $PLL_partial_rate)
                # Split the output into two parts using the special character
                IFS=';' read -ra arrays <<< "$set_values_output"
                # Split each part into an array using space as the delimiter
                IFS=' ' read -ra CONF_MOMNs <<< "${arrays[0]}"
                IFS=' ' read -ra HALF_USE_Ws <<< "${arrays[1]}"
                # echo "CONF_MOMNs: ${CONF_MOMNs[@]}, HALF_USE_Ws: ${HALF_USE_Ws[@]}"

                for TOP_POOL in "${TOP_POOLs[@]}"; do
                    for CONF_MOMN in "${CONF_MOMNs[@]}"; do
                        for POOL_INITRATIO in "${POOL_INITRATIOs[@]}"; do
                            for MAX_POOLNUM in "${MAX_POOLNUMs[@]}"; do
                                for HALF_USE_W in "${HALF_USE_Ws[@]}"; do
                                    total_iterations=$((${#SEEDs[@]} * ${#DATASETs[@]} * ${#loss_types[@]} * ${#PLL_partial_rates[@]} * ${#TOP_POOLs[@]} * ${#CONF_MOMNs[@]} * ${#POOL_INITRATIOs[@]} * ${#MAX_POOLNUMs[@]} * ${#HALF_USE_Ws[@]}))
                                    echo "The loop will iterate $total_iterations times."

                                    common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}_FILT-${USE_LABEL_FILTER}_cMomn-${CONF_MOMN}_topP-${TOP_POOL}_MAXPOOL-${MAX_POOLNUM}_initR-${POOL_INITRATIO}_halfW-${HALF_USE_W}"
                                    DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots-${TAG}/SEED${SEED}/${common_id}
                                    
                                    if [ -d "$DIR" ]; then
                                        echo -e "------------\n Results are available in ${DIR}. Skip this job"
                                    else
                                        echo "======>>> Run this job and save the output to ${DIR}"
                                        run_job

                                        ACCURACY=$(grep -A4 'Do evaluation on test set' ${DIR}/log.txt | grep 'accuracy:' | awk -F' ' '{print $3}')
                                        RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
                                        echo "${RECORD}" | tee -a ${LOG_FILE}
                                        echo "${RECORD}" >> ${DIR}/log.txt
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






# for SEED in {1..3}
# do
#     for DATASET in "${DATASETs[@]}"
#     do
#         LOG_FILE="logs_scripts/log_${TAG}_${DATASET}.txt"
#         for loss_type in loss_types
#         do
#             if [ "$loss_type" == "cav_refine" ] && (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
#                 declare -a CONF_MOMNs=(0.00)
#                 declare -a HALF_USE_Ws=(0.5 0.6 0.8)
#             fi
#             if [ "$loss_type" == "cav_refine" ] && (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
#                 declare -a CONF_MOMNs=(0.00)
#                 declare -a HALF_USE_Ws=(0.0 0.1 0.2)
#             fi

#             if [ "$loss_type" == "rc_refine" ] && (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
#                 declare -a CONF_MOMNs=(0.3 0.4)
#                 declare -a HALF_USE_Ws=(0.3 0.4)
#             fi
#             if [ "$loss_type" == "rc_refine" ] && (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
#                 declare -a CONF_MOMNs=(0.2 0.3)
#                 declare -a HALF_USE_Ws=(0.1 0.2)
#             fi

#             if [ "$loss_type" == "lw_refine" ] && (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
#                 declare -a CONF_MOMNs=(0.2 0.3)
#                 declare -a HALF_USE_Ws=(0.2 0.3 0.4)
#             fi
#             if [ "$loss_type" == "cc_refine" ] && (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
#                 declare -a CONF_MOMNs=(0.2 0.3 0.4)
#                 declare -a HALF_USE_Ws=(0.2 0.3)
#             fi

#             if [ "$loss_type" == "cc_refine" ] && (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
#                 declare -a HALF_USE_Ws=(0.3 0.4 0.5)
#                 declare -a CONF_MOMNs=(0.03 0.05)
#             elif [ "$loss_type" == "lw_refine" ] && (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
#                 declare -a HALF_USE_Ws=(0.5 0.6 0.7)
#                 declare -a CONF_MOMNs=(0.01 0.05)

#             for TOP_POOL in "${TOP_POOLs[@]}"
#             do
#                 for CONF_MOMN in "${CONF_MOMNs[@]}"
#                 do
#                     for POOL_INITRATIO in "${POOL_INITRATIOs[@]}"
#                     do
#                         for MAX_POOLNUM in "${MAX_POOLNUMs[@]}"
#                         do
#                             for HALF_USE_W in "${HALF_USE_Ws[@]}"
#                             do
#                                 common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}_FILT-${USE_LABEL_FILTER}_cMomn-${CONF_MOMN}_topP-${TOP_POOL}_MAXPOOL-${MAX_POOLNUM}_initR-${POOL_INITRATIO}_halfW-${HALF_USE_W}"
#                                 DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots-${TAG}/SEED${SEED}/${common_id}
#                                 if [ -d "$DIR" ]; then
#                                     echo "------------Results are available in ${DIR}. Skip this job"
#                                 else
#                                     echo "Run this job and save the output to ${DIR}"
#                                     python upl_train.py \
#                                     --root ${DATA} \
#                                     --seed ${SEED} \
#                                     --trainer ${TRAINER} \
#                                     --dataset-config-file configs/datasets/${DATASET}.yaml \
#                                     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#                                     --output-dir ${DIR} \
#                                     --num-fp ${FP} \
#                                     --loss_type ${loss_type} \
#                                     TRAINER.UPLTrainer.N_CTX ${NCTX} \
#                                     TRAINER.UPLTrainer.CSC ${CSC} \
#                                     TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
#                                     DATASET.NUM_SHOTS ${SHOTS} \
#                                     DATASET.CLASS_EQULE ${CLASS_EQULE} \
#                                     TEST.FINAL_MODEL last_step \
#                                     TRAINER.PLL.BETA ${BETA} \
#                                     TRAINER.PLL.USE_REGULAR ${USE_REGULAR} \
#                                     TRAINER.PLL.USE_PLL ${use_PLL} \
#                                     TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate} \
#                                     TRAINER.PLL.USE_LABEL_FILTER ${USE_LABEL_FILTER} \
#                                     TRAINER.PLL.CONF_MOMN ${CONF_MOMN} \
#                                     TRAINER.PLL.MAX_POOLNUM ${MAX_POOLNUM} \
#                                     TRAINER.PLL.POOL_INITRATIO ${POOL_INITRATIO} \
#                                     TRAINER.PLL.SAFE_FACTOR ${SAFT_FACTOR} \
#                                     TRAINER.PLL.HALF_USE_W ${HALF_USE_W} \
#                                     TRAINER.PLL.TOP_POOLS ${TOP_POOL}
                                    
#                                     ACCURACY=$(grep -A4 'Do evaluation on test set' ${DIR}/log.txt | grep 'accuracy:' | awk -F' ' '{print $3}')
#                                     RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
#                                     echo "${RECORD}" | tee -a ${LOG_FILE}
#                                     echo "${RECORD}" >> ${DIR}/log.txt
#                                 fi
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
