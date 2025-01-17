#!/bin/bash

cd ..

# custom config
DATA=./data
TRAINER=UPLTrainer
exp_ID="10.23-test_rc_refine_ep100_refillloop_1"    #NOTE +time
# TODO: 
#1. change oonf clean threshold and set safe factor and range
#10.19-test_cc_refine_ep100_safe&clean2
# rememberi it use tfm_test now 10.19


# TAG=$(date +"%m-%d_%H-%M-%S")   # get current time stamp
TAG="${exp_ID}"      

PLL_partial_rate=$1      #ssucf101 ssoxford_pets
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
#---------------------------individual settings: ---------------------------
# USE_REGULAR=False     #add 2
# USE_LABEL_FILTER=True
# declare -a LOSS_MINs=(0.0 0.07 0.1 0.15)
# # declare -a BETAS=(0.0)
# TEMPERATURE=1.0
# BETA=0.0

# for SEED in {1..3}
# do
#     for DATASET in 'ssoxford_pets' 'ssucf101' 
#     do
#         LOG_FILE="logs_scripts/log_${TAG}_${DATASET}.txt"
#         for loss_type in 'rc_rc' 'cc'
#         do
#             for LOSS_MIN in "${LOSS_MINs[@]}"
#             do
#                 common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}_FILT-${USE_LABEL_FILTER}_T-${TEMPERATURE}_lossMin-${LOSS_MIN}"
#                 DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots-${TAG}/SEED${SEED}/${common_id}
#                 if [ -d "$DIR" ]; then
#                     echo "Results are available in ${DIR}. Skip this job"
#                 else
#                     echo "Run this job and save the output to ${DIR}"
#                     python upl_train.py \
#                     --root ${DATA} \
#                     --seed ${SEED} \
#                     --trainer ${TRAINER} \
#                     --dataset-config-file configs/datasets/${DATASET}.yaml \
#                     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#                     --output-dir ${DIR} \
#                     --num-fp ${FP} \
#                     --loss_type ${loss_type} \
#                     TRAINER.UPLTrainer.N_CTX ${NCTX} \
#                     TRAINER.UPLTrainer.CSC ${CSC} \
#                     TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
#                     DATASET.NUM_SHOTS ${SHOTS} \
#                     DATASET.CLASS_EQULE ${CLASS_EQULE} \
#                     TEST.FINAL_MODEL best_val \
#                     TRAINER.PLL.BETA ${BETA} \
#                     TRAINER.PLL.USE_REGULAR ${USE_REGULAR} \
#                     TRAINER.PLL.USE_PLL ${use_PLL} \
#                     TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate} \
#                     TRAINER.PLL.LOSS_MIN ${LOSS_MIN} \
#                     TRAINER.PLL.USE_LABEL_FILTER ${USE_LABEL_FILTER} \
#                     TRAINER.PLL.TEMPERATURE ${TEMPERATURE}
                    
#                     ACCURACY=$(grep -A4 'Do evaluation on test set' ${DIR}/log.txt | grep 'accuracy:' | awk -F' ' '{print $3}')
#                     RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
#                     echo "${RECORD}" | tee -a ${LOG_FILE}
#                     echo "${RECORD}" >> ${DIR}/log.txt
#                 fi
#             done
#         done
#     done
# done

#---------------------------individual settings: ---------------------------
USE_REGULAR=False     #add 2
USE_LABEL_FILTER=True
# declare -a BETAS=(0.0 0.1 0.2 0.3)
BETA=0.0
declare -a CONF_MOMNs=(1.0)
declare -a TOP_POOLs=(100)
# declare -a MAX_POOLNUMs=(16 19)     
declare -a DATASETs=('ssucf101')
declare -a SAFT_FACTORs=(1.0)
# declare -a HALF_USE_Ws=(1.0)
HALF_USE_W=0.5
declare -a INIT_EPOCHs=(2 0 20)

if (( $(echo "$PLL_partial_rate == 0.1" | bc -l) )); then
    declare -a MAX_POOLNUMs=(4 6 8 10)  
elif (( $(echo "$PLL_partial_rate == 0.3" | bc -l) )); then
    declare -a MAX_POOLNUMs=(4 6 8 10)  
else 
    echo "Invalid rate for MAX_POOLNUMs, rate is: $rate"
fi


# declare -a SHRINK_FACTORs=(0.5 0.3 0.7)


for SEED in {1..3}
do
    for DATASET in "${DATASETs[@]}"
    do
        LOG_FILE="logs_scripts/log_${TAG}_${DATASET}.txt"
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
                            for INIT_EPOCH in "${INIT_EPOCHs[@]}"
                            do
                                common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}_FILT-${USE_LABEL_FILTER}_cMomn-${CONF_MOMN}_topP-${TOP_POOL}_MAXPOOL-${MAX_POOLNUM}_safeF-${SAFT_FACTOR}_initE-${INIT_EPOCH}"
                                DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots-${TAG}/SEED${SEED}/${common_id}
                                if [ -d "$DIR" ]; then
                                    echo "Results are available in ${DIR}. Skip this job"
                                else
                                    echo "Run this job and save the output to ${DIR}"
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
                                    TEST.FINAL_MODEL best_val \
                                    TRAINER.PLL.BETA ${BETA} \
                                    TRAINER.PLL.USE_REGULAR ${USE_REGULAR} \
                                    TRAINER.PLL.USE_PLL ${use_PLL} \
                                    TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate} \
                                    TRAINER.PLL.USE_LABEL_FILTER ${USE_LABEL_FILTER} \
                                    TRAINER.PLL.CONF_MOMN ${CONF_MOMN} \
                                    TRAINER.PLL.MAX_POOLNUM ${MAX_POOLNUM} \
                                    TRAINER.PLL.SAFE_FACTOR ${SAFT_FACTOR} \
                                    TRAINER.PLL.HALF_USE_W ${HALF_USE_W} \
                                    TRAINER.PLL.INIT_EPOCH ${INIT_EPOCH} \
                                    TRAINER.PLL.TOP_POOLS ${TOP_POOL}
                                    
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
