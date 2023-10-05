#!/bin/bash

cd ..

# custom config
DATA=./data
TRAINER=UPLTrainer
TIME=$(date +"%m-%d_%H-%M-%S")   # get current time stamp

DATASET=$1
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
# PLL_partial_rate=$11
use_PLL=True    #$10
TAG=$TIME # log tag (multiple_models_random_init or rn50_random_init)
LOG_FILE="logs_scripts/log_${TIME}_${DATASET}.txt"
#---------------------------individual settings: ---------------------------
USE_REGULAR=False     #add 2
declare -a BETAS=(0.0)

for SEED in {1..3}
do
    for PLL_partial_rate in 0.1 0.3
    do
        for loss_type in 'CE' 'cc' 'rc_cav'
        do
            for BETA in "${BETAS[@]}"
            do
                common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}"
                DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_${TAG}/SEED${SEED}/${common_id}
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
                    TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate}
                    
                    ACCURACY=$(grep -A4 'Do evaluation on test set' ${DIR}/log.txt | grep 'accuracy:' | awk -F' ' '{print $3}')
                    RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
                    echo "${RECORD}" | tee -a ${LOG_FILE}
                    echo "${RECORD}" >> ${DIR}/log.txt
                fi
            done
        done
    done
done

#---------------------------individual settings: ---------------------------
USE_REGULAR=True     #add 2
declare -a BETAS=(0.1 0.2 0.3 0.5)

for SEED in {1..3}
do
    for PLL_partial_rate in 0.1 0.3
    do
        for loss_type in 'CE' 'cc' 'rc_cav'
        do
            for BETA in "${BETAS[@]}"
            do
                common_id="data-${DATASET}_model-${CFG}_shots-${SHOTS}_nctx-${NCTX}_ctp-${CTP}_fp-${FP}_usePLL${use_PLL}-${PLL_partial_rate}_loss-${loss_type}_seed-${SEED}_beta-${BETA}"
                DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_${TAG}/${common_id}
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
                    TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate}
                    
                    ACCURACY=$(grep -A4 'Do evaluation on test set' ${DIR}/log.txt | grep 'accuracy:' | awk -F' ' '{print $3}')
                    RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
                    echo "${RECORD}" | tee -a ${LOG_FILE}
                    echo "${RECORD}" >> ${DIR}/log.txt
                fi
            done
        done
    done
done



# for loss_type in 'rc+'
# do
#     for PLL_partial_rate in 0.1 0.3 0.5
#     do
#         for CONF_LOSS_TYPE in 'gce' 'ce' 'gce_rc' 'rc_rc' 'rc_cav'
#         do
#             for BETA in 0.05 0.075 0.1 0.125 1.5
#             do
#                 DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_EQULE_${CLASS_EQULE}_${TAG}/nctx${NCTX}_csc${CSC}_ctp${CTP}_fp${FP}_usePLL${use_PLL}${PLL_partial_rate}-${loss_type}${CONF_LOSS_TYPE}/seed${SEED}/beta${BETA}
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
#                     TRAINER.PLL.USE_PLL ${use_PLL} \
#                     TRAINER.PLL.PARTIAL_RATE ${PLL_partial_rate} \
#                     TRAINER.PLL.CONF_LOSS_TYPE ${CONF_LOSS_TYPE} \
#                     TRAINER.PLL.BETA ${BETA}
                    
#                     ACCURACY=$(grep '* accuracy:' ${DIR}/log.txt | awk '{print $3}')
#                     RECORD="id: nctx${NCTX}_csc${CSC}_ctp${CTP}_fp${FP}_usePLL${use_PLL}${PLL_partial_rate}-${loss_type}${CONF_LOSS_TYPE}/seed${SEED}/beta${BETA} ----> test * accuracy: ${ACCURACY}"
#                     echo "${RECORD}" | tee -a ${LOG_FILE}
#                 fi
#             done
#         done
#     done
# done



#  collect all the result in the log.txt files in the corresponding file path, and print to show

# for SEED in {1..4}
# do  
#     DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_EQULE_${CLASS_EQULE}_${TAG}/nctx${NCTX}_csc${CSC}_ctp${CTP}_fp${FP}_usePLL${use_PLL}${PLL_partial_rate}-${loss_type}/seed${SEED}
#     if [ -d "$DIR" ]; then
#         echo "Results are available in ${DIR}. Skip this job"
#     else
#         echo "Run this job and save the output to ${DIR}"
#         python upl_train.py \
#         --root ${DATA} \
#         --seed ${SEED} \
#         --trainer ${TRAINER} \
#         --dataset-config-file configs/datasets/${DATASET}.yaml \
#         --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#         --output-dir ${DIR} \
#         --num-fp ${FP} \
#         TRAINER.UPLTrainer.N_CTX ${NCTX} \
#         TRAINER.UPLTrainer.CSC ${CSC} \
#         TRAINER.UPLTrainer.CLASS_TOKEN_POSITION ${CTP} \
#         DATASET.NUM_SHOTS ${SHOTS} \
#         DATASET.CLASS_EQULE ${CLASS_EQULE}
#     fi
# done