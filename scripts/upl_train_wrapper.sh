#!/bin/bash

while (( "$#" )); do
  case "$1" in
    --dataset)
      DATASET=$2
      shift 2
      ;;
    --cfg)
      CFG=$2
      shift 2
      ;;
    --ctp)
      CTP=$2
      shift 2
      ;;
    --nctx)
      NCTX=$2
      shift 2
      ;;
    --shots)
      SHOTS=$2
      shift 2
      ;;
    --csc)
      CSC=$2
      shift 2
      ;;
    --class-equle)
      CLASS_EQULE=$2
      shift 2
      ;;
    --fp)
      FP=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

CUDA_VISIBLE_DEVICES=0 ./upl_train.sh $DATASET $CFG $CTP $NCTX $SHOTS $CSC $CLASS_EQULE $FP
# now use: ./wrapper_script.sh --dataset ssucf101 --cfg rn50_ep50 --ctp end --nctx 16 --shots 16 --csc False --class-equle True --fp 0
