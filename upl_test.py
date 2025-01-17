import argparse
import torch
# torch.set_printoptions(profile="full")
import os
import random
import matplotlib.pyplot as plt
import numpy as np


from dassl.utils import setup_logger, set_random_seed, collect_env_info
from configs.upl_default_config.upl_default import get_cfg_default
from dassl.engine import build_trainer
from utils_temp.utils_ import become_deterministic

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import trainers.upltrainer
import trainers.hhzsclip


def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg,args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.UPLTrainer = CN()
    cfg.TRAINER.UPLTrainer.N_CTX = 16  # number of context vectors
    cfg.TRAINER.UPLTrainer.CSC = False  # class-specific context
    cfg.TRAINER.UPLTrainer.CTX_INIT = ""  # initialization words
    cfg.TRAINER.UPLTrainer.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.UPLTrainer.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.UPLTrainer.NUM_FP = args.num_fp  # #false positive training samples per class


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. hh config
    if args.hh_config_file:
        cfg.merge_from_file(args.hh_config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        # set_random_seed(cfg.SEED)
        become_deterministic(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True  # Enable CUDA benchmarking for faster training

    print_args(args, cfg)  # Print the provided arguments and configuration
    print('Collecting env info ...')  # Print a message indicating the collection of environment information
    print('** System info **\n{}\n'.format(collect_env_info()))  # Print the collected environment information

    trainer_list = []  # List to store the trainers
    for i in range(int(cfg.TRAINER.ENSEMBLE_NUM)):
        trainer = build_trainer(cfg)  # Build a trainer based on the configuration
        if args.model_dir:
            trainer.load_model_by_id(args.model_dir, epoch=args.load_epoch, model_id=i)  # Load a model from the specified directory
        trainer_list.append(trainer)  # Add the trainer to the list

        prob_end = []  # List to store probabilities
        results_dict = {}  # Dictionary to store results
        for SHOTS in [4]:  # Iterate over the specified number of shots
            print('SHOTS:', SHOTS, '\n\n\n\n\n')  # Print the number of shots
            for seed in range(1, 5):  # Iterate over the specified range of seeds
                try:
                    prob = torch.load('./analysis_results_test/{}/{}/fp{}/50_{}_{}_random_initend/test_logits.pt'.format(args.tag, cfg.DATASET.NAME, args.num_fp, seed, SHOTS))  # Load the probabilities from a file
                    prob_end.append(prob)  # Add the probabilities to the list
                except:
                    print('loss')  # Print a message indicating a loss occurred

            print(len(prob_end), ' shots ensemble')  # Print the number of shots in the ensemble
            prob_test = sum(prob_end) / len(prob_end)  # Calculate the average probability
            results_end = trainer_list[0].test_with_existing_logits(prob_test)  # Test the model with the calculated probabilities

            save_path = './analysis_results_test/{}/{}/fp{}/'.format(args.tag, cfg.DATASET.NAME, args.num_fp)  # Set the save path for the results
            with open(os.path.join(save_path, 'ensemble_accuracy_result.txt'), "w") as f:
                print('ensemble: {:.2f}'.format((results_end['accuracy'])), file=f)  # Write the ensemble accuracy result to a file
            print('ensemble: {:.2f}'.format((results_end['accuracy'])))  # Print the ensemble accuracy result
            results_dict[len(prob_end)] = results_end['accuracy']  # Store the ensemble accuracy result in the dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--hh-config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--num-fp',
        type=int,
        default=0,
        help='number of false positive training samples per class'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    parser.add_argument(
        '--tag',
        type=str,
        default='',
        help='tag for method'
    )
    args = parser.parse_args()
    main(args)
