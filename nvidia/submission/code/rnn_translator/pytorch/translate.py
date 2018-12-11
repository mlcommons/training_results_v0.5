#!/usr/bin/env python
import logging
import argparse
import warnings
from ast import literal_eval

import torch
import torch.distributed as dist

from seq2seq.models.gnmt import GNMT
from seq2seq.inference.inference import Translator
from seq2seq.data.dataset import TextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.utils import setup_logging


def parse_args():
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(description='GNMT Translate',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    dataset = parser.add_argument_group('data setup')
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en/',
                         help='path to directory with training/validation data')
    dataset.add_argument('-i', '--input', required=True,
                         help='full path to the input file (tokenized)')
    dataset.add_argument('-o', '--output', required=True,
                         help='full path to the output file (tokenized)')
    dataset.add_argument('-r', '--reference', default=None,
                         help='full path to the reference file (for sacrebleu)')
    dataset.add_argument('-m', '--model', required=True,
                         help='full path to the model checkpoint file')
    # parameters
    params = parser.add_argument_group('inference setup')
    params.add_argument('--batch-size', default=128, type=int,
                        help='batch size per GPU')
    params.add_argument('--beam-size', default=5, type=int,
                        help='beam size')
    params.add_argument('--max-seq-len', default=80, type=int,
                        help='maximum generated sequence length')
    params.add_argument('--len-norm-factor', default=0.6, type=float,
                        help='length normalization factor')
    params.add_argument('--cov-penalty-factor', default=0.1, type=float,
                        help='coverage penalty factor')
    params.add_argument('--len-norm-const', default=5.0, type=float,
                        help='length normalization constant')
    # general setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp16', choices=['fp32', 'fp16'],
                         help='arithmetic type')

    bleu_parser = general.add_mutually_exclusive_group(required=False)
    bleu_parser.add_argument('--bleu', dest='bleu', action='store_true',
                             help='compares with reference and computes BLEU \
                             (use \'--no-bleu\' to disable)')
    bleu_parser.add_argument('--no-bleu', dest='bleu', action='store_false',
                             help=argparse.SUPPRESS)
    bleu_parser.set_defaults(bleu=True)

    batch_first_parser = general.add_mutually_exclusive_group(required=False)
    batch_first_parser.add_argument('--batch-first', dest='batch_first',
                                    action='store_true',
                                    help='uses (batch, seq, feature) data \
                                    format for RNNs')
    batch_first_parser.add_argument('--seq-first', dest='batch_first',
                                    action='store_false',
                                    help='uses (seq, batch, feature) data \
                                    format for RNNs')
    batch_first_parser.set_defaults(batch_first=True)

    cuda_parser = general.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true',
                             help='enables cuda (use \'--no-cuda\' to disable)')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                             help=argparse.SUPPRESS)
    cuda_parser.set_defaults(cuda=True)

    cudnn_parser = general.add_mutually_exclusive_group(required=False)
    cudnn_parser.add_argument('--cudnn', dest='cudnn', action='store_true',
                              help='enables cudnn (use \'--no-cudnn\' to disable)')
    cudnn_parser.add_argument('--no-cudnn', dest='cudnn', action='store_false',
                              help=argparse.SUPPRESS)
    cudnn_parser.set_defaults(cudnn=True)

    general.add_argument('--print-freq', '-p', default=1, type=int,
                         help='print log every PRINT_FREQ batches')

    # distributed support
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                             help='url used to set up distributed training')

    args = parser.parse_args()

    if args.bleu and args.reference is None:
        parser.error('--bleu requires --reference')

    return args


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.

    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.

    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def main():
    """
    Launches translation (inference).
    Inference is executed on a single GPU, implementation supports beam search
    with length normalization and coverage penalty.
    """
    args = parse_args()

    # initialize distributed backend
    distributed = args.world_size > 1
    if distributed:
        backend = 'nccl' if args.cuda else 'gloo'
        dist.init_process_group(backend=backend, rank=args.rank,
                                init_method=args.dist_url,
                                world_size=args.world_size)
    setup_logging()
    logging.info(f'Run arguments: {args}')

    if args.cuda:
        torch.cuda.set_device(args.rank)
    if not args.cuda and torch.cuda.is_available():
        warnings.warn('cuda is available but not enabled')
    if args.math == 'fp16' and not args.cuda:
        raise RuntimeError('fp16 requires cuda')
    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # load checkpoint and deserialize to CPU (to save GPU memory)
    checkpoint = torch.load(args.model, map_location={'cuda:0': 'cpu'})

    # build GNMT model
    tokenizer = Tokenizer()
    tokenizer.set_state(checkpoint['tokenizer'])
    vocab_size = tokenizer.vocab_size
    model_config = dict(vocab_size=vocab_size, math=checkpoint['config'].math,
                        **literal_eval(checkpoint['config'].model_config))
    model_config['batch_first'] = args.batch_first
    model = GNMT(**model_config)

    state_dict = checkpoint['state_dict']
    if checkpoint_from_distributed(state_dict):
        state_dict = unwrap_distributed(state_dict)

    model.load_state_dict(state_dict)

    if args.math == 'fp32':
        dtype = torch.FloatTensor
    if args.math == 'fp16':
        dtype = torch.HalfTensor

    model.type(dtype)
    if args.cuda:
        model = model.cuda()
    model.eval()

    # construct the dataset
    test_data = TextDataset(src_fname=args.input,
                            tokenizer=tokenizer,
                            sort=False)

    # build the data loader
    test_loader = test_data.get_loader(batch_size=args.batch_size,
                                       batch_first=args.batch_first,
                                       shuffle=False,
                                       pad=True,
                                       num_workers=0,
                                       drop_last=False)

    # build the translator object
    translator = Translator(model=model,
                            tokenizer=tokenizer,
                            loader=test_loader,
                            beam_size=args.beam_size,
                            max_seq_len=args.max_seq_len,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            cuda=args.cuda,
                            print_freq=args.print_freq,
                            dataset_dir=args.dataset_dir)

    # execute the inference
    translator.run(calc_bleu=args.bleu, eval_path=args.output,
                   reference_path=args.reference, summary=True)

if __name__ == '__main__':
    main()
