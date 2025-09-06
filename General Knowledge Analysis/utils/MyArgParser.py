import argparse

from utils.MyConfig import *

def arg_parser():
    parser = argparse.ArgumentParser()
    ## Provide your token
    parser.add_argument("--hf_token", required=True,  type=str)
    ## For applying arrow or gks.
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--gks", action='store_true', help="Use general knowledge subtraction")
    parser.add_argument("--base_model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    ## For downloading general dataset like wiki dataset.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--use_attention", action='store_true')
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--data_file_train", default=DATA_FILE['train'], type=str)
    parser.add_argument("--data_file_test", default=DATA_FILE['test'], type=str)
    ## For applying cka
    parser.add_argument("--export_data", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()

def plot_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=None, type=str, required=True)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--normalized", action='store_true')
    parser.add_argument("--show_stats", action='store_true')
    parser.add_argument("--title", default=None, type=str)
    parser.add_argument("--xlabel", default=None, type=str)
    parser.add_argument("--ylabel", default=None, type=str)
    return parser.parse_args()

def perplexity_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True,  type=str)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--apply_arrow", action='store_true')
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--apply_gks", action='store_true')
    parser.add_argument("--data_file_train", default=DATA_FILE['train'], type=str)
    parser.add_argument("--data_file_test", default=DATA_FILE['test'], type=str)
    parser.add_argument("--save_path", default=None, type=str)
    return parser.parse_args()

def train_boolq_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True, type=str)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    return parser.parse_args()
