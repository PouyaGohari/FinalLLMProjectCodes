from argparse import ArgumentError
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.MyArgParser import perplexity_parser
from utils.MyConfig import *
from utils.util import (
    set_seed,
    model_and_tokenizer,
    load_general_dataset,
    get_samples,
    create_torch_dataset,
    dataloader,
)
import math
from utils.apply_arrow_gks import apply_arrow_or_gks
from tqdm import tqdm

from huggingface_hub import hf_hub_download, login

if __name__ == "__main__":
    args = perplexity_parser()
    login(args.hf_token)
    set_seed(args.seed)
    base_model, tokenizer = model_and_tokenizer(args.model_name)

    if args.apply_gks and not args.apply_arrow:
        raise  ArgumentError(
            "GKS works with Arrow but you set apply_gks and not mention apply_arrow in your argument."
        )
    if args.apply_arrow:
        model = apply_arrow_or_gks(
            base_model_name=args.model_name,
            cluster_names=list(CLUSTER_NAMES.values()) if 'Phi-3' in args.model_name else list(CLUSTER_NAMES2.values()),
            arrow_top_k=3,
            arrow_router_temperature=1.0,
            ts_repo_id= TS_REPO_ID if 'Phi-3' in args.model_name else TS_REPO_ID2,
            gks=args.apply_gks,
            language_experts=list(LANGUAGE_EXPERTS.values()) if 'Phi-3' in args.model_name else list(LANGUAGE_EXPERTS2.values()),
            gen_repo_id=GEN_REPO_ID if 'Phi-3' in args.model_name else GEN_REPO_ID2,
        )
    else:
        model = base_model

    if "LE_dataset_5token_pred" in args.dataset_name:
        data_file=DATA_FILE
        dataset = load_general_dataset(path=args.dataset_name, data_file=data_file)

    elif "le_datasets_from_machine" in args.dataset_name:
        train_path = hf_hub_download(repo_id=args.dataset_name, filename=args.data_file_train, repo_type="dataset")
        test_path = hf_hub_download(repo_id=args.dataset_name, filename=args.data_file_test, repo_type="dataset")

        data_file = {'train': train_path, 'test': test_path}
        dataset = load_general_dataset(path="json", data_file=data_file)

    else:
        data_file = {'train': args.data_file_train, 'test': args.data_file_test}
        dataset = load_general_dataset(path=args.dataset_name, data_file=data_file)

    sub_dataset = get_samples(your_dataset=dataset['test'], n_samples=args.n_samples, seed=args.seed)
    my_generator = torch.Generator()
    my_generator.manual_seed(args.seed)
    print(sub_dataset[0])

    compatible_dataset = create_torch_dataset(sub_dataset, tokenizer, True)
    print(compatible_dataset[0])
    print(compatible_dataset[0]['input_ids'].shape)

    my_dataloader = dataloader(
        compatible_dataset=compatible_dataset,
        batch=args.batch,
        generator=my_generator
    )

    with torch.no_grad():
        total_nll = 0.0
        total_tokens = 0
        perplexity_per_batch = []

        for batch in tqdm(my_dataloader, desc="Calculating Perplexity}"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            mask = attention_mask[:, 1:].contiguous()

            loss_tokens = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none"
            ).view_as(labels)
            loss_tokens = loss_tokens * mask
            total_nll += loss_tokens.sum().item()
            total_tokens += mask.sum().item()
            perplexity_per_batch.append(math.exp(loss_tokens.sum().item() / mask.sum().item()))
        perplexity = math.exp(total_nll / total_tokens)

    if args.save_path:
        ppl_per_batch = np.array(perplexity_per_batch)
        out_file = Path(args.save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_file, perplexity_per_batch)
    print(f"Perplexity for whole batch: {perplexity}")

