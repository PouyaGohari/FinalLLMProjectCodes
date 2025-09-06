from pathlib import Path

import torch
import gc
import logging
import numpy as np
import pandas as pd

from huggingface_hub import login, hf_hub_download

from utils.CKA import apply_cka
from utils.MyConfig import *
from utils.MyArgParser import arg_parser
from utils.apply_arrow_gks import apply_arrow_or_gks

from utils.util import (
    set_seed,
    model_and_tokenizer,
    load_general_dataset,
    get_samples,
    create_torch_dataset,
    dataloader
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

if __name__=='__main__':
    args = arg_parser()
    set_seed(args.seed)
    login(token=args.hf_token)

    general_model, tokenizer = model_and_tokenizer(model_name=MODEL_NAME)
    if "LE_dataset_5token_pred" in args.dataset_path:
        data_file=DATA_FILE
        dataset = load_general_dataset(path=args.dataset_path, data_file=data_file)
    elif "le_datasets_from_machine" in args.dataset_path:
        train_path = hf_hub_download(repo_id=args.dataset_path, filename=args.data_file_train, repo_type="dataset")
        test_path = hf_hub_download(repo_id=args.dataset_path, filename=args.data_file_test, repo_type="dataset")

        data_file = {'train': train_path, 'test': test_path}
        dataset = load_general_dataset(path="json", data_file=data_file)
    else:
        data_file = {'train': args.data_file_train, 'test': args.data_file_test}
        dataset = load_general_dataset(path=args.dataset_path, data_file=data_file)

    sub_dataset = get_samples(your_dataset=dataset['test'], n_samples=args.n_samples, seed=args.seed)

    print(f"------------- Subsampling from {args.dataset_path} has been finished and enhanced model is starting to be processed------------------------")

    enhanced_model = apply_arrow_or_gks(
        base_model_name=args.base_model_name,
        cluster_names=list(CLUSTER_NAMES.values()),
        arrow_top_k=args.top_k,
        arrow_router_temperature=args.temperature,
        gks=args.gks,
        language_experts=list(LANGUAGE_EXPERTS.values()),
    )

    my_generator = torch.Generator()
    my_generator.manual_seed(args.seed)
    print(sub_dataset[0])

    compatible_dataset = create_torch_dataset(sub_dataset, tokenizer, args.use_attention)
    print(compatible_dataset[0])
    print(compatible_dataset[0]['input_ids'].shape)

    my_dataloader = dataloader(
        compatible_dataset=compatible_dataset,
        batch=args.batch,
        generator=my_generator
    )


    print(f"------------- Starting to apply cka -------------")

    if args.gks:
        second_model_name = "GenKowlSub"
    else:
        second_model_name = "Arrow"
    layers_of_interest_first_model = [f"model.layers.{i}" for i in range(32)]
    layers_of_interest_peft = [f"base_model.model.model.layers.{i}" for i in range(32)]


    exported_data = apply_cka(
        first_loader=my_dataloader,
        base_model=general_model,
        base_model_layers=layers_of_interest_first_model,
        enhanced_model_layers=layers_of_interest_peft,
        enhanced_model=enhanced_model,
        first_model_name="Baseline",
        second_model_name=second_model_name,
        export_data=args.export_data,
        show_plot=args.show_plot,
        device=args.device,
    )

    torch.cuda.empty_cache()
    gc.collect()
    result = [
        {
            'first_layers': layers_of_interest_first_model,
            'second_layers': layers_of_interest_peft,
            'exported_data_path': args.save_path,
        }
    ]
    out_file = Path(args.save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, exported_data)
    df = pd.DataFrame(result)
    df.to_csv(out_file, index=False)