import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score
from data_handler.test_datasets import (
    read_test_dataset,
)
from peft import create_arrow_model, ArrowConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loglike_loss(logits, labels, reduction="none"):
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # reshape back
    if reduction == "none":
        loss = loss.view((bs, -1))
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss
    
    return loss.float()  # Convert to float32 before returning
    
    
multi_choice_datasets = ['piqa', 'boolq', 'swag', 'hswag', 'arc-easy', 'arc-challenge',
                         'wg', 'oqa', 'bbh', 'xnli', 'mmlu']


def evaluate_on_multi_choice_batched(
    eval_dataset, model, tokenizer, ds_name, labels, predictions, args,
    batch_size=32, max_length=512, device="cuda"
):
    # Local import to mirror your original function
    from data_handler.test_datasets import extract_input_content, create_multi_choice_options, extract_multi_choice_target_index

    model.eval()

    for start in tqdm(range(0, len(eval_dataset), batch_size), total=(len(eval_dataset)+batch_size-1)//batch_size):
        rows = [eval_dataset[i] for i in range(start, min(start + batch_size, len(eval_dataset)))]

        # Build the flattened option texts for this batch
        all_texts = []
        options_per_sample = []     # number of options for each sample
        ctx_lens_per_option = []    # context length replicated per option

        for row in rows:
            options = create_multi_choice_options(row, ds_name)
            options_per_sample.append(len(options))

            # compute context length once per sample (align with your -1 shift)
            content = extract_input_content(ds_name, row)
            context_prompt = f"Instruct: {content}.\nOutput: "
            ctx_len = len(tokenizer.encode(context_prompt)) - 1

            all_texts.extend(options)
            ctx_lens_per_option.extend([ctx_len] * len(options))

            # collect gold label
            labels.append(extract_multi_choice_target_index(row, ds_name))

        # Tokenize all options in one go
        tokenized = tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        # Create masked labels: ignore context and padding
        masked_labels = tokenized["input_ids"].clone()
        for i, ctx_len in enumerate(ctx_lens_per_option):
            masked_labels[i, :ctx_len] = -100
        masked_labels[tokenized["attention_mask"] == 0] = -100

        with torch.no_grad():
            logits = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"]
            ).logits
            # per-sequence losses
            losses = compute_loglike_loss(logits, masked_labels, reduction="none").detach().cpu()

        # Reduce per sample (argmin across its options)
        idx = 0
        for n_opt in options_per_sample:
            pred = torch.argmin(losses[idx:idx + n_opt]).item()
            predictions.append(pred)
            idx += n_opt

    print(f"Accuracy for dataset {args['dataset_name']} is: {accuracy_score(labels, predictions)}")



args = {
        "model_name": "microsoft/phi-2",
        "model_max_length": 2048,
        "dataset_name": "arc-easy",
    }

if __name__ == "__main__":

    # Loading the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(
        args["model_name"],
        use_fast=True, 
        padding_side='right', 
        model_max_length=args["model_max_length"],
    )

    tokenizer.add_special_tokens({'pad_token': '|pad_id|'})
    
    # Quantisation config
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    
    # Loading the model
    base_model = AutoModelForCausalLM.from_pretrained(
        args["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # quantization_config=bnb_config,
    )

    # Creating the Arrow config
    arrow_config = ArrowConfig(
        top_k = 3,
        router_temperature = 1.0,
        use_gks = False,
        rng_seed=42,
    )
    
    # Task-specific and General adapters paths.
    task_specific_adapter_paths = [f"AliEdalat/phi2_mbc_peft/cluster_{i}" for i in range(1,11)]
    general_adapter_paths = [
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langen/checkpoint-17",
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langfr/checkpoint-35",
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langger/checkpoint-17"
    ]
    
    # Creating the Arrow model
    model = create_arrow_model(
        base_model = base_model,
        task_specific_adapter_paths = task_specific_adapter_paths,
        general_adapter_paths = general_adapter_paths,
        arrow_config = arrow_config,
    )
    
    # print(model)
    
    # model.merge_and_unload()
    # model.unload()

    # Evaluation
    routing_test_dataset = read_test_dataset(args["dataset_name"])
    # routing_test_dataset = routing_test_dataset.train_test_split(test_size=400, seed=args.seed)['test']
    print(f"{args['dataset_name']} is loaded with size: {len(routing_test_dataset)}.")
    
    # Loading the model with 

    labels, predictions = [], []
    with torch.no_grad():
        if args["dataset_name"] in multi_choice_datasets:
            evaluate_on_multi_choice_batched(
                routing_test_dataset, model, tokenizer, args["dataset_name"],
                labels, predictions, args,
                batch_size=64,   # tune this
                max_length=512,  # tune if options are long
                device="cuda"
            )