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

from routing_analyzer import RoutingAnalyzer, VizMode


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loglike_loss(logits, labels, reduction="none"):
    # (Your original function, no changes needed)
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    if reduction == "none":
        loss = loss.view((bs, -1))
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss
    return loss.float()
    
    
def evaluate_on_multi_choice_batched(
    eval_dataset, model, tokenizer, ds_name,
    analyzer: RoutingAnalyzer = None,
    batch_size=32, max_length=512, device="cuda"
):
    from data_handler.test_datasets import extract_input_content, create_multi_choice_options, extract_multi_choice_target_index

    model.eval()
    labels, predictions = [], []

    for start in tqdm(range(0, len(eval_dataset), batch_size), total=(len(eval_dataset)+batch_size-1)//batch_size, desc=f"Eval {ds_name}"):
        rows = [eval_dataset[i] for i in range(start, min(start + batch_size, len(eval_dataset)))]
        all_texts, options_per_sample, ctx_lens_per_option = [], [], []

        for row in rows:
            options = create_multi_choice_options(row, ds_name)
            options_per_sample.append(len(options))
            content = extract_input_content(ds_name, row)
            context_prompt = f'<|user|>\n{content}<|end|>\n<|assistant|>'
            ctx_len = len(tokenizer.encode(context_prompt)) - 1
            all_texts.extend(options)
            ctx_lens_per_option.extend([ctx_len] * len(options))
            labels.append(extract_multi_choice_target_index(row, ds_name))

        tokenized = tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        masked_labels = tokenized["input_ids"].clone()
        for i, ctx_len in enumerate(ctx_lens_per_option):
            masked_labels[i, :ctx_len] = -100
        masked_labels[tokenized["attention_mask"] == 0] = -100

        with torch.no_grad():
            if analyzer:
                analyzer.log_forward_pass(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"]
                )
            
            logits = model(**tokenized).logits
            losses = compute_loglike_loss(logits, masked_labels, reduction="none").detach().cpu()

        idx = 0
        for n_opt in options_per_sample:
            pred = torch.argmin(losses[idx:idx + n_opt]).item()
            predictions.append(pred)
            idx += n_opt
    
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy for dataset {ds_name} is: {accuracy}")
    return accuracy


if __name__ == "__main__":
    args = {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "model_max_length": 2048,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"], use_fast=True, padding_side='right', model_max_length=args["model_max_length"])
    base_model = AutoModelForCausalLM.from_pretrained(args["model_name"], torch_dtype=torch.bfloat16, device_map="auto")

    task_specific_adapter_paths = [f"TahaBa/phi3-mini-clustered-flan/ts_expert_{i}" for i in range(10)]
    general_adapter_paths = [
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langen/checkpoint-17",
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langfr/checkpoint-35",
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langger/checkpoint-17"
    ]

    analyzer = RoutingAnalyzer()

    RESULTS_FILE = "analysis_results/mc_results.pkl"
    all_analysis_results = RoutingAnalyzer.load_state(RESULTS_FILE)

    # --- 2. Define Datasets and Experiments to Run ---
    datasets_to_run = ['piqa', 'boolq', 'swag', 'hswag', 'arc-easy', 'arc-challenge', 'wg', 'oqa', 'bbh', 'xnli', 'mmlu']
    # datasets_to_run = ['boolq', 'hswag', 'arc-easy', 'arc-challenge']
    # datasets_to_run = ['piqa', 'arc-easy', 'arc-challenge']
    
    experiment_configs = {
        "with_gks": {"top_k": 3, "router_temperature": 1.0, "use_gks": True},
        "without_gks": {"top_k": 3, "router_temperature": 1.0, "use_gks": False},
    }

    for dataset_name in datasets_to_run:
        for config_name, config in experiment_configs.items():
            run_name = f"{dataset_name}_{config_name}"
            
            if run_name in all_analysis_results:
                print(f"\n✅ Skipping experiment '{run_name}' as results are already loaded.")
                continue

            print(f"\n{'#'*25} LOADING DATASET: {dataset_name.upper()} {'#'*25}")
            try:
                routing_test_dataset = read_test_dataset(dataset_name)
                print(f"{dataset_name} is loaded with size: {len(routing_test_dataset)}.")
            except Exception as e:
                print(f"Could not load dataset {dataset_name}. Error: {e}. Skipping.")
                continue

            print(f"\n{'='*20} RUNNING EXPERIMENT: {run_name} {'='*20}")
            
            arrow_config = ArrowConfig(
                top_k=config["top_k"],
                router_temperature=config["router_temperature"],
                use_gks=config["use_gks"],
                routing_callback=analyzer.log_routing_decision,
                top_k_callback=3,
                remove_intruders=True,
                intruder_threshold=0.5
            )
            
            model = create_arrow_model(
                base_model=base_model,
                task_specific_adapter_paths=task_specific_adapter_paths,
                general_adapter_paths=general_adapter_paths,
                arrow_config=arrow_config,
            )
            
            evaluate_on_multi_choice_batched(
                routing_test_dataset, model, tokenizer, dataset_name,
                analyzer=analyzer,
                batch_size=8,
                max_length=512,
            )
            
            result = analyzer.analyze(name=run_name, temperature=config["router_temperature"])
            all_analysis_results[result.name] = result
            
            RoutingAnalyzer.save_state(all_analysis_results, RESULTS_FILE)
            analyzer.clear()

    # --- 4. Generate Final Analysis Plots ---
    if all_analysis_results:
        print(f"\n{'='*20} GENERATING ANALYSIS PLOTS {'='*20}")
        
        # Helper function to generate all three plot types
        def generate_plots(results_dict, subfolder_name):
            if not results_dict:
                print(f"No results for '{subfolder_name}', skipping plot generation.")
                return

            print(f"\n--- Generating plots for: {subfolder_name} ---")
            save_path = f"analysis_results/{subfolder_name}"
            
            # Layer Distribution Plot
            RoutingAnalyzer.visualize(
                mode=VizMode.LAYER_DISTRIBUTION,
                results=results_dict, 
                save_path=save_path
            )

            # Overall Similarity Plot
            RoutingAnalyzer.visualize(
                mode=VizMode.SIMILARITY_OVERALL,
                results=results_dict,
                save_path=save_path,
            )
            
            # Per-Layer Similarity Plots
            first_run_name = next(iter(results_dict.keys()))
            num_layers = len(results_dict[first_run_name].per_layer_dist)
            for i in range(num_layers):
                RoutingAnalyzer.visualize(
                    mode=VizMode.SIMILARITY_PER_LAYER,
                    results=results_dict,
                    layer_idx=i,
                    save_path=save_path,
                )

        # Filter results into groups
        gks_results = {name: res for name, res in all_analysis_results.items() if "with_gks" in name}
        no_gks_results = {name: res for name, res in all_analysis_results.items() if "without_gks" in name}

        generate_plots(gks_results, "gks_only_plots")
        generate_plots(no_gks_results, "no_gks_only_plots")
        generate_plots(all_analysis_results, "combined_plots")

        print(f"\n✅ All comparison plots have been generated in the 'analysis_results' directory.")






