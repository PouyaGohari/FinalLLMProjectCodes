import numpy as np
import random
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import login

# --- Updated Imports ---
from utils.arg_parser import routing_analysis_arg_parser
from utils.config import *
# Use the new, refactored analyzer class and its VizMode enum
from routing_analyzer import RoutingAnalyzer, VizMode 

from peft import (
    ArrowConfig,
    create_arrow_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def model_and_tokenizer(args):
    """Initializes and returns the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, padding_side='right', model_max_length=MAX_LENGTH, cache_dir=args.local_dir
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, quantization_config=bnb_config, cache_dir=args.local_dir, device_map="auto"
    )
    return model, tokenizer

if __name__=='__main__':
    args = routing_analysis_arg_parser()
    set_seed(args.seed)
    login(token=args.hf_token)

    # --- 1. Initial Setup (done once) ---
    os.makedirs("my_analysis", exist_ok=True)
    base_model, tokenizer = model_and_tokenizer(args)
    
    # Instantiate the new analyzer
    analyzer = RoutingAnalyzer()

    # Configure the model to use the analyzer's callback method
    arrow_config = ArrowConfig(
        top_k=args.top_k,
        router_temperature=args.router_temperature,
        use_gks=args.gks,
        top_k_callback=args.top_k_callback,
        routing_callback=analyzer.log_routing_decision, # Updated callback
        arrow_k_max=2,
        remove_intruders=True,
        intruder_threshold=0.5
    )
    model = create_arrow_model(
        base_model=base_model,
        task_specific_adapter_paths=list(CLUSTER_NAMES.values()),
        general_adapter_paths=args.language_experts,
        arrow_config=arrow_config,
    )
    model.eval()
    
    print("⏳ Loading the full dataset...")
    full_dataset = load_dataset("TahaBa/flan-routing-MoE-dataset", split="validation")

    def tokenize_function(examples):
        return tokenizer(
            examples["source"], padding="max_length", truncation=True, max_length=MAX_LENGTH
        )
    
    # This dictionary will store the results of each analysis run
    all_analysis_results = {}

    # --- 2. Main Loop: Log and Analyze data for each template ---
    for template_id in range(10):
        print(f"\n{'='*20} Processing Template ID: {template_id} {'='*20}")

        filtered_dataset = full_dataset.filter(lambda example: example['template_idx'] == template_id)
        if len(filtered_dataset) == 0:
            print(f"⚠️ No samples found for template_idx {template_id}. Skipping.")
            continue
        print(f"Found {len(filtered_dataset)} samples.")
        
        tokenized_dataset = filtered_dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size)

        # --- 2a. Inference Loop to capture routing data ---
        print("🚀 Starting inference...")
        num_batches_to_process = 256
        for i, batch in enumerate(tqdm(dataloader, desc=f"Template {template_id} Batches")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Log inputs *before* the forward pass. This prepares the analyzer for the upcoming decisions.
            analyzer.log_forward_pass(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            with torch.no_grad():
                # The model's forward pass will trigger the `log_routing_decision` callback
                outputs = model(**batch)

            if i + 1 >= num_batches_to_process:
                break
        
        print("✅ Inference complete for this template.")

        # --- 2b. Analyze the logged data for the current template ---
        print("💾 Analyzing routing data...")
        # try:
        result = analyzer.analyze(name=f"template_{template_id}")
        all_analysis_results[result.name] = result
        # except ValueError as e:
        #     print(f"Could not analyze data for template {template_id}: {e}")

        # --- 2c. Clear the analyzer's logs to prepare for the next template ---
        analyzer.clear()

    print("\n🎉 All templates processed successfully!")

    # --- 3. Visualization: Use the collected results to generate plots ---
    if not all_analysis_results:
        print("No analysis results were generated. Skipping visualization.")
    else:
        print("📊 Generating comparison visualizations...")

        # Combined layer distribution plot
        save_path = "my_analysis/comparison_plots"
        RoutingAnalyzer.visualize(
            mode=VizMode.LAYER_DISTRIBUTION,
            results=all_analysis_results, 
            save_path=save_path
        )

        # Overall similarity of distributions
        RoutingAnalyzer.visualize(
            mode=VizMode.SIMILARITY_OVERALL,
            results=all_analysis_results,
            save_path="my_analysis/similarity_plots"
        )

        num_layers_to_visualize = len(next(iter(all_analysis_results.values())).per_layer_dist)
        for layer_to_check in range(num_layers_to_visualize):
            RoutingAnalyzer.visualize(
                mode=VizMode.SIMILARITY_PER_LAYER,
                results=all_analysis_results,
                layer_idx=layer_to_check,
                save_path="my_analysis/similarity_plots"
            )

        print("✅ All comparison plots have been generated.")

        # --- Optional: Generate detailed plots for a specific result ---
        # Example for the first analyzed template
        # first_result_name = next(iter(all_analysis_results.keys()))
        # first_result_dict = {first_result_name: all_analysis_results[first_result_name]}
        #
        # print(f"\n📊 Generating detailed plots for '{first_result_name}'...")
        # RoutingAnalyzer.visualize(
        #     mode=VizMode.LAYER_DISTRIBUTION,
        #     results=first_result_dict,
        #     save_path=f"my_analysis/detailed_plots/{first_result_name}"
        # )
        #
        # for i in range(num_layers_to_visualize):
        #     RoutingAnalyzer.visualize(
        #         mode=VizMode.TOKEN_WEIGHTS,
        #         results=first_result_dict,
        #         batch_idx=0,
        #         sample_idx=0,
        #         layer_idx=i,
        #         save_path=f"my_analysis/detailed_plots/{first_result_name}"
        #     )