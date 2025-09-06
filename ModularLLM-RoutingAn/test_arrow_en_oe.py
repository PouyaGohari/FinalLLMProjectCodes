import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import (
    PeftModel,
    LoraConfig,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from data_handler.flan_functions import (
    apply_preprocessing,
    create_message_column_for_test
)
from data_handler.metrics import compute_generation_metrics


args = {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "model_max_length": 2048,
        "expert_repo_path": "AliEdalat/cluster_experts_phi3",
        "dataset_name": "andersonbcdefg/supernatural-instructions-2m",
        "data_portion": 0.0001,
        "batch_size": 4,
    }

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        args["model_name"],
        use_fast=True, 
        padding_side='right', 
        model_max_length=args["model_max_length"],
    )
    
    # Loading the model
    model = AutoModelForCausalLM.from_pretrained(
        args["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Loading the experts on the model (note that the )
    initial_cluster = "cluster0"
    model = PeftModel.from_pretrained(
        model, 
        model_id=args['expert_repo_path'], 
        adapter_name=initial_cluster, 
        subfolder=initial_cluster,
    )

    for i in range(1, 10):
        cluster_name = f"cluster{i}"
        model.load_adapter(
            model_id=args['expert_repo_path'], 
            adapter_name=cluster_name, 
            subfolder=cluster_name,
        )
    
    # 1) Define the router’s config:
    router_cfg = LoraConfig(
        r = 2,       # rank of router adapter (dummy)
        use_arrow = True,    # <— turns on Arrow routing instead of vanilla LoRA
        arrow_expert_num = 10,       # total number of experts you loaded
        arrow_top_k = 3,       # “k” in top-k arrow routing
        arrow_router_temperature = 1.0,
        target_modules = ["qkv_proj", "o_proj"],
        use_gks = True,
        le_names = ["le_en", "le_fr", "le_de"],
        ts_names = [f"cluster{i}" for i in range(10)]
    )

    # 2) Create the adapter “router” (weights are randomly init’d LoRA mats,
    #    but will never actually be applied because use_arrow=True):
    model.add_adapter(          
        adapter_name="router",
        peft_config=router_cfg,
    )

    # 3) Tell the model “hey, from now on use the ‘router’ adapter”:
    model.set_adapter("router")

    print("Experts are loaded successfully.")
    print(model)

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, truncation=True, padding=True)
    
    routing_test_dataset = load_dataset(args["dataset_name"])['train']
    routing_test_dataset = routing_test_dataset if args["data_portion"] == 1.0 \
        else routing_test_dataset.train_test_split(test_size=args["data_portion"])['test']
    routing_test_dataset = routing_test_dataset.rename_columns({"prompt": "source", "response": "target"})
    routing_test_dataset = routing_test_dataset.map(create_message_column_for_test)
    routing_test_dataset = routing_test_dataset.map(
        lambda sample:
        {'text': pipe.tokenizer.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=True)}
    )
    
    test_dataloader = DataLoader(routing_test_dataset, batch_size=args["batch_size"])
    references, predictions = [], []
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        # Calling the model's forward path to apply Arrow Routing
        # tokenised_batch = tokenizer(batch['text'], return_tensors="pt", truncation=True, padding=True).to('cuda')
        # model(**tokenised_batch)

        
        
        
        
        

        # Generate the answer using the new adapter
        outputs = pipe(batch['text'], max_new_tokens=100)
        preds = [output[0]['generated_text'].split("<|assistant|>\n")[1].strip() for output in outputs]
        # print(preds)

        references.extend(batch['target'])
        predictions.extend(preds)
    
    metrics = compute_generation_metrics(references, predictions)
    print('=' * 100)
    print('BLEU:', metrics['bleu'])
    print('ROUGE:', metrics['rouge'])
    print('=' * 100)
