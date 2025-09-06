##Note that this is peft from the written in Guidance file
from peft import ArrowConfig, create_arrow_model, PeftModel
from typing import (
    List,
    Optional
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch

def apply_arrow_or_gks(
        base_model_name:str,
        cluster_names:List[str],
        arrow_top_k:int,
        arrow_router_temperature:float,
        gks:Optional[bool],
        ts_repo_id:str,
        language_experts:Optional[List[str]],
        gen_repo_id:str
   ) -> PeftModel:
    """
    This function will either apply arrow routing mechanism to adapters or general knowledge subtraction in conjunction with arrow router based on the gks parameter.
    :param base_model_name: The model name or path for containing safetensors.
    :param cluster_names: A dictionary for clusters(tasks-specific lora-adapters) where keys is the adapter names and values are the paths for corresponding adapters.
    :param arrow_top_k: The top k of loras for routing among.
    :param arrow_router_temperature: The temperature that applies to softmax of arrow.
    :param gks: If applying general knowledge subtraction.
    :param ts_repo_id: The repo id for TS.
    :param language_experts: A dictionary where keys are the name of adapters(e.g, English or French) and values contains the paths for corresponding adapters.
    :param gen_repo_id: The repo id for general adapter.
    :return:
    A Peft Model where all adapters applied with respect to the method.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, quantization_config=bnb_config , cache_dir="models")
    arrow_config = ArrowConfig(
        top_k = arrow_top_k,
        router_temperature = arrow_router_temperature,
        use_gks = gks,
    )
    model = create_arrow_model(
        base_model = base_model,
        task_specific_adapter_paths = cluster_names,
        ts_repo_id = ts_repo_id,
        general_adapter_paths = language_experts,
        arrow_config = arrow_config,
        gen_repo_id = gen_repo_id
    )
    return model