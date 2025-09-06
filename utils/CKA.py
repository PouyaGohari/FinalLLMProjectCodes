from ckatorch import CKA
from ckatorch.cka import ModelInfo

import gc
from torch import nn
import inspect
from collections.abc import Callable
from ckatorch.core import cka_batch
import re

from tqdm import tqdm
from warnings import warn
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import (
    Any,
    Union,
    List,
    Optional
)
from ModularLLM.peft.src.peft import PeftModel
from transformers import AutoModelForCausalLM

class MyCKA(CKA):
    def __init__(self,
        first_model: Union[nn.Module, PeftModel, AutoModelForCausalLM],
        second_model: Union[nn.Module, PeftModel, AutoModelForCausalLM],
        layers: list[str],
        second_layers: list[str] | None = None,
        first_name: str | None = None,
        second_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # Check if no layers were passed
        if layers is None or len(layers) == 0:
            raise ValueError(
                "You can not pass 'None' or an empty list as layers. We suggest using 'first_model.named_modules()'"
                "in order to see which layers can be passed."
            )

        # Remove potential duplicates
        layers = sorted(set(layers), key=layers.index)

        # Check if to many layers were passed
        if len(layers) > 100:
            warn(
                f"You passed {len(layers)} distinct layers, which is way too high. Consider passing only those"
                f"layers whose features you are really interested about."
            )

        # Copy the first model's layers if they are not passed
        if second_layers is None or len(second_layers) == 0:
            second_layers = layers.copy()
        else:
            # Remove potential duplicates
            second_layers = sorted(set(second_layers), key=second_layers.index)

            # Check if too many layers were passed
            if len(second_layers) > 100:
                warn(
                    f"You passed {len(second_layers)} distinct layers for the second model, which is way too high."
                    f"Consider passing only those layers whose features you are really interested about."
                )

        # Dicts where the output of each layer (i.e.: the features) will be saved while using hooks
        self.first_features: dict[str, torch.Tensor] = {}
        self.second_features: dict[str, torch.Tensor] = {}

        # Insert a hook for each layer
        # layers, second_layers = self._insert_hooks(first_model, second_model, layers, second_layers)
        self.first_model = first_model.to(device)
        self.second_model = second_model.to(device)

        # Manage the models names
        first_name = first_name if first_name is not None else first_model.__repr__().split("(")[0]
        if first_model is second_model:
            second_name = first_name
        else:
            second_name = second_name if second_name is not None else second_model.__repr__().split("(")[0]
            if first_name == second_name:
                warn(f"Both models are called {first_name}. This may cause confusion when analyzing the results.")

        # Set up the models info
        first_name = re.sub("[^0-9a-zA-Z_]+", "", first_name.replace(" ", "_"))
        second_name = re.sub("[^0-9a-zA-Z_]+", "", second_name.replace(" ", "_"))
        self.first_model_info = ModelInfo(first_name, layers)
        self.second_model_info = ModelInfo(second_name, second_layers)

    def _hook(self, model: str, module_name: str, module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        pass

    def _insert_hooks(
            self,
            first_model: nn.Module,
            second_model: nn.Module,
            layers: list[str],
            second_layers: list[str],
    ) -> tuple[list[str], list[str]]:
        pass

    def __call__(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        f_extract: Callable[..., dict[str, torch.Tensor]] | None = None,
        f_args: dict[str, Any] | None = None,
    ) -> np.ndarray:

        self.first_model.eval()
        self.second_model.eval()

        with torch.no_grad():
            n = len(self.first_model_info.layers)
            m = len(self.second_model_info.layers)

            # Iterate through the dataset
            num_batches = len(dataloader)
            cka_matrices = []
            for epoch in tqdm(range(epochs), desc="| Computing CKA |", total=epochs):
                cka_epoch = torch.zeros(32, 32, device=self.device)
                for batch in tqdm(dataloader, desc=f"| Computing CKA epoch {epoch} |", total=num_batches, leave=False):
                    if f_extract is not None:
                        # Apply the provided function and put everything on the device
                        f_extract = {} if f_extract is None else f_extract
                        batch = f_extract(batch, **f_args)
                        batch = {f"{name}": batch_input.to(self.device) for name, batch_input in batch.items()}
                    elif isinstance(batch, list | tuple):
                        arg_method = self.first_model.forward
                        args_list = inspect.getfullargspec(arg_method).args[1:]  # skip "self" arg
                        batch = {f"{args_list[i]}": batch_input.to(self.device) for i, batch_input in enumerate(batch)}
                    elif not isinstance(batch, dict):
                        raise ValueError(
                            f"Type {type(batch)} is not supported for the CKA computation. We suggest building a custom"
                            f"'Dataset' class such that the '__get_item__' method returns a dict[str, Any]."
                        )

                    # Do a forward pass for both models
                    first_output = torch.stack(self.first_model(**batch, output_hidden_states=True)["hidden_states"][1:], dim=0).detach() * batch['attention_mask'].unsqueeze(0).unsqueeze(-1).to(torch.float16)
                    second_output = torch.stack(self.second_model(**batch, output_hidden_states=True)["hidden_states"][1:], dim=0).detach() * batch['attention_mask'].unsqueeze(0).unsqueeze(-1).to(torch.float16)

                    for i in range(first_output.shape[0]):
                        for j in range(second_output.shape[0]):
                            cka_epoch[i, j] = cka_batch(first_output[i, :], second_output[j, :])

                    # Compute the CKA values for each output pair
                    # for i, (_, x) in enumerate(first_outputs.items()):
                    #     for j, (_, y) in enumerate(second_outputs.items()):
                    #         cka_epoch[i, j] = cka_batch(x, y)

                cka_matrices.append(cka_epoch)
                torch.cuda.empty_cache()
                gc.collect()
        cka = torch.stack(cka_matrices)
        cka = cka.sum(0) / epochs
        if torch.isnan(cka).any():
            raise ValueError("CKA computation resulted in NANs.")
        return cka.cpu().detach().numpy()


def apply_cka(
        first_loader:DataLoader,
        base_model:AutoModelForCausalLM,
        enhanced_model:PeftModel,
        first_model_name:str,
        second_model_name:str,
        export_data:Optional[bool] = False,
        show_plot:Optional[bool] = False,
        base_model_layers: Optional[List[str]] = None,
        enhanced_model_layers: Optional[List[str]] = None,
        second_loader: Optional[DataLoader] = None,
        device:str='cuda',
    ) -> np.ndarray:
    """
    This function will compare two given models and two different dataset as you wish just like documentation of the torch_cka library.
    :param first_loader: The data loader for first dataset.
    :param base_model: The first model in our case base model.
    :param enhanced_model:  The configured model like general knowledge subtracted model or arrow itself model.
    :param first_model_name: The name of first model.
    :param second_model_name: The name of second model.
    :param export_data: If you want to export the data after comparing two models.
    :param show_plot: If you want to plot the data after comparing two models.
    :param base_model_layers: The specified layer of the base model.
    :param enhanced_model_layers: The specified layer of the second model.
    :param second_loader: If you have two dataset, pass the second loader correspond to that.
    :param device: If you want to use cuda or cpu.
    :return:
    If export_data has been set to true it would return a dictionary contain the data after comparing two models otherwise None would be returned.
    """
    cka = MyCKA(
        first_model=base_model,
        second_model=enhanced_model,
        layers=base_model_layers,
        second_layers=enhanced_model_layers,
        first_name=first_model_name,
        second_name=second_model_name,
        device=device,
    )
    cka_matrix_different = cka(first_loader, epochs=1)
    return cka_matrix_different