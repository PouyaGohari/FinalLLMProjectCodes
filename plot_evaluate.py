from typing import Dict, Union, List

import numpy as np
import pandas as pd
import torch
import ast
from utils.MyArgParser import plot_parser
import matplotlib.pyplot as plt
import seaborn as sn

def string_to_tensor(cka: str) -> torch.Tensor:
    """
    This function will get a string to convert it to tensor.
    :param cka: The cka row that stored in string.
    :return:
    Torch
    """
    list_str = cka.replace('tensor(', '').rstrip(')')
    return torch.tensor(ast.literal_eval(list_str), dtype=torch.float64)

def string_to_list(layers:str) -> List[str]:
    """
    This function will get a string to convert it to list of string.
    :param layers: The layers that stored in string.
    :return:
    List of strings for the layers.
    """
    return ast.literal_eval(layers)

def stacking_layers(dataframe:pd.DataFrame) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    A dictionary with specified layers and stacked tensors.
    :param dataframe:
    :return:
    A dictionary with specified layers and stacked tensors.
    """
    num_layers = len(dataframe)
    result = {
        'model1_layers': [],
        'model2_layers': dataframe.iloc[0]['model2_layers'],
        'CKA': torch.zeros((num_layers, len(dataframe.iloc[0]['model2_layers'])))
    }
    for index, row in dataframe.iterrows():
        result['model1_layers'].append(row['model1_layers'][0] if isinstance(row['model1_layers'], list) else row['model1_layers'])
        cka_tensor = row['CKA']
        if cka_tensor.ndim == 2 and cka_tensor.shape[0] == 1:
            cka_tensor = cka_tensor.squeeze(0)
        result['CKA'][index, :] = cka_tensor
    return result

def plot_cka(
    cka_matrix: torch.Tensor,
    first_layers: list[str],
    second_layers: list[str],
    first_name: str = "First Model",
    second_name: str = "Second Model",
    save_path: str | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
    show_ticks_labels: bool = False,
    short_tick_labels_splits: int | None = None,
    use_tight_layout: bool = True,
    show_annotations: bool = True,
    show_img: bool = True,
    show_half_heatmap: bool = False,
    invert_y_axis: bool = True,
) -> None:
    """Plot the CKA matrix obtained by calling CKA class __call__() method.
    (Note that this is from Bert-Similarity Project and had a bug and we fixed this.)
    Args:
        cka_matrix (torch.Tensor): the CKA matrix.
        first_layers (list[str]): list of the names of the first model's layers.
        second_layers (list[str]): list of the names of the second model's layers.
        first_name (str): name of the first model (default="First Model").
        second_name (str): name of the second model (default="Second Model").
        save_path (str | None): where to save the plot, if None then the plot will not be saved (default=None).
        title (str | None): the plot title, if None then a simple text with the name of both models will be used
            (default=None).
        vmin (float | None): values to anchor the colormap, otherwise they are inferred from the data and other keyword
            arguments (default=None).
        vmax (float | None): values to anchor the colormap, otherwise they are inferred from the data and other keyword
            arguments (default=None).
        cmap (str): the name of the colormap to use (default="magma").
        show_ticks_labels (bool): whether to show the tick labels (default=False).
        short_tick_labels_splits (int | None): only works when show_tick_labels is True. If it is not None, the tick
            labels will be shortened to the defined sublayer starting from the deepest level. E.g.: if the layer name
            is 'encoder.ff.linear' and this parameter is set to 1, then only 'linear' will be printed on the heatmap
            (default=None).
        use_tight_layout (bool): whether to use a tight layout in order not to cut any label in the plot (default=True).
        show_annotations (bool): whether to show the annotations on the heatmap (default=True).
        show_img (bool): whether to show the plot (default=True).
        show_half_heatmap (bool): whether to mask the upper left part of the heatmap since those valued are duplicates
            (default=False).
        invert_y_axis (bool): whether to invert the y-axis of the plot (default=True).

    Raises:
        ValueError: if ``vmax`` or ``vmin`` are not defined together or both equal to None.
    """
    # Deal with vmin and vmax
    if (vmin is not None) ^ (vmax is not None):
        raise ValueError("'vmin' and 'vmax' must be defined together or both equal to None.")

    vmin = min(vmin, torch.min(cka_matrix).item()) if vmin is not None else vmin
    vmax = max(vmax, torch.max(cka_matrix).item()) if vmax is not None else vmax

    # Build the mask
    mask = torch.tril(torch.ones_like(cka_matrix, dtype=torch.bool), diagonal=-1) if show_half_heatmap else None

    # Build the heatmap
    if mask:
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap, mask=mask.cpu().numpy())
    else:
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap, mask=mask)
    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_xlabel(f"{second_name} layers", fontsize=12)
    ax.set_ylabel(f"{first_name} layers", fontsize=12)

    # Deal with tick labels
    if show_ticks_labels:
        if short_tick_labels_splits is None:
            ax.set_xticklabels(second_name)
            ax.set_yticklabels(first_name)
        else:
            ax.set_xticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in second_layers])
            ax.set_yticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in first_layers])

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    # Put the title if passed
    if title is not None:
        ax.set_title(title, fontsize=14)
    else:
        title = f"{first_name} vs {second_name}"
        ax.set_title(title, fontsize=14)

    # Set the layout to tight if the corresponding parameter is True
    if use_tight_layout:
        plt.tight_layout()

    # Save the plot to the specified path if defined
    if save_path is not None:
        title = title.replace(" ", "_").replace("/", "-")
        path_rel = f"{save_path}/{title}.png"
        plt.savefig(path_rel, dpi=400, bbox_inches="tight")

    # Show the image if the user chooses to do so
    if show_img:
        plt.show()

if __name__ == '__main__':
    args = plot_parser()
    df = None
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError as fnf_error:
        print(fnf_error)

    if len(df) == 1:
        first_layers = df['first_layers']
        second_layers = df['second_layers']
        cka_results = torch.tensor(np.load(args.csv_path+".npy"), dtype=torch.float64).to('cuda')

    else:
        first_layers = df.columns.tolist()
        second_layers = [f"base_model.model.layers{i}" for i in range(len(df))]
        cka_results = torch.zeros((len(first_layers), len(second_layers)))
        for index, row in df.iterrows():
            cka_results[index, :] = string_to_tensor(row['exported_data'])
        cka_results.to('cuda')

    if args.normalized:
        normalized_cka = (cka_results - cka_results.min()) / (cka_results.max() - cka_results.min())
    else:
        normalized_cka = cka_results

    if args.show_stats:
        stats_config = "normalized" if args.normalized else "raw"
        print(f"Calculating statistics for {stats_config} cka results...")
        print(f"Diagonal mean is: {torch.diagonal(normalized_cka).mean()}")
        print(f"Mean is: {torch.mean(normalized_cka).item()}")

    plot_cka(
        normalized_cka,
        first_layers=first_layers,
        second_layers=second_layers,
        title=args.title,
        first_name=args.ylabel,
        second_name=args.xlabel,
        save_path=args.save_path,
        show_annotations=False
    )