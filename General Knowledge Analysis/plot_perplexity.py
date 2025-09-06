import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_perplexity(path:str, dataset_name:str, title:str, x_label:str, y_label:str, save_path : str = None) -> None:
    """
    Plot perplexity over batch for a dataset and save the figure if save_path is not None.
    :param path: Path of saved numpy arrays.
    :param dataset_name: The specified dataset name.
    :param title: The title of the plot.
    :param x_label: The x-axis label.
    :param y_label: The y-axis label.
    :param save_path: Path to save the plot.
    :return:
    None.
    """
    arrays = {}
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if dataset_name + '_gks' in file:
                    arrays['gks'] = np.load(os.path.join(root, file))
                elif dataset_name + '_arrow' in file:
                    arrays['arrow'] = np.load(os.path.join(root, file))
                elif dataset_name + '_base' in file:
                    arrays['base'] = np.load(os.path.join(root, file))
    plt.figure(figsize=[10,6])
    x = np.arange(len(arrays['base']))
    plt.plot(x, arrays['base'], label=f"Base Model")
    plt.plot(x, arrays['arrow'], label=f"Arrow Model")
    plt.plot(x, arrays['gks'], label=f"GKS Model")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)


if __name__ == "__main__":
    path = "results/perplexity/"
    datasets = ['wiki', 'c4', 'books']
    save_main_path = "results/plots/perplexity/"
    titles = "Perplexity over {dataset_name} for Base and Arrow and GKS Models"
    # path = "results/Phi2/perplexity/"
    # datasets = ['wiki', 'c4', 'books']
    # save_main_path = "results/plots/perplexity/phi2/"
    # titles = "Perplexity over {dataset_name} for Base and Arrow and GKS (Phi2)"
    for dataset_name in datasets:
        plot_perplexity(
            path=path,
            dataset_name=dataset_name,
            title=titles.format(dataset_name=dataset_name),
            x_label='Batch',
            y_label='Perplexity',
            save_path=save_main_path + dataset_name
        )
