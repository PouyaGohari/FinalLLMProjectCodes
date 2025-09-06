# FinalLLMProject
This repository contains modular, well-documented code for analyzing large language models using the CKA metric.
The implementation is designed to be flexible, extensible, and suitable for experiments with different model architectures and evaluation setups.
## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Example](#examples)
- [Requirements](#requirements)

# Introduction
This project analyzes two different models using the CKA (Centered Kernel Alignment) metric. It serves as the final project for the LLM
course at the University of Tehran (UT), focusing on evaluating models in line with the methodology
from this [paper](https://arxiv.org/abs/2505.10939). Our primary goal is to investigate the impact of general knowledge subtraction on 
the base model.

In this project, we apply both the Arrow method and general knowledge subtraction models to the Wiki Dataset 
to explore how these techniques influence the general knowledge encoded in large language models.
# Usage
We develop our code in a modular fashion, but run the scripts in a Colab environment.
To get started, please review the [guidance](Guidance.text), and copy the installation instructions into the first cell of your Jupyter notebook.

After cloning our repository ([repo](https://github.com/PouyaGohari/FinalLLMProject.git)), you should install the [requirements](requirements.txt):

```
!pip install -qU -r /content/FinalLLMProject/requirements.txt
```
Then you can run the script below:
```
python main.py --hf_token "your_token" --seed 42 \
  --dataset_path "AliEdalat/LE_dataset_5token_pred" --top_k 3 --temperature 1.0 \
  --base_model_name "microsoft/Phi-3-mini-4k-instruct" --n_samples 96 --batch 16 \
  --use_attention --export_data --device cuda --save_path results/arrow/arrow_results.csv
```

For a full list of available command-line arguments, please see the [MyArgParser.py](utils/MyArgParser.py) file.

You can see my scripts in script folder to get information more.

# Examples
For a usage example, please refer to the code provided in the [main.py](main.py) file.

# Requirements
As mentioned in the Guidance, you should first install the appropriate version of peft as described, and then install the remaining packages listed in [requirements](requirements.text).

