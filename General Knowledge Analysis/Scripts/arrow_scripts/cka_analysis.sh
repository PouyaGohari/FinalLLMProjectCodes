python main.py --hf_token "your_token" --seed 42 \
  --dataset_path "AliEdalat/LE_dataset_5token_pred" --top_k 3 --temperature 1.0 \
  --base_model_name "microsoft/Phi-3-mini-4k-instruct" --n_samples 96 --batch 16 \
  --use_attention --export_data --device cuda --save_path results/arrow/arrow_results.csv
