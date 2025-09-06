python main.py --hf_token "your_token" --seed 42 \
  --dataset_path "AliEdalat/le_datasets_from_machine" --top_k 3 --temperature 1.0 --data_file_train "en_c4_10k_LM_511_1_machine" --data_file_test "en_c4_10k_LM_511_1_test_machine"\
  --base_model_name "microsoft/Phi-3-mini-4k-instruct" --n_samples 96 --batch 16 \
  --use_attention --export_data --device cuda --save_path results/arrow/arrow_results_c4.csv
