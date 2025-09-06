python preplexity.py --hf_token "your_token" --batch 16 --dataset_name "AliEdalat/le_datasets_from_machine" \
  --seed 42 --n_samples 96 --save_path results/perplexity/gks/c4_gks --data_file_train "en_c4_10k_LM_511_1_machine" --data_file_test "en_c4_10k_LM_511_1_test_machine" --apply_arrow --apply_gks
