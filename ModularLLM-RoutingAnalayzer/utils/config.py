MAX_LENGTH = 4000

LORA_TARGET_MODULES = ["o_proj", "qkv_proj"]
# OTHER_TRAINABLE_MODULES = ['embed_tokens', 'lm_head']
TASK_TYPE = "CAUSAL_LM"

EXPERTS_FOLDER_PATH = 'results'

EXPERT_REPO_ID = "AliEdalat/le_experts_phi3_diff_lang"
CLUSTER_NAMES = {f'cluster{i}':f"TahaBa/phi3-mini-clustered-flan/ts_expert_{i}" for i in range(10)}
# Since we are using a single expert in English in our directory is a single dictionary. Change it with your configurations.
LANGUAGE_EXPERTS = {'English_adapter': 'TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langen/checkpoint-16'}
DATA_FILE = {'train': 'en_Wiki_10k_LM_511_1 (1).json', 'test': 'en_Wiki_10k_LM_511_1_test (1).json'}
MAX_LENGTH = 1024