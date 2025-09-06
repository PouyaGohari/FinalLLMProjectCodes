from torch.utils.data import Dataset
from transformers import AutoTokenizer
import datasets
from utils.MyConfig import *

class CustomDataset(Dataset):
    def __init__(self, text_dataset:datasets.Dataset, tokenizer:AutoTokenizer.from_pretrained, use_attention:bool=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = text_dataset
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.use_attention = use_attention

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.tokenizer.chat_template:
            user = {
                "content":f"{self.dataset[idx]['prompt']}",
                "role":"user"
            }
            text = self.tokenizer.apply_chat_template(
                [user],
                add_generation_prompt=True,
                padding='max_length',
                truncation=True,
                tokenize=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
                return_dict=self.use_attention,
            )
            if self.use_attention:
                return {'input_ids': text['input_ids'].squeeze(0).to('cuda'), 'attention_mask': text['attention_mask'].squeeze(0).to('cuda')}
            return {'input_ids':text.squeeze(0).to('cuda')}
        else:
            text = self.tokenizer(
                self.dataset[idx]['prompt'],
                return_tensors="pt",
                return_attention_mask=self.use_attention,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
            )
            if self.use_attention:
                return {'input_ids': text['input_ids'].squeeze(0).to('cuda'), 'attention_mask': text['attention_mask'].squeeze(0).to('cuda')}
            else:
                return {'input_ids':text.squeeze(0).to('cuda')}

