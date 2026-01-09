import logging
from typing import List
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================== Data Prepration ===========================
class FinancialSentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_map = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }

        # reverse mapping for predictions
        self.id_to_label = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.label_map[label], dtype=torch.long)
            }

