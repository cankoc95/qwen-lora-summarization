# SamSum dataset class
# 

import pathlib
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

HF_SYSTEM_MARKER = "system"
HF_USER_MARKER = "user"
HF_ASSISTANT_MARKER = "assistant"

class SamSumDataset(Dataset):
    """
    Dataset class for SamSum
    """
    def __init__(self, split, tokenizer, max_seq_len, data_dir=None, subset_size=None):
        """
        split: 'train', 'validation' or 'test'
        tokenizer: Qwen tokenizer
        max_seq_len: Maximum length of a sequence (context length)
        data_dir: Optional path to the local data directory.
        subset_size: Optional. Only for debugging on Mac. Limits the data to the subset size.
        """

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        raw_data = SamSumDataset._load_data(split=split, data_dir=data_dir)
        if subset_size:
            raw_data = raw_data.select(range(subset_size))
        self.data = raw_data

    @staticmethod
    def _load_data(split: str, data_dir: pathlib.Path = None):
        if data_dir is not None:
            data_path = data_dir / f"{split}"

            if pathlib.Path(data_path).exists():
                print(f"Loading data from {data_path}")
                return load_from_disk(data_path)

        return load_dataset("knkarthick/samsum")[split]

    def __len__(self):
        """ Returns the size of the dataset """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single item in the dataset at the given index.

        Raw data contains 'dialogue' and 'summary' texts where dialogue is what the human/user inputs and summary is what the llm should predict.
        This function extracts the texts for human and llm, and formats it into the format accepted by Qwen: {"input_ids", "attention_mask", "labels"}
        """
        example = self.data[idx]
        # Dialogue is what user inputs, summary is what the model should generate.
        dialogue, summary = example["dialogue"], example["summary"]

        # Format the raw text into role and content dictionary.
        # messages_prefix contains system + user section where messages_full has system + user + assistant
        messages_prefix, messages_full = SamSumDataset.get_example(human=dialogue, assistant=summary)

        # Token ids for prefix. Prefix doesn't contain labels (summary).
        # add_generation_prompt=True because prefix does NOT contain assistant token.
        prefix_ids = self.tokenizer.apply_chat_template(messages_prefix, add_generation_prompt=True, return_tensors="pt")[0]

        # Token ids that also include the labels (summary).
        # add_generation_prompt=False because this contains assistant token.
        full_ids = self.tokenizer.apply_chat_template(messages_full, add_generation_prompt=False, return_tensors="pt")[0]

        # Truncate the sequence to the maximum sequence length
        if full_ids.shape[0] > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]

        # Prefix length is where labels should be ignored
        prefix_len = min(prefix_ids.shape[0], full_ids.shape[0])
        labels = full_ids.clone()
        labels[:prefix_len] = -100  # ignores system + user part.
        attention_mask = torch.ones_like(full_ids, dtype=torch.long)

        return {"input_ids": full_ids,
                "attention_mask": attention_mask,
                "labels": labels,
        }

    @staticmethod
    def get_example(human: str, assistant: str = None):
        system_var = HF_SYSTEM_MARKER
        user_var = HF_USER_MARKER
        assistant_var = HF_ASSISTANT_MARKER

        system_prompt = (
            "You are a helpful assistant that summarizes given text into brief, clear summaries."
        )
        user_prompt = "Summarize the following text:\n\n" + human

        messages_prefix = [
            {"role": system_var, "content": system_prompt},
            {"role": user_var, "content": user_prompt},
        ]

        messages_full = None
        if assistant is not None:
            messages_full = messages_prefix + [{"role": assistant_var, "content": assistant},]

        return messages_prefix, messages_full

def collate_fn(batch, pad_token_id):
    """
    Samsum collate function.
    Given a list of dicts (batch) where each dict is a sample returned by the dataset.
    Pads each each element in the batch such that all have the same length

    Each element in the batch will look like:
    input_ids:       [batch_size, max_len]
    attention_mask:  [batch_size, max_len]
    labels:          [batch_size, max_len]
    """
    batch_size = len(batch)

    # Find max length in the given list of items.
    # Note: We only need to ensure each element within a batch has the same length.
    # Different batches can have different lengths.
    max_len = max([item["input_ids"].shape[0] for item in batch])

    # Add pad tokens for rows that are shorter than max_len
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    # Prepare attention mask. mask = 0 means the token is a pad token and mask = 1 is a real token.
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    # Prepare labels. Use -100 for the pad tokens since we want CE loss to ignore it.
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    # Iterate over each input in the batch and copy the inputs into [0:max_len]
    for i, item in enumerate(batch):
        # Remember, input_ids, attention_mask and labels already have same length = input_ids.shape[0]
        # This was guaranteed in the dataset setup.
        # We just need to ensure that each batch has the same length = max_len.
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
    }
