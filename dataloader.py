# !pip install datasets # command to install datasets package from huggingface in colab
# !pip install transformers to install transformers package (BERT tokenizer)
import datasets
import torch
import torch.utils.data as d
from transformers import BertTokenizer
import math


class MyIterableDataset(d.IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len, num_articles):
        super(MyIterableDataset).__init__()
        self.dataset = dataset
        self.num_articles = num_articles
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def __iter__(self):
        def helper(start, end):
            for i in range(start, end):
                index = 0
                article = self.dataset[i]["text"]
                tokenized = self.tokenizer(article, return_tensors='pt')
                tokenized = tokenized["input_ids"][0][1:] # unpack from 2dim tensor and skip the prepended bert mask token
                length = len(tokenized)
                while (index + self.seq_len <= length):
                    yield tokenized[index:index+self.seq_len]
                    index += self.seq_len

        worker_info = d.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start = 0
            end = self.num_articles
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.num_articles / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self.num_articles)
        return helper(start, end)
    
def give_dataloaders(batch_size=1, seq_len=513, proportions = [0.5, 0.4, 0.05, 0.05]):
    """Returns a dictionary with train_dataloader, val_dataloader and test_dataloader
    which are created from huggingface wikipedia dataset
    that is split according to "proportions" variable

    dataloaders yield batches of sequnces of token_ids
    (in fact: a tensor of shape (batch_size, seq_len))

    proportions - fractions of articles in respectively 
    null, train, val, test dataset split, where null is discarded,
    those should be fractions that add up to one"""
    wiki_huggingface_dataset = datasets.load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
    
    wiki_huggingface_dataset = wiki_huggingface_dataset["train"]
    l = d.random_split(wiki_huggingface_dataset, lengths=proportions, generator=torch.Generator().manual_seed(42))
    train_dataset, val_dataset, test_dataset = l[1], l[2], l[3]
    l = None

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = MyIterableDataset(train_dataset, tokenizer, seq_len, num_articles=len(train_dataset.indices))
    val_dataset = MyIterableDataset(val_dataset, tokenizer, seq_len, num_articles=len(val_dataset.indices))
    test_dataset = MyIterableDataset(test_dataset, tokenizer, seq_len, num_articles=len(test_dataset.indices))
    return {"train_dataloader": d.DataLoader(train_dataset, batch_size, pin_memory=True),
            "val_dataloader": d.DataLoader(val_dataset, batch_size, pin_memory=True),
            "test_dataloader": d.DataLoader(test_dataset, batch_size, pin_memory=True)}

# # example of usage:
# l = give_dataloaders()
# train_dataloader = l["train_dataloader"]
# val_dataloader = l["val_dataloader"]
# test_dataloader = l["test_dataloader"]
# sample = next(iter(train_dataloader))
# print(sample)