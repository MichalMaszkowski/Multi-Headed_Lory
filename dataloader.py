# !pip install datasets # command to install datasets package from huggingface in colab
# !pip install transformers to install transformers package (BERT tokenizer)
from datasets import load_dataset
import torch.utils.data as d
from transformers import BertTokenizer
import math

class MyIterableDataset(d.IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len, article_indices):
        super(MyIterableDataset).__init__()
        self.dataset = dataset
        self.num_articles = len(article_indices)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def __iter__(self):
        def helper(start, end):
            for i in range(start, end):
                all_whole = False
                index = 0
                article = self.dataset[i]["text"]
                tokenized = self.tokenizer(article, return_tensors='pt')
                tokenized = tokenized["input_ids"][0][1:]
                length = len(tokenized)
                if index + self.seq_len > length:
                    all_whole = True
                while (not all_whole):
                    yield tokenized[index:index+self.seq_len]
                    index += self.seq_len
                    if index + self.seq_len > length:
                        all_whole = True
                if all_whole:
                    yield tokenized[index:]

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
    
def give_dataloader(batch_size=1, development=True):
    if development:
        wiki_huggingface_dataset = load_dataset("wikipedia", "20220301.simple") # a smaller dataset for development
    else:
        wiki_huggingface_dataset = load_dataset("wikipedia", "20220301.en") # large dataset for training

    wiki_huggingface_dataset = wiki_huggingface_dataset["train"]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    ds = MyIterableDataset(wiki_huggingface_dataset, tokenizer, 20, article_indices=range(wiki_huggingface_dataset.num_rows))
    return d.DataLoader(ds, batch_size)

#example of usage:
# data_loader = give_dataloader()
# sample = next(iter(data_loader))
# print(sample)