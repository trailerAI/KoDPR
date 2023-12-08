import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from functools import partial
import json

class DPR_Dataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.question = df['question'].to_numpy()
        self.positive_passage = df['positive_passage'].to_numpy()
        self.bm_25 = df['bm_25'].to_numpy()
        self.ctx_idx = df['context'].to_numpy()
        self.answer = df['answer'].to_numpy()

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, idx):
        return {
            "question": self.question[idx],
            "positive_passage": self.positive_passage[idx],
            "bm_25": self.bm_25[idx],
            "ctx_idx": self.ctx_idx[idx],
            "answer": self.answer[idx]
        }
    
def load_data(path: str):
    print(path)
    with open(path) as f:
        dataset = json.load(f)    
    
    return dataset

def collate_fn(batch_size, batch):
    unique_ctx_idx = list(set([item["ctx_idx"] for item in batch]))  # Get unique ctx_idx values in the batch
    batch_data = {
        "question": [],
        "positive_passage": [],
        "bm_25": [],
        "ctx_idx": unique_ctx_idx,
        "answer": []
    }

    for item in batch:
        batch_data["question"].append(item["question"])
        batch_data["positive_passage"].append(item["positive_passage"])
        batch_data["bm_25"].append(item["bm_25"])
        batch_data["answer"].append(item["answer"])

    if len(batch_data["ctx_idx"]) == batch_size:
        return batch_data
    else:
        return None
    

def parse_data(config, dtype):
    dataset_path = config['data'][dtype]
    positive_passage_dataset_path = config['data']['context']
    dataset = load_data(dataset_path)
    positive_passage_dataset = load_data(positive_passage_dataset_path)
    dataset = pd.DataFrame(dataset['data'])
    dataset['positive_passage'] = [ positive_passage_dataset[context_idx] for context_idx in dataset['context']]


    if dtype == "train":
        data_loader = DataLoader(
                                    DPR_Dataset(dataset), 
                                    batch_size=config['hyper_params']['batch_size'], 
                                    shuffle=True, 
                                    num_workers=5,
                                    pin_memory=True,
                                    drop_last=True,
                                    persistent_workers=True,
                                    collate_fn=partial(collate_fn, config['hyper_params']['batch_size'])
                            )
    else:
        data_loader = DataLoader(
                                DPR_Dataset(dataset), 
                                batch_size=config['hyper_params']['batch_size'], 
                                shuffle=False, 
                                num_workers=5,
                                pin_memory=True,
                                drop_last=False,
                                persistent_workers=True,
                                collate_fn=partial(collate_fn, config['hyper_params']['batch_size'])
                        )
    
    # Filter out incomplete batches and convert to list
    data_loader = list(filter(lambda x: x is not None, data_loader))        

    return data_loader