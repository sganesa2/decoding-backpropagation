import random
from pathlib import Path
from dataclasses import dataclass, field

import torch

def stoi()->dict[str,int]:
    start_index, total_chars = 97, 26
    stoi_dict = {chr(i):i-start_index+1 for i in range(start_index, start_index+total_chars+1)}
    return {".":0, **stoi_dict}

def itos()->dict[int,str]:
    stoi_dict = stoi()
    return {v:k for k,v in stoi_dict.items()}

@dataclass
class NgramDataset:
    """
    Creates dataset splits with each split containing:
        x(input): torch.tensor of shape (split_size, n)
        y(output): torch.tensor of shape (split_size, 1)
    """
    n: int
    file_name: str
    train_size: int
    test_size: int
    dev_size: int

    x: torch.Tensor = field(default= torch.tensor(0), init=False)
    y: torch.Tensor = field(default= torch.tensor(0), init=False)

    @property
    def trainset(self)->tuple[torch.Tensor, torch.Tensor]:
        start,end =0, self.train_size
        return self.x[start:end], self.y[start:end]
    
    @property
    def testset(self)->tuple[torch.Tensor, torch.Tensor]:
        start,end =self.train_size, start+self.test_size
        return self.x[start:end], self.y[start:end]

    @property
    def devset(self)->tuple[torch.Tensor, torch.Tensor]:
        start,end =self.train_size+self.test_size, start+self.dev_size
        return self.x[start:end], self.y[start:end]

    def get_complete_dataset(self)->tuple[torch.Tensor, torch.Tensor]:
        stoi_dict = stoi()

        with open(Path(__file__).parent.joinpath(self.file_name), 'r') as f:
            names = f.read().splitlines()
            random.seed(0)
            random.shuffle(names)
        X,Y = [], []
        for name in names:
            context = [0]*self.n
            chrs = list(name)
            for c in chrs:
                Y.append(stoi_dict[c])
                X.append((context))
                context = context[1:] + [stoi_dict[c]]

        self.x, self.y = torch.tensor(X), torch.tensor(Y)
        return self.x,self.y