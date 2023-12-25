import os
import torch
import pandas as pd 
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class Vocabulary:
    def __init__(self, frequency_threshold, ):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unknown>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unknown>": 3}
        self.frequency_threshold = frequency_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [token.lower().strip() for token in text.split(' ')]

    def build_vocabulary(self, sentences):
        frequency = {}
        index = 4

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1
                
                if frequency[word] == self.frequency_threshold:
                    self.itos[index] = word
                    self.stoi[word] = index

                    index += 1

    def preprocess(self, caption):
        tokenized_caption = self.tokenizer_eng(caption)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unknown>"]
            for token in tokenized_caption
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file, frequency_threshold = 5, transform = None):
        self.root_dir = root_dir
        self.data = pd.read_csv(caption_file)
        self.transform = transform

        # retrieving image/caption columns
        self.imgs = self.data['image']
        self.captions = self.data['caption']

        # initializing and building vocabulary
        self.vocab = Vocabulary(frequency_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]

        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        processed_caption = [self.vocab.stoi["<start>"]]
        processed_caption += self.vocab.preprocess(caption)
        processed_caption.append(self.vocab.stoi["<end>"])

        return img, torch.tensor(processed_caption)
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)

        return imgs, targets

def get_loader(root_folder, annotation_file, transform, batch_size = 32, 
               num_workers = 8, shuffle = True, pin_memory = True): 
    
    dataset = FlickrDataset(root_folder, annotation_file, transform = transform)
    pad_idx = dataset.vocab.stoi["<pad>"]

    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn = MyCollate(pad_idx = pad_idx)
    )

    return loader, dataset