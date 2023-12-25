import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from loader import get_loader
from model import CNNtoRNN

from tqdm import tqdm

def train_step():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader, dataset = get_loader(root_folder = 'FLICKR8k/Images',
                                       annotation_file = 'FLICKR8k/captions.txt',
                                       transform = transform,
                                       num_workers = 2)
    
    torch.backends.mps.benchmark = True
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # hyperparameters
    embed_size = 256
    hidden_state = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    lr = 3e-4
    epochs = 5

    # initialize model
    model = CNNtoRNN(embed_size, hidden_state, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = dataset.vocab.stoi["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr = lr)

    model.train()

    for epoch in tqdm(range(epochs)):
        for idx, (imgs, captions) in enumerate(train_loader):
            
            imgs = imgs.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            loss.backward()
            
            optimizer.step()
        
if __name__ == "__main__":
    train_step()