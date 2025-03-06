import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.text import Perplexity
from tqdm import tqdm
import matplotlib.pyplot as plt

# NOTE Dan you are not allowed to look at this

class TokenDataSet(Dataset):
    def __init__(self, token_list, window_size):
        self.token_list = token_list
        self.window_size = window_size
    def __len__(self):
        return len(self.token_list) - self.window_size - 1
        # return 1000
    def __getitem__(self, idx):
        wind = torch.LongTensor(self.token_list[idx:idx + self.window_size])
        lab = torch.LongTensor([self.token_list[idx + self.window_size + 1]]).squeeze()
        return wind, lab

def get_dataloaders(window_size, batch_size):
    # read in data, tokenize and make dataloaders
    with open('wiki2tokenized.train.pkl', 'rb') as f:
        train_token_list = pickle.load(f)
        train_dataset = TokenDataSet(token_list=train_token_list, window_size=window_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=False)

    with open('wiki2tokenized.valid.pkl', 'rb') as f:
        valid_token_list = pickle.load(f)
        valid_dataset = TokenDataSet(token_list=valid_token_list, window_size=window_size)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, shuffle=False)

    with open('wiki2tokenized.test.pkl', 'rb') as f:
        test_token_list = pickle.load(f)
        test_dataset = TokenDataSet(token_list=test_token_list, window_size=window_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

# TODO dropout?
class RNNLangModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.dropout_again = torch.nn.Dropout(p=0.1)
        self.embeddings = torch.nn.Embedding(
            num_embeddings=9892, 
            embedding_dim=100
        )
        self.rnn = torch.nn.RNN(
            input_size=100,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity='tanh',
            bias=True,
            batch_first=True
        )
        self.U = torch.nn.Linear(
            in_features=hidden_size,
            out_features=9892
        )
            
    def forward(self, x):
        embedded_x = self.embeddings(x)
        pred, hidden = self.rnn.forward(self.dropout(embedded_x))
        pred, hidden = self.rnn.forward(embedded_x)

        # hidden = self.dropout_again(self.U.forward(hidden).squeeze(0))
        hidden = self.U.forward(hidden).squeeze(0)
        return hidden
        

if __name__ == '__main__':
    torch.manual_seed(0)
    # get GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model
    net = RNNLangModel(hidden_size=200)
    net.to(device)
    # get dataloaders for each split
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(window_size = 30, batch_size = 128)
    # optimizer and loss
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    # logging
    best_loss = float('inf')
    train_losses, valid_losses = [], []
    train_perps, valid_perps = [], []

    epochs = 20
    for epoch in range(epochs):
        net.train()
        metric = Perplexity()
        metric.to(device)
        # train
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_index, (token_windows, next_tokens) in progress_bar:
            token_windows = token_windows.to(device)
            next_tokens = next_tokens.to(device)
            # zero gradients
            optim.zero_grad()
            # get prediction and hidden state from RNN 
            prediction_logits = net.forward(token_windows)
            # loss calc
            train_loss = loss_fn(prediction_logits, next_tokens)
            running_loss += train_loss.item()
            average_loss = running_loss / (batch_index + 1)

            # backprop
            train_loss.backward()
            # update weights
            optim.step()
            # compute training perplexity
            metric.update(prediction_logits.unsqueeze(1), next_tokens.unsqueeze(-1))
        # final perplexity computation
        perp = metric.compute().cpu().item()
        train_perps.append(perp)
        print(f'Train perplexity for Epoch {epoch}: {perp}')

        train_losses.append(average_loss)
        print(f'Average training loss at Epoch {epoch}: {average_loss}')

        # validation
        with torch.no_grad():
            net.eval()
            # initialize perplexity metric
            metric = Perplexity()
            metric.to(device)
            running_loss = 0.0
            for batch_index, (token_windows, next_tokens) in enumerate(valid_dataloader):
                token_windows = token_windows.to(device)
                next_tokens = next_tokens.to(device)
                prediction_logits = net.forward(token_windows)
                # loss calc
                valid_loss = loss_fn(prediction_logits, next_tokens)
                running_loss += valid_loss.item()
                average_loss = running_loss / (batch_index + 1)
                # compute perplexity
                metric.update(prediction_logits.unsqueeze(1), next_tokens.unsqueeze(-1))
            # final perplexity
            perp = metric.compute().cpu().item()
            valid_perps.append(perp)
            print(f'Validation perplexity for Epoch {epoch}: {perp}')
            # add loss for logging
            valid_losses.append(average_loss)
            print(f'Average validation loss at Epoch {epoch}: {average_loss}')

        # save weights if validation performance is better        
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(net.state_dict(), f'model.pth')
            print(' > Model saved!')
    # testing
    with torch.no_grad():
        net.eval()
        # initialize perplexity metric
        metric = Perplexity()
        metric.to(device)
        for batch_index, (token_windows, next_tokens) in enumerate(test_dataloader):
            token_windows = token_windows.to(device)
            next_tokens = next_tokens.to(device)
            prediction_logits = net.forward(token_windows)
            # compute perplexity
            metric.update(prediction_logits.unsqueeze(1), next_tokens.unsqueeze(-1))
        # final perplexity
        perp = metric.compute().cpu().item()
        print(f'Test perplexity: {perp}')

    plt.plot([i+1 for i in range(epochs)], train_perps, label='Training Perplexity')
    plt.plot([i+1 for i in range(epochs)], valid_perps, label='Validation Perplexity')
    plt.xlabel(f'Epochs')
    plt.ylabel(f'Perplexity')
    plt.title(f'Training Curve for 1-Layer RNN')
    plt.legend()
    plt.savefig(f'LearningCurve.png')
    plt.show()
