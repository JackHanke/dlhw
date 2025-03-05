import torch
import torcheval

# NOTE Dan you are not allowed to look at this

def preprocess_data():
    # read in data, tokenize and make dataloaders
    with open('wiki2.train.txt', 'r') as f:
        train_data = f.read()
        train_dataloader = torch.utils.data.dataloaders(dataset=train_data, batch_size=4, shuffle=False)
    with open('wiki2.valid.txt', 'r') as f:
        valid_data = f.read()
        valid_dataloader = torch.utils.data.dataloaders(dataset=valid_data, batch_size=4, shuffle=False)
    with open('wiki2.test.txt', 'r') as f:
        test_data = f.read()
        test_dataloader = torch.utils.data.dataloaders(dataset=test_data, batch_size=4, shuffle=False)

if __name__ == '__main__':
    torch.manual_seed(0)
    net = torch.nn.RNN(
        input_size=100,
        hidden_size=10,
        num_layers=1,
        nonlinearity='tanh',
        bias=True,
        dropout=0.1
    )

    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    


    epochs = 20
    for epoch in range(epochs):
        for batch_index, batch in enumerate(train_dataloader):
            # prediction
            net.forward()
            # loss calc

            loss = loss_fn()

            # backprop
            loss.backward()

            # update weights
            optimizer.step()

            # perplexity calc
            torcheval.metrics.Perplexity()




    pass

