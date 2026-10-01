import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def relu(neuron: torch.Tensor) -> torch.Tensor:
    a = torch.zeros_like(neuron)
    return torch.max(neuron, a)
        

class MLPScratch(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_neurons: int, learning_rate: float = 0.01) -> None:
        super().__init__()
        # hyperparams
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_neurons 
        self.learning_rate = learning_rate
        # params
        self.W1 = nn.Parameter(torch.randn(num_inputs, self.num_hiddens) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(self.num_hiddens))
        self.W2 = nn.Parameter(torch.randn(self.num_hiddens, self.num_outputs))
        self.b2 = nn.Parameter(torch.zeros(self.num_outputs)) 
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape((-1, self.num_inputs)) # flatten
        h = relu(X @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        return y 
    
    
    

if __name__ == '__main__':   
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    image, label = training_data[0]
    print("Image shape:", image.shape)
    print("Label index:", label)
    print("Class name:", training_data.classes[label])
    print(f"Size of training data: {len(training_data)}")
    print(f"Size of testing data: {len(test_data)}")
    X, y = training_data[0]

    
    train_loader = DataLoader(
        training_data,
        batch_size=256,
        shuffle=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=256,
        shuffle=False
    )
    
    model = MLPScratch(
        num_inputs=28*28,
        num_outputs=10,
        num_neurons=256,
        learning_rate=0.01
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    epochs = 20
    for epoch in range(epochs):
        
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            losses.append(loss)
            loss.backward()
            with torch.no_grad():
                model.W1 -= model.learning_rate * model.W1.grad
                model.b1 -= model.learning_rate * model.b1.grad
                model.W2 -= model.learning_rate * model.W2.grad 
                model.b2 -= model.learning_rate * model.b2.grad 
            model.zero_grad()
    print(losses[0])
    print(losses[-1])

    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)
            
            prediction = y_hat.argmax(dim=1)
            
            
            correct += (prediction == y).sum().item()
            total += y.size(0)
    print(f"Accuracy: {correct/total}")
        