import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import flwr as fl
from collections import OrderedDict
import numpy as np
import random

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

BATCH_SIZE = 32
K = 64  # no. of levels of Quantization
print(K)


def load_dataset():
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=trf)
    val_set = datasets.MNIST(root='./data', train=False,
                             download=True, transform=trf)

    # randomly taking 6000 samples from the dataset (per client).
    train_size = len(train_set)//10
    val_size = len(val_set)//10

    train_set = random_split(
        train_set, [train_size, len(train_set)-train_size])[0]
    val_set = random_split(val_set, [val_size, len(val_set)-val_size])[0]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, trainloader, epochs):
    model.train(True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for X, y in tqdm(trainloader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            yhat = model(X)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()


def val(model, valloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    crct, loss = 0, 0.0
    with torch.no_grad():
        for X, y in tqdm(valloader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            yhat = model(X)
            loss += loss_fn(yhat, y)
            crct += (torch.max(yhat.data, 1)[1] == y).sum().item()
    acc = crct / len(valloader.dataset)
    return loss, acc


model = LeNet5().to(DEVICE)
trainloader, valloader = load_dataset()
def to_gray_code(n):
    return n ^ (n >> 1)


def from_gray_code(gray):
    n = gray & 1
    for bit in range(1, gray.bit_length()):
        n |= ((gray >> bit) ^ (n >> 1)) << bit
    return n


def flip_bit(bit):
    # Flip the bit with a probability of 0.1
    if random.random() < 0.4:
        return '0' if bit == '1' else '1'
    else:
        return bit
flip_probability = 0.1

def qunatization(params):

    # maxs, mins for each block and thus finding si
    #print("paramters",params)
    mins = torch.min(params, axis=1, keepdims=True).values
    maxs = torch.max(params, axis=1, keepdims=True).values
    si = maxs - mins

    # quantization levels for each block
    Bi = mins + si*(torch.arange(K)/(K-1))
    print("Bi",Bi)
    print("bi shape",Bi.shape)
   

    # finding B(r) as per paper.
    ids = torch.searchsorted(Bi, params, side='right')-1


    # making them into (B(r+1), B(r))
    points = torch.cat([
        torch.unsqueeze(ids, -1),
        torch.unsqueeze((ids+1), -1)
    ], axis=-1)
    #print("points",points.shape)

    # marking B(r) = B(r+1) for max in each block.
    points[points == K] = K-1
    #print("points",points.shape)
    

    # converting indices into quantization values.
    Brs = Bi[torch.arange(Bi.shape[0]).unsqueeze(1),
             points.view(Bi.shape[0], -1)].view(points.shape)
    
    #print("BRs",Brs)
    
    #Brs_conv=Brs.view(44,1024)

    #print("shape of nre brs",Brs_conv.shape)
    # finding probability for each parameter. probability of max element is 0.
    probs = torch.where(
        Brs[..., 1] != Brs[..., 0],
        (params-Brs[..., 0])/(Brs[..., 1]-Brs[..., 0]),
        0
    )
    # returns 1s, 0s based on paper.
    #print("probs",probs)
    encs =  torch.bernoulli(probs)
    

    # replacing 1s with B(r+1) and 0s with B(r) and flattenning the whole array.
    dec = torch.where(
        encs == 1,
        Brs[..., 1],
        Brs[..., 0]
    )
    row_no=0
    indices_list=[]
    for b_row in dec:
    # Initialize an empty list to store the indices for the current row
        row_indices = []
    # Iterate over each value in the current row of dec
        for b_value in b_row:
        # Find the indices in Bi where the value matches the value in the current row of dec
            matching_indices = torch.where(Bi[row_no] == b_value)
            a=matching_indices[0][0]
        # Add the matching indices to the list for the current row
            #row_indices.extend(list(matching_indices[1].numpy()))  # Convert tensor to NumPy array for compatibility with lists
            row_indices.append(a)
    # Append the list of indices for th e current row to the indices list
        indices_list.append(row_indices)
        row_no=row_no+1
    indices_array = np.array(indices_list)
    print("Indices of values in dec found in Bi:")
    print(indices_array)
    flipped_tensor = []
    for row in indices_array:
        flipped_row = []
        for element in row:
            gray_code = to_gray_code(element.item())
            gray_code=torch.tensor(gray_code)
            binary_representation = format(gray_code.item(), '06b')
            flipped_gray_code = ''.join([flip_bit(bit) for bit in binary_representation])
            #print("flipped gray ",flipped_gray_code)
            binary = flipped_gray_code[0]  
            for i in range(1, len(flipped_gray_code)):
                binary += '1' if flipped_gray_code[i] != binary[i - 1] else '0'
            integer_value = int(binary, 2)
            flipped_row.append(integer_value)
        flipped_tensor.append(flipped_row)
    flipped_tensor = torch.tensor(flipped_tensor)
    print("flipped ids",flipped_tensor)
    dec_list=[]
    row_no=0
    for b_row in flipped_tensor:
        row_ind=Bi[row_no][b_row].numpy()
        print("shape of row inf",row_ind)
        dec_list.append(row_ind)
        row_no=row_no+1
    
    dec_list=np.array(dec_list)
    print("dec_list",dec_list.shape)
    
    dec=dec_list.flatten()
    dec=torch.tensor(dec)
    
    # reconstructing the parameters into their corresponding shapes.
    revert = []
    ptr = 0
    for layer in model.parameters():
        size = layer.numel()
        revert.append(dec[ptr: ptr+size].reshape(layer.shape).cpu().numpy())
        ptr += size
    return revert


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print("[SENDING PARAMETERS TO SERVER]")

        # flattening the parameters
        flat_params = nn.utils.parameters_to_vector(
            self.model.parameters()).detach()

        # splitting parameters into 1024 batches each
        params = torch.cat([flat_params,
                            torch.zeros(1024 - (flat_params.numel() % 1024))]).view(-1, 1024)
        return qunatization(params)

    def set_parameters(self, parameters, config):
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in param_dict
        })
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        local_epochs = config["local_epochs"]

        print(f"[FIT, config: {config}]")
        print("[FIT, RECEIVED PARAMETERS FROM SERVER]")

        self.set_parameters(parameters, config)
        train(self.model, self.trainloader, epochs=local_epochs)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("[EVAL, RECEIVED PARAMETERS FROM SERVER]")
        self.set_parameters(parameters, config)
        loss, acc = val(self.model, self.valloader)
        print("[EVAL, SENDING METRICS TO SERVER]")
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc),
                                                          "losss": float(loss)}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(model, trainloader, valloader),
    )