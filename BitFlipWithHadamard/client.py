import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import random_split, DataLoader,Subset
from tqdm import tqdm
import flwr as fl
from collections import OrderedDict
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from scipy.linalg import hadamard


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

BATCH_SIZE = 16
rows=44
def quantization_new(x, centroids, boundaries,bits):
  bound=boundaries[bits]
  cent=centroids[bits]

  bound = [-np.inf] + bound + [+np.inf]
  for i,(a, b) in enumerate(zip(bound[:-1], bound[1:])):
      #print(a,b)
    if x>=a and x<=b:
      y=(cent[i])
  return y

def quantization_new_1(x, centroids, boundaries,bits):
  bound=boundaries[bits]
  cent=centroids[bits]

  bound = [-np.inf] + bound + [+np.inf]
  #print(bound)
  outs = []
  for elem in x:
    for i,(a, b) in enumerate(zip(bound[:-1], bound[1:])):
      #print(a,b)
      if elem>=a and elem<=b:
        outs.append(cent[i])
  assert len(x)==len(outs)
  #print(outs)
  return outs


opt_hn_centroids = {1: [0.7978845608028654],
                    2: [0.4527800398860679, 1.5104176087114887],
                    3: [0.24509416307340598, 0.7560052489539643, 1.3439092613750225, 2.151945669890335],
                    4: [0.12839501671105813, 0.38804823445328507, 0.6567589957631145, 0.9423402689122875,
                        1.2562309480263467, 1.6180460517130526, 2.069016730231837, 2.732588804065177],
                    5: [0.06588962234909321, 0.1980516892038791, 0.3313780514298761, 0.4666991751197207,
                        0.6049331689395434, 0.7471351317890572, 0.89456439585444, 1.0487823813655852, 1.2118032120324,
                        1.3863389353626248, 1.576226389073775, 1.7872312118858462, 2.0287259913633036,
                        2.3177364021261493, 2.69111557955431, 3.260726295605043],
                    6: [0.0334094558802581, 0.1002781217139195, 0.16729660990171974, 0.23456656976873475,
                        0.3021922894403614, 0.37028193328115516, 0.4389488009177737, 0.5083127587538033,
                        0.5785018460645791, 0.6496542452315348, 0.7219204720694183, 0.7954660529025513,
                        0.870474868055092, 0.9471530930156288, 1.0257343133937524, 1.1064859596918581,
                        1.1897175711327463, 1.2757916223519965, 1.3651378971823598, 1.458272959944728,
                        1.5558274659528346, 1.6585847114298427, 1.7675371481292605, 1.8839718992293555,
                        2.009604894545278, 2.146803022259123, 2.2989727412973995, 2.471294740528467, 2.6722617014102585,
                        2.91739146530985, 3.2404166403241677, 3.7440690236964755],
                    7: [0.016828143177728235, 0.05049075396896167, 0.08417241989671888, 0.11788596825032507,
                        0.1516442630131618, 0.18546025708680833, 0.21934708340331643, 0.25331807190834565,
                        0.2873868062260947, 0.32156710392315796, 0.355873075050329, 0.39031926330596733,
                        0.4249205523979007, 0.4596922300454219, 0.49465018161031576, 0.5298108436256188,
                        0.565191195643323, 0.600808970989236, 0.6366826613981411, 0.6728315674936343,
                        0.7092759460939766, 0.746037126679468, 0.7831375375631398, 0.8206007832455021,
                        0.858451939611374, 0.896717615963322, 0.9354260757626341, 0.9746074842160436,
                        1.0142940678300427, 1.054520418037026, 1.0953237719213182, 1.1367442623434032,
                        1.1788252655205043, 1.2216138763870124, 1.26516137869917, 1.309523700469555, 1.3547621051156036,
                        1.4009441065262136, 1.448144252238147, 1.4964451375010575, 1.5459387008934842,
                        1.596727786313424, 1.6489283062238074, 1.7026711624156725, 1.7581051606756466,
                        1.8154009933798645, 1.8747553268072956, 1.9363967204122827, 2.0005932433837565,
                        2.0676621538384503, 2.1379832427349696, 2.212016460501213, 2.2903268704925304,
                        2.3736203164211713, 2.4627959084523208, 2.5590234991374485, 2.663867022558051,
                        2.7794919110540777, 2.909021527386642, 3.0572161028423737, 3.231896182843021,
                        3.4473810105937095, 3.7348571053691555, 4.1895219330235225],
                    8: [0.008445974137017219, 0.025338726226901278, 0.042233889994651476, 0.05913307399220878,
                        0.07603788791797023, 0.09294994306815242, 0.10987089037069565, 0.12680234584461386,
                        0.1437459285205906, 0.16070326074968388, 0.1776760066764216, 0.19466583496246115,
                        0.21167441946986007, 0.22870343946322488, 0.24575458029044564, 0.2628295721769575,
                        0.2799301528634766, 0.29705806782573063, 0.3142150709211129, 0.3314029639954903,
                        0.34862355883476864, 0.3658786774238477, 0.3831701926964899, 0.40049998943716425,
                        0.4178699650069057, 0.4352820704086704, 0.45273827097956804, 0.4702405882876,
                        0.48779106011037887, 0.505391740756901, 0.5230447441905988, 0.5407522460590347,
                        0.558516486141511, 0.5763396823538222, 0.5942241184949506, 0.6121721459546814,
                        0.6301861414640443, 0.6482685527755422, 0.6664219019236218, 0.684648787627676,
                        0.7029517931200633, 0.7213336286470308, 0.7397970881081071, 0.7583450032075904,
                        0.7769802937007926, 0.7957059197645721, 0.8145249861674053, 0.8334407494351099,
                        0.8524564651728141, 0.8715754936480047, 0.8908013031010308, 0.9101374749919184,
                        0.9295877653215154, 0.9491559977740125, 0.9688461234581733, 0.9886622867721733,
                        1.0086087121824747, 1.028689768268861, 1.0489101021225093, 1.0692743940997251,
                        1.0897875553561465, 1.1104547388972044, 1.1312812154370708, 1.1522725891384287,
                        1.173434599389649, 1.1947731980672593, 1.2162947131430126, 1.238005717146854,
                        1.2599130381874064, 1.2820237696510286, 1.304345369166531, 1.3268857708606756,
                        1.349653145284911, 1.3726560932224416, 1.3959037693197867, 1.419405726021264,
                        1.4431719292973744, 1.4672129964566984, 1.4915401336751468, 1.5161650628244996,
                        1.541100284490976, 1.5663591473033147, 1.5919556551358922, 1.6179046397057497,
                        1.6442219553485078, 1.6709244249695359, 1.6980300628044107, 1.7255580190748743,
                        1.7535288357430767, 1.7819645728459763, 1.81088895442524, 1.8403273195729115, 1.870306964218662,
                        1.9008577747790962, 1.9320118435829472, 1.9638039107009146, 1.9962716117712092,
                        2.0294560760505993, 2.0634026367482017, 2.0981611002741527, 2.133785932225919,
                        2.170336784741086, 2.2078803102947337, 2.2464908293749546, 2.286250990303635, 2.327254033532845,
                        2.369604977942217, 2.4134218838650208, 2.458840003415269, 2.506014300608167, 2.5551242195294983,
                        2.6063787537827645, 2.660023038604595, 2.716347847697055, 2.7757011083910723, 2.838504606698991,
                        2.9052776685316117, 2.976670770545963, 3.0535115393558603, 3.136880130166507,
                        3.2282236667414654, 3.3295406612081644, 3.443713971315384, 3.5751595986789093,
                        3.7311414987004117, 3.9249650523739246, 4.185630113705256, 4.601871059539151]}

def gen_boundaries(centroids):
  return [(a + b) / 2 for a, b in zip(centroids[:-1], centroids[1:])]

centroids = {i: [-j for j in reversed(c)] + c for i, c in opt_hn_centroids.items()}


boundaries = {i: gen_boundaries(c) for i, c in centroids.items()}


def MSE(x,y): 
    x_hat=[]
    b=[]
    for i in range(len(y)):
        a = y[i]    
        x_hat.append(a)
    for j in range(len(x)):
        y  = (1/len(x))*(x_hat[j]-x[j])**2

        b.append(y)
    return(np.sum(b))


def quantize_coordinates(coordinates, b):
    xyz = []
    floor_b = mt.floor(b) 
    fractional_part = b - floor_b
    quantization_levels = [mt.floor(b), mt.ceil(b)]
    probabilities = [1 - fractional_part, fractional_part]
    for coordinate in coordinates:
        quantized_value = random.choices(quantization_levels, probabilities)[0]
        xyz.append(quantized_value)
    return xyz

def random_sparsification(x, p):
    d = len(x)

    # Generate a random mask of size d * p
    mrs_size = int(d * p)
    mrs = np.zeros(d, dtype=int)
    mrs[:mrs_size] = 1

    np.random.shuffle(mrs)
    #print(mrs)

    # Perform coordinate-wise multiplication
    sparse_x = 1/p * mrs * x

    return sparse_x,mrs

def map_centroids_to_bits(centroids, bits, centroid_level):
    if centroid_level not in centroids or centroid_level not in bits:
        return {}

    centroid_mapping = {}
    centroid_values = centroids[centroid_level]
    binary_values = bits[centroid_level]

    for i in range(len(centroid_values)):
        centroid_mapping[centroid_values[i]] = binary_values[i]

    return centroid_mapping
binary = {
    1: ['0', '1'],
    2: ['00', '01', '10', '11'],
    3: ['000', '001', '011', '010', '110', '111', '101', '100'],# this is gray labelling
    #3: ['000', '001', '010', '011','100','101','110','111'],#this is binary
    #3: ['000', '001', '010', '011', '111', '110', '101', '100'],this is the best labelling  
    4: ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'],
    5: ['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010', '01011', '01100', '01101', '01110', '01111', '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111', '11000', '11001', '11010', '11011', '11100', '11101', '11110', '11111'],
    6: ['000000', '000001', '000010', '000011', '000100', '000101', '000110', '000111', '001000', '001001', '001010', '001011', '001100', '001101', '001110', '001111', '010000', '010001', '010010', '010011', '010100', '010101', '010110', '010111', '011000', '011001', '011010', '011011', '011100', '011101', '011110', '011111', '100000', '100001', '100010', '100011', '100100', '100101', '100110', '100111', '101000', '101001', '101010', '101011', '101100', '101101', '101110', '101111', '110000', '110001', '110010', '110011', '110100', '110101', '110110', '110111', '111000', '111001', '111010', '111011', '111100', '111101', '111110', '111111'],
    7: ['0000000', '0000001', '0000010', '0000011', '0000100', '0000101', '0000110', '0000111', '0001000', '0001001', '0001010', '0001011', '0001100', '0001101', '0001110', '0001111', '0010000', '0010001', '0010010', '0010011', '0010100', '0010101', '0010110', '0010111', '0011000', '0011001', '0011010', '0011011', '0011100', '0011101', '0011110', '0011111', '0100000', '0100001', '0100010', '0100011', '0100100', '0100101', '0100110', '0100111', '0101000', '0101001', '0101010', '0101011', '0101100', '0101101', '0101110', '0101111', '0110000', '0110001', '0110010', '0110011', '0110100', '0110101', '0110110', '0110111', '0111000', '0111001', '0111010', '0111011', '0111100', '0111101', '0111110', '0111111', '1000000', '1000001', '1000010', '1000011', '1000100', '1000101', '1000110', '1000111', '1001000', '1001001', '1001010', '1001011', '1001100', '1001101', '1001110', '1001111', '1010000', '1010001', '1010010', '1010011', '1010100', '1010101', '1010110', '1010111', '1011000', '1011001', '1011010', '1011011', '1011100', '1011101', '1011110', '1011111', '1100000', '1100001', '1100010', '1100011', '1100100', '1100101', '1100110', '1100111', '1101000', '1101001', '1101010', '1101011', '1101100', '1101101', '1101110', '1101111', '1110000', '1110001', '1110010', '1110011', '1110100', '1110101', '1110110', '1110111', '1111000', '1111001', '1111010', '1111011', '1111100', '1111101', '1111110', '1111111'],
    8: ['00000000', '00000001', '00000010', '00000011', '00000100', '00000101', '00000110', '00000111', '00001000', '00001001', '00001010', '00001011', '00001100', '00001101', '00001110', '00001111', '00010000', '00010001', '00010010', '00010011', '00010100', '00010101', '00010110', '00010111', '00011000', '00011001', '00011010', '00011011', '00011100', '00011101', '00011110', '00011111', '00100000', '00100001', '00100010', '00100011', '00100100', '00100101', '00100110', '00100111', '00101000', '00101001', '00101010', '00101011', '00101100', '00101101', '00101110', '00101111', '00110000', '00110001', '00110010', '00110011', '00110100', '00110101', '00110110', '00110111', '00111000', '00111001', '00111010', '00111011', '00111100', '00111101', '00111110', '00111111', '01000000', '01000001', '01000010', '01000011', '01000100', '01000101', '01000110', '01000111', '01001000', '01001001', '01001010', '01001011', '01001100', '01001101', '01001110', '01001111', '01010000', '01010001', '01010010', '01010011', '01010100', '01010101', '01010110', '01010111', '01011000', '01011001', '01011010', '01011011', '01011100', '01011101', '01011110', '01011111', '01100000', '01100001', '01100010', '01100011', '01100100', '01100101', '01100110', '01100111', '01101000', '01101001', '01101010', '01101011', '01101100', '01101101', '01101110', '01101111', '01110000', '01110001', '01110010', '01110011', '01110100', '01110101', '01110110', '01110111', '01111000', '01111001', '01111010', '01111011', '01111100', '01111101', '01111110', '01111111', '10000000', '10000001', '10000010', '10000011', '10000100', '10000101', '10000110', '10000111', '10001000', '10001001', '10001010', '10001011', '10001100', '10001101', '10001110', '10001111', '10010000', '10010001', '10010010', '10010011', '10010100', '10010101', '10010110', '10010111', '10011000', '10011001', '10011010', '10011011', '10011100', '10011101', '10011110', '10011111', '10100000', '10100001', '10100010', '10100011', '10100100', '10100101', '10100110', '10100111', '10101000', '10101001', '10101010', '10101011', '10101100', '10101101', '10101110', '10101111', '10110000', '10110001', '10110010', '10110011', '10110100', '10110101', '10110110', '10110111', '10111000', '10111001', '10111010', '10111011', '10111100', '10111101', '10111110', '10111111', '11000000', '11000001', '11000010', '11000011', '11000100', '11000101', '11000110', '11000111', '11001000', '11001001', '11001010', '11001011', '11001100', '11001101', '11001110', '11001111', '11010000', '11010001', '11010010', '11010011', '11010100', '11010101', '11010110', '11010111', '11011000', '11011001', '11011010', '11011011', '11011100', '11011101', '11011110', '11011111', '11100000', '11100001', '11100010', '11100011', '11100100', '11100101', '11100110', '11100111', '11101000', '11101001', '11101010', '11101011', '11101100', '11101101', '11101110', '11101111', '11110000', '11110001', '11110010', '11110011', '11110100', '11110101', '11110110', '11110111', '11111000', '11111001', '11111010', '11111011', '11111100', '11111101', '11111110', '11111111']
}

n_bits = 3
centroid_mapping = map_centroids_to_bits(centroids, binary, n_bits)
print(centroid_mapping)

def converter(decimal_array):
  bits_array=[]
  for value in decimal_array:
    closest_centroid = min(centroid_mapping.keys(), key=lambda x: abs(x - value))
    bits_array.append(centroid_mapping[closest_centroid])
  bits_array = np.array(bits_array)
  return bits_array

def binary_to_gray(binary_array):
    gray_array = []
    for binary in binary_array:
        gray = binary[0]
        for i in range(1, len(binary)):
            gray += str(int(binary[i]) ^ int(binary[i-1]))
        gray_array.append(gray)
    return gray_array
####flip prob 
def flip_bit(bit):
    if random.random() < 0.05:
        return '0' if bit == '1' else '1'
    else:
        return bit

def flip_gray_code(gray_array):
    flipped_gray_array = []
    for gray in gray_array:
        flipped_gray = ''.join(flip_bit(bit) for bit in gray)
        flipped_gray_array.append(flipped_gray)
    return flipped_gray_array

def gray_to_binary(gray_array):
    binary_array = []
    for gray in gray_array:
        binary = gray[0]  
        for i in range(1, len(gray)):
            binary += '1' if gray[i] != binary[i - 1] else '0'
        binary_array.append(binary)
    return binary_array


def load_dataset():
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=trf)
    val_set = datasets.MNIST(root='./data', train=False,
                             download=True, transform=trf)
    
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
            probabilities = nn.functional.softmax(yhat, dim=1)
            #print("Probabilities for batch:",probabilities)
            
    acc = crct / len(valloader.dataset)
    return loss, acc


model = LeNet5().to(DEVICE)
trainloader, valloader = load_dataset()
midpoints=[-2.151945669890335, -1.3439092613750225, -0.7560052489539643, -0.24509416307340598, 0.24509416307340598, 0.7560052489539643, 1.3439092613750225, 2.151945669890335]
counts = defaultdict(int)

def qunatization(params):
    M = 10
    m=2**M
    bits=[3]
    errors_for_each_user_bits=[]
    for user in range(rows):  
        x=params[user]
        errors_for_each_user=[]
        z_hat_est=[]
        for bit in bits:
            z_hat_big = []
            for i in range(0,1):
                if(bit<1):
                    p = random.randint(1, 9) / 10.0
                    sparse_x,mrs = random_sparsification(x, p)
                    rotation_matrix = np.random.randn(np.array(x).shape[0], np.array(x).shape[0])
                    rotation_matrix, _ = np.linalg.qr(rotation_matrix)                          
                    rotated_vector_1 = np.transpose(np.dot(rotation_matrix, x))
                    norm = np.linalg.norm(x, ord=2)
                    nita=((mt.sqrt(m))/norm)
                    rotated_vector=nita*rotated_vector_1
                    t_data = quantization_new_1(rotated_vector,centroids,boundaries,1)                      
                    t_data = np.array(t_data)
                    inner_product=np.dot(rotated_vector_1,t_data)
                    scaling_factor=(norm*norm)/inner_product
                    z_hat = scaling_factor*(np.transpose(np.dot(rotation_matrix.T, t_data)))
                    z_hat=z_hat*mrs*p
                    z_hat_big.append(z_hat)
                else:
                    rotation_matrix = np.random.randn(np.array(x).shape[0], np.array(x).shape[0])
                    rotation_matrix, _ = np.linalg.qr(rotation_matrix)   #rot matrix by QR 
                    #H = hadamard(m)                 
                    #D_values = np.random.choice([-1, 1], size=m)
                    #D = np.diag(D_values)
                    #rotation_matrix = np.dot(H, D)     #by R=HD
                    rotated_vector_1 = np.transpose(np.dot(rotation_matrix, x))
                    norm = np.linalg.norm(x, ord=2)
                    nita=((mt.sqrt(m))/norm)
                    rotated_vector=nita*rotated_vector_1
                    xyz = quantize_coordinates(rotated_vector,bit)
                    t_data=[]
                    for _ in range(0,m):
                        abc = quantization_new(rotated_vector[_],centroids,boundaries,xyz[_])
                        t_data.append(abc)                      
                    t_data=np.array(t_data)
                    #print("t_data:",t_data)
                    for element in t_data:
                        if element in midpoints:
                            counts[element] += 1    ### to plot the histogram 
                    bits_arr=converter(t_data)
                    #print("bits_array:",bits_arr)
                    #gray_array=binary_to_gray(bits_arr)
                    #print("gray coded",gray_array)
                    flipped_gray_array=flip_gray_code(bits_arr)
                    #print("flipped gray array",flipped_gray_array)
                    #binary_array=gray_to_binary(flipped_gray_array)
                    #print("final binary array ",binary_array)
                    inverse_centroid_mapping = {v: k for k, v in centroid_mapping.items()} 

                    centroid_array = []

                    for va in flipped_gray_array:
                        centroid_array.append(inverse_centroid_mapping[va])
                    #print(np.array_equal(t_data,centroid_array))
                    inner_product=np.dot(rotated_vector_1,t_data)
                    scaling_factor=(norm*norm)/inner_product
                    z_hat_temp = scaling_factor*(np.transpose(np.dot(rotation_matrix.T, centroid_array)))
                    z_hat_big.append(z_hat_temp)
                    #print(np.array(t_data).shape)
            z_hat1 = np.array(z_hat_big)

            z_hat_average_temp = np.mean(z_hat1, axis=0)
            #z_hat_est.append(z_hat_average_temp)
            errors_for_each_user.append(z_hat_average_temp)
        errors_for_each_user_bits.append(errors_for_each_user)
    errors_for_each_user_bits = np.array(errors_for_each_user_bits)
    #print("444:",errors_for_each_user_bits.shape)
    dec=errors_for_each_user_bits.flatten()
    print(dec.shape)
    for element, count in counts.items():
        print(f"Element {element}: Count {count}") #histogram values 
    # reconstructing the parameters into their corresponding shapes.
    revert = []
    ptr = 0
    for layer in model.parameters():
        size = layer.numel()
        print(dec[ptr: ptr+size].reshape(layer.shape).shape)
        revert.append(dec[ptr: ptr+size].reshape(layer.shape))
        ptr += size
    return revert

for element, count in counts.items():
    print(f"Element {element}: Count {count}")

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
        print("flat params shape:",flat_params.shape)
        # splitting parameters into 1024 batches each
        params = torch.cat([flat_params,
                            torch.zeros(1024 - (flat_params.numel() % 1024))]).view(-1, 1024)
        print(params.shape)
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
        print("[FIT, RECEIVED PARAMETERS FROM SEn RVER]")

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