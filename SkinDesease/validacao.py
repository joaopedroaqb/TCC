import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

torch.manual_seed(123)

class classificador(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)) 
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=14*14*64, out_features=256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 7)

    def forward(self, X):
        X = self.pool(self.bnorm(self.activation(self.conv1(X))))
        X = self.pool(self.bnorm(self.activation(self.conv2(X))))
        X = self.flatten(X)

        # Camadas densas
        X = self.activation(self.linear1(X))
        X = self.activation(self.linear2(X))
        
        # Saída
        X = self.output(X)

        return X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classificadorLoaded = classificador()
state_dict = torch.load('checkpoint.pth', map_location=device)  # Carregando o modelo treinado
classificadorLoaded.load_state_dict(state_dict)
classificadorLoaded.eval()

def classificarImagem(nome):
    imagem_teste = Image.open(nome)
    imagem_teste = imagem_teste.resize((64, 64))
    imagem_teste = imagem_teste.convert('RGB') 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagem_teste = transform(imagem_teste)
    imagem_teste = imagem_teste.unsqueeze(0)  # Adiciona uma dimensão de lote
    imagem_teste = imagem_teste.to(device)
    output = classificadorLoaded(imagem_teste)
    output = F.softmax(output, dim=1)
    resultado = torch.argmax(output, dim=1).item()
    doencas = ['Ceratose Actínica', 'Dermatite Atópica', 'Verruga Seborreica','Dermatofibra', 'Nevo Melanocítico', 'Carcinoma de Células Escamosas', 'Lesão Vascular']
    print('Nome Imagem: ' + nome.split('/')[-1] + f' Resultado: {doencas[resultado]}')

pasta = "val"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
print(caminhos)
for caminho in caminhos:
    file_list = os.listdir(caminho)
    for imagem in file_list:
        classificarImagem(os.path.join(caminho, imagem))
