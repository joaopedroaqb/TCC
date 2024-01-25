import os
import telebot
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Configuração do bot
bot = telebot.TeleBot("TOKEN_AQUI")

# Configuração do modelo de classificação
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
    return doencas[resultado]

# Lista de caminhos das imagens a serem classificadas
pasta = "val"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]

# Função para classificar todas as imagens em uma pasta
def classificarTodasImagens():
    for caminho in caminhos:
        file_list = os.listdir(caminho)
        for imagem in file_list:
            resultado = classificarImagem(os.path.join(caminho, imagem))
            print('Nome Imagem: ' + imagem + f' Resultado: {resultado}')

# Comando para lidar com as imagens enviadas pelos usuários
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        # Obtém o ID do chat
        chat_id = message.chat.id

        # Obtém o arquivo da imagem
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Salva a imagem localmente
        image_path = "temp_image.jpg"
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Classifica a imagem
        resultado = classificarImagem(image_path)

        # Envia a resposta de volta ao usuário
        bot.reply_to(message, f"A imagem foi classificada como: {resultado}\n\n\nPara classificar outra imagem digite ou clique em /classificar \n\nPara retornar ao Menu digite ou clique em /menu.")

        # Remove a imagem temporária
        os.remove(image_path)

    except Exception as e:
        print(e)
        bot.reply_to(message, "Ocorreu um erro ao processar a imagem.")

# Comando para lidar com o comando /classificar
@bot.message_handler(commands=["classificar"])
def conversao(mensagem):
      bot.send_message(mensagem.chat.id,"Você escolheu classificar. Por favor, envie a imagem para ser classificada.")

# Comando para lidar com o comando /menu
@bot.message_handler(commands=["menu"])
def responder(mensagem):
    texto = "Olá! \n\nAqui sou um classificador de doenças pulmonares. Para ter um diagnóstico de uma imagem pulmonar, digite ou clique em /classificar.  \n\nCaso precise de ajuda, digite ou clique em /help."
    bot.reply_to(mensagem, texto)

# Comando para lidar com o comando /help
@bot.message_handler(commands=["help"])
def responder(mensagem):
    texto = "Você solicitou ajuda. O classificador aceita imagens para fazer a classificação de doenças pulmonares. Ele classifica quatro tipos de doenças e faz a classificação de imagens normais (imagens sem presença de doença). Para classificar uma imagem, digite ou clique em /classificar."
    bot.reply_to(mensagem, texto)

# Função para verificar mensagens (pode ser personalizada conforme necessário)
def verificar(mensagem):
    return True

# Resposta padrão para mensagens não reconhecidas
@bot.message_handler(func=verificar)
def responder(mensagem):
    texto = "Olá! \n\nAqui sou um classificador de doenças pulmonares. Para ter um diagnóstico de uma imagem pulmonar, digite ou clique em /classificar.  \n\nCaso precise de ajuda, digite ou clique em /help."
    bot.reply_to(mensagem, texto)

# Inicia o bot
bot.polling(none_stop=True)
