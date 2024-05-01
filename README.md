# Skin Disease Detection with Neural Networks

Este repositório contém um projeto de detecção de doenças de pele usando redes neurais.

## Descrição

O projeto visa desenvolver um sistema de detecção de doenças de pele utilizando técnicas de inteligência artificial, especificamente redes neurais convolucionais (CNNs). O objetivo é criar um modelo capaz de analisar imagens dermatológicas e identificar diferentes tipos de doenças de pele com precisão.

O conjunto de dados utilizado para treinar e testar o modelo foi obtido a partir do seguinte link: [Skin Disease Classification Image Dataset](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset).

Além disso, o projeto inclui um bot do Telegram para facilitar o processo de diagnóstico. Os usuários podem enviar imagens de lesões cutâneas para o bot, que então utiliza o modelo de rede neural para realizar a análise e fornecer uma avaliação preliminar.

## Conteúdo

O repositório está organizado da seguinte forma:

- `Doenças`: Pasta contendo informações sobre as principais doenças de pele que serão identificadas pelo modelo, incluindo Ceratose Actínica, Dermatite Atópica, Verruga Seborreica, Dermatofibra, Nevo Melanocítico, Carcinoma de Células Escamosas e Lesão Vascular.
- `app`: Pasta contendo o código-fonte do aplicativo para detecção de doenças de pele.
- `TelegramBot.py`: Arquivo com o código-fonte do bot do Telegram.
- `requirements.txt`: Arquivo contendo as dependências do projeto.
- `referencias.txt`: Arquivo com as referências utilizadas no projeto.
- `PassosParaExec.txt`: Arquivo com instruções sobre como executar o projeto.

## Como executar

Para executar o projeto, siga estas etapas:

1. Clone este repositório para sua máquina local.
2. Instale as dependências listadas no arquivo `requirements.txt` usando o comando `pip install -r requirements.txt`.
3. Execute o arquivo `TelegramBot.py` para iniciar o bot do Telegram.
4. Siga as instruções no aplicativo para enviar imagens e receber diagnósticos.

## Referências

- [Esteva, A. et al. Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118, 2017. DOI: 10.1038/nature21056](https://doi.org/10.1038/nature21056)
- [Brinker, T. J. et al. Deep learning outperformed 136 of 157 dermatologists in a head-to-head dermoscopic melanoma image classification task. *European Journal of Cancer*, 103, 114-121, 2018.](#)
- [Tschandl, P. et al. Human-computer collaboration for skin cancer recognition. *Nature Medicine*, 25(8), 1215-1218, 2019.](#)
- [Goodfellow, I. et al. *Deep Learning*. MIT Press, 2016.](#)
- [LeCun, Y. et al. Deep learning. *Nature*, 521(7553), 436-444, 2015.](#)
- [Haenssle, H. A. et al. Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists. *Annals of Oncology*, 29(8), 1836-1842, 2018. DOI: 10.1093/annonc/mdy166](https://doi.org/10.1093/annonc/mdy166)
- [Nóbrega, M. M. et al. Diagnóstico das principais dermatoses comuns em clínica geral. *Anais Brasileiros de Dermatologia*, 84(3), 257-270, 2009.](#)
- [Marghoob, A. A. et al. *Dermoscopy: The Essentials*. Elsevier, 2017.](#)
- [Goldsmith, L. A. et al. *Fitzpatrick's Dermatology in General Medicine*. McGraw Hill Professional, 8ª edição, 2012.](#)
- [Rajpurkar, P. et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv preprint arXiv:1711.05225*, 2017.](#)
- [Narla, A., Patel, R., & Jonmichael, H. DermBot: A Chatbot for Early Detection of Skin Cancer Using Deep Learning. *arXiv preprint arXiv:2004.03801*.](#)
