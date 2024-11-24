# DigitAI

DigitAI é um projeto em Python que utiliza TensorFlow para reconhecer números escritos à mão. O modelo de rede neural, previamente treinado e salvo no arquivo "mnist_model.keras", foi treinado com o conjunto de dados MNIST e está pronto para uso imediato.

O principal destaque deste repositório é uma interface gráfica interativa, desenvolvida com Pygame, que permite ao usuário desenhar um dígito diretamente na tela e obter a predição do modelo ao clicar no botão "Predict".

## Arquitetura da Rede Neural

O modelo de rede neural para reconhecimento de dígitos manuscritos no DigitAI é uma rede feedforward sequencial com três camadas densas (ou totalmente conectadas) e uma camada de entrada de flatten. Essa arquitetura é simples e eficiente para tarefas de classificação de imagens pequenas, como o conjunto de dados MNIST, e é composta pelas seguintes camadas:

- Entrada (Flatten): Converte a imagem 28x28 em um vetor de 784 valores.
- Primeira Camada Oculta: 128 neurônios com ativação ReLU.
- Segunda Camada Oculta: 64 neurônios com ativação ReLU.
- Saída: 10 neurônios com ativação Softmax para prever as classes de 0 a 9.

![DigitAI_UI](https://github.com/Leonardo-Nunes-Armelim/DigitAI/blob/main/UI.png)

## Setup

Python 3.10.11

```python
python -m venv ./venv
.\venv\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## Rodando UI
```python
python DigitAI_UI.py
```
