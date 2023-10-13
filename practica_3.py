import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from mpl import NeuralNet

#*Lee el archivo csv y lo divide en entradas y salidas
def readCsv(file):

    data = pd.read_csv(file)
    inputs = data.iloc[:,:2]
    outputs = data.iloc[:,2:]
    #*Lo divide en 80% para entrenamiento, 20% para testing
    inTrain, inTest, outTrain, outTest = train_test_split(inputs, outputs, test_size=0.2, random_state=2)

    return inputs, outputs, inTrain, inTest, outTrain, outTest

#*Creamos las variables y tensores para poder crear la red neuronal
inputs, outputs, inTrain, inTest, outTrain, outTest = readCsv("concentlite.csv")

tenInTrain = torch.from_numpy(inTrain.values).float().to("cpu")
tenInTest = torch.from_numpy(inTest.values).float().to("cpu")
tenOutTrain = torch.from_numpy(outTrain.values).float().to("cpu")
tenOutTest = torch.from_numpy(outTest.values).float().to("cpu")

#*Creamos dos funciones que van a convertir nuestras salidas y predicciones 
#* -1 = 0 y 1 = 1
def transformOutput(out):
    return torch.where(out == -1, torch.tensor(0.0), torch.tensor(1.0))

def transformPrediction(pred):
    return torch.where(pred > 0.5, torch.tensor(1.0), torch.tensor(-1.0))

#*Variables para entrenar la red neuronal
#* learning rate, funcion de perdida, epocas y optimizador para el gradiente descendiente
learningRate = 0.001
epochs = 5000
lossFun = nn.BCELoss()

myNet = NeuralNet(2, 3, 4, 1)
optim = torch.optim.Adam(params=myNet.parameters(), lr=learningRate)

#!Comienza a hacer el entrenamiento de la red neuronal
print(f"Comienza el entrenamiento con {epochs} epocas")
for i in range(epochs):
    #*Obtiene la prediccion
    prediction = myNet(tenInTrain)
    #*Se calcula el error
    perdida = lossFun(prediction, transformOutput(tenOutTrain))
    #*Hace el backpropagation y reajusta los pesos
    perdida.backward()
    optim.step()
    optim.zero_grad()

#!Hace el testing de la red neuronal
prediction = myNet(tenInTest)

predictObtenido = transformPrediction(prediction)

#*Grafico para las predicciones correctas
plt.scatter(
    tenInTest[predictObtenido.squeeze() == tenOutTest.squeeze()][:, 0],
    tenInTest[predictObtenido.squeeze() == tenOutTest.squeeze()][:, 1],
    marker='x',
    c='g'
)

#*Grafico para las predicciones incorrectas
plt.scatter(
    tenInTest[predictObtenido.squeeze() != tenOutTest.squeeze()][:, 0],
    tenInTest[predictObtenido.squeeze() != tenOutTest.squeeze()][:, 1],
    marker='s',
    c='r'
)

plt.show()