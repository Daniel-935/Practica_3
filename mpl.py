import torch
import torch.nn as nn
import numpy as np
import pandas as pd

#!Usaremos pytorch para crear la red neuronal, se crea clase que toma como padre a "Module"
class NeuralNet(nn.Module):

    #*A traves del constructor se definen las entradas, salidas, neuronas y capas
    def __init__(self, noInputs, noCapas, noNeuronas, noOutput):
        #*Usa el constructor de la clase padre
        super(NeuralNet, self).__init__()

        #*Crea la primer capa de entrada y de salida
        self.capaInput = nn.Linear(noInputs, noNeuronas)
        self.capaOutput = nn.Linear(noNeuronas, noOutput)

        #*Crea las capas ocultas y las almacena en una lista de torch
        #*Cada capa del tipo lineal
        self.capasOcultas = nn.ModuleList()
        for i in range(noCapas):
            self.capasOcultas.append(nn.Linear(noNeuronas, noNeuronas))

    #*Metodo para hacer el feedforward
    def forward(self, inputs):
        #*Comienza por recorrer todas las capas ocultas y usar la funcion de activacion con el fin de obtener una prediccion
        #*Inicia con la capa de entrada y recorre la lista creada de torch, se va a usar la funcion sigmoide
        prediction = torch.sigmoid(self.capaInput(inputs))
        for capa in self.capasOcultas:
            prediction = torch.sigmoid(capa(prediction))
        #*Obtiene la prediccion final (capa de salida)
        prediction = torch.sigmoid(self.capaOutput(prediction))
        return prediction