# Projeto: 
# Estudo de implementação de uma rede neural artificial em Python
# Autor: Prof. Ms. Renato Pivesso Franzin

# Objetivos:
    
"""
* Aprender passo a passo todos os cálculos matemáticos que envolvem redes 
neurais artificiais;

* Entender conceitos como perceptron, funções de ativação, 
backpropagation (retropropagação) e gradient descent 
(descida do gradiente).

"""

# Descrição da Aplicação: Implementação de rede com 3 camadas ocultas
# Data: 08/01/2021
# Referência: Redes Neurais Artificiais em Python (Aula: 24)
# Instrutor: Jones Granatyr


############################################################################## 
#                                   Imports                                  #
############################################################################## 

import numpy as np


############################################################################## 
#                                    Dados                                   #
############################################################################## 

entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) # valores de x
saidas = np.array([[0],[1],[1],[0]]) # valores de y


pesosEntrada = 2*np.random.random((2,6)) - 1 # Matriz 2x6 (valores: -1 a +1)
pesosOculta_1 = 2*np.random.random((6,6)) - 1 # Matriz 6x6 (valores: -1 a +1)
pesosOculta_2 = 2*np.random.random((6,6)) - 1 # Matriz 6x6 (valores: -1 a +1)
pesosOculta_3 = 2*np.random.random((6,1)) - 1 # Matriz 6x1 (valores: -1 a +1)

taxaAprendizagem = 0.1
momento = 1


############################################################################## 
#                               Função Sigmoid                               #
############################################################################## 

def sigmoid(soma):
    return 1/(1 + np.exp(-soma))


############################################################################## 
#                        Função derivada Sigmoid                             #
############################################################################## 

def derivadaSigmoid(sig):
    return sig * (1 - sig)


############################################################################## 
#                              Rede multicamada                              #
############################################################################## 

epocas = 100000

for j in range(epocas):
    
    # camada oculta 1
    camadaEntrada = entradas
    somaCamad_01 = np.dot(camadaEntrada, pesosEntrada)
    camadaOculta_01 = sigmoid(somaCamad_01)
    
    # camada oculta 2
    somaCamad_02 = np.dot(camadaOculta_01, pesosOculta_1)
    camadaOculta_02 = sigmoid(somaCamad_02)
    
    # camada oculta 3
    somaCamad_03 = np.dot(camadaOculta_02, pesosOculta_2)
    camadaOculta_03 = sigmoid(somaCamad_03)
    
    # camada de saída
    somaCamad_04 = np.dot(camadaOculta_03, pesosOculta_3)
    camadaSaida = sigmoid(somaCamad_04) ## saída (y) ##
    

    # cálculo do erro
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
 
   
    # ajuste dos pesos
    
    # cálculo do delta da camada de saída
    derivadaSaida = derivadaSigmoid(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    
    # cálculo do delta da camada oculta 3
    pesosOculta_3T = pesosOculta_3.T # Matriz Transposta
    deltaSaida_x_peso_03 = deltaSaida.dot(pesosOculta_3T)
    delta_03 = deltaSaida_x_peso_03 * derivadaSigmoid(camadaOculta_03)
    
    
    # cálculo do delta da camada oculta 2
    pesosOculta_2T = pesosOculta_2.T # Matriz Transposta
    deltaSaida_x_peso_02 =  delta_03.dot(pesosOculta_2T)
    delta_02 = deltaSaida_x_peso_02 * derivadaSigmoid(camadaOculta_02)
    
    
    # cálculo do delta da camada oculta 1
    pesosOculta_1T = pesosOculta_1.T # Matriz Transposta
    deltaSaida_x_peso_01 = delta_02.dot(pesosOculta_1T)
    delta_01 = deltaSaida_x_peso_01 * derivadaSigmoid(camadaOculta_01)
    

    # valores de pesos atualizados (camada oculta 3)
    camadaOculta_03T = camadaOculta_03.T
    pesosOculta_3New =  camadaOculta_03T.dot(deltaSaida)
    pesosOculta_3 = (pesosOculta_3 * momento) + (pesosOculta_3New * 
                                                   taxaAprendizagem)


    # valores de pesos atualizados (camada oculta 2)
    camadaOculta_02T = camadaOculta_02.T
    pesosOculta_2New =  camadaOculta_02T.dot(delta_03)
    pesosOculta_2 = (pesosOculta_2 * momento) + (pesosOculta_2New * 
                                                   taxaAprendizagem)
    
    
    # valores de pesos atualizados (camada oculta 1)
    camadaOculta_01T = camadaOculta_01.T
    pesosOculta_1New =  camadaOculta_01T.dot(delta_02)
    pesosOculta_1 = (pesosOculta_1 * momento) + (pesosOculta_1New * 
                                                   taxaAprendizagem)
    
    
    # valores de pesos atualizados (camada de entrada)
    camadaEntradaT = camadaEntrada.T
    pesosEntradaNew = camadaEntradaT.dot(delta_01)
    pesosEntrada = (pesosEntrada * momento) + (pesosEntradaNew * 
                                                   taxaAprendizagem)
    