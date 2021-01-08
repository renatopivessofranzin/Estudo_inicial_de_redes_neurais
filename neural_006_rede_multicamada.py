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

# Descrição da Aplicação: Implementação de uma rede multicamada 
# Data: 07/01/2021
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

# camada entrada: entrada (2x3); -> x(entrada) -> 2; oculta -> 3
# camada oculta: oculta (3x1);  oculta -> 3;  saída - 1

# Matriz 2x3 (valores: -1 a +1)
pesosCamadaEntrada = 2*np.random.random((2,3)) - 1 
# Matriz 3x1 (valores: -1 a +1)
pesosCamadaOculta = 2*np.random.random((3,1)) - 1 

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
    
    # camada oculta
    camadaEntrada = entradas
    somaCamadaEntrada = np.dot(camadaEntrada, pesosCamadaEntrada)
    camadaOculta = sigmoid(somaCamadaEntrada)
    
    # camada de saída
    somaCamadaOculta = np.dot(camadaOculta, pesosCamadaOculta)
    camadaSaida = sigmoid(somaCamadaOculta) ## saída (y) ##
    
    
    # cálculo do erro
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
    
    
    # ajuste dos pesos
    
    # cálculo do delta de saída
    derivadaSaida = derivadaSigmoid(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    
     # cálculo do delta da camada oculta
    pesosCamadaOculta_transp = pesosCamadaOculta.T # Matriz Transposta
    deltaSaida_x_peso = deltaSaida.dot(pesosCamadaOculta_transp)
    deltaCamadaOculta = deltaSaida_x_peso * derivadaSigmoid(camadaOculta)
    
    
    ## camada oculta ##
    camadaOcultaTransp = camadaOculta.T
    pesosNovosOculta = camadaOcultaTransp.dot(deltaSaida)
    
    # valores de pesos atualizados (camada 2)
    pesosCamadaOculta = (pesosCamadaOculta * momento) + (pesosNovosOculta * 
                                                   taxaAprendizagem)
    
    ## camada entrada ##
    camadaEntradaTransp = camadaEntrada.T
    pesosNovosEntrada = camadaEntradaTransp.dot(deltaCamadaOculta)
    
    # valores de pesos atualizados (camada 1)
    pesosCamadaEntrada = (pesosCamadaEntrada * momento) + (pesosNovosEntrada * 
                                                   taxaAprendizagem)
    
    