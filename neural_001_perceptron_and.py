
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

# Descrição da Aplicação: Função AND com perceptron
# Data: 06/01/2021
# Referência: Redes Neurais Artificiais em Python (Aula: 14)
# Instrutor: Jones Granatyr


############################################################################## 
#                                   Imports                                  #
############################################################################## 

import numpy as np


############################################################################## 
#                                    Dados                                   #
############################################################################## 

entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([0, 0, 0, 1])
pesos = np.array([0.0, 0.0])

taxaAprendizagem = 0.1


############################################################################## 
#                             Função de Ativação                             #
############################################################################## 

def stepFunction(y):
    if(y >= 1):
        return 1
    return 0


##############################################################################
#                           Aplicação do Perceptron                          #
##############################################################################

def calcularSaida (registro):
    s = registro.dot(pesos)
    return stepFunction(s)


##############################################################################     
#                               Ajuste dos pesos                             #
############################################################################## 

def ajuste():
    etapa = 0
    erroTotal = 1
    while(erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)): # len(saidas) = 4
            saida_n = calcularSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saida_n)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * 
                                       erro)
                print("Peso = " + str(pesos[j])) # str: converte para string
        print("Total de erros = " + str(erroTotal))
        etapa += 1
        print ("Etapa finalizada: ", etapa)
        print ("\n")

ajuste()    

print('Rede neural treinada')
print(calcularSaida(entradas[0]))
print(calcularSaida(entradas[1]))
print(calcularSaida(entradas[2]))
print(calcularSaida(entradas[3]))                
            
            
    




    