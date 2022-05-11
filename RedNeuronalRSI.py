"""
pandas.- pandas es una herramienta de manipulación y análisis de datos de código abierto rápida, potente, flexible y fácil de usar, construida sobre el lenguaje de programación Python.
numpy.- es una librería de Python especializada en el cálculo numérico y el análisis de datos, especialmente para un gran volumen de datos. Incorpora una nueva clase de objetos llamados arrays que permite representar colecciones de datos de un mismo tipo en varias dimensiones, y funciones muy eficientes para su manipulación.
tensorflow.- La principal biblioteca de código abierto para enseñarte a desarrollar y entrenar modelos de AA. Comienza enseguida y ejecuta notebooks de Colab directamente en tu navegador.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

print("Comenzando entrenamiento  la compra...")
InputNet1=pd.read_csv('EURUSDInputNet5Min.csv', delimiter=';',header=None)  #Lee un archivo de valores separados (csv).  Minimos del dia - Precios
OutNet1=pd.read_csv('EURUSDOutputNet5Min.csv', delimiter=';',header=None) #Lee un archivo como termina 

Net1Min = Sequential()  #Un Sequential modelo es apropiado para una simple pila de capas donde cada capa tiene exactamente un tensor de entrada y un tensor de salida .
Net1Min.add(Dense(21, activation='relu', input_shape=(InputNet1.shape[1],))) #Dense: capas completamente conectadas con activación ReLU.  Ingresaron los 21 datos.
Net1Min.add(Dense(50))  #Dense 
Net1Min.add(Dense(50))  #Dense
Net1Min.add(Dense(60))  #Dense
Net1Min.add(Dense(1))  #Salida

Net1Min.compile(optimizer='adam', loss='mse', metrics=['mse']) #An optimizer is one of the two arguments required for compiling a Keras model:
print(Net1Min.summary())

print("Comenzando entrenamiento...")
historial = Net1Min.fit(InputNet1,OutNet1, epochs=40, batch_size=10,verbose=False,validation_split=0.3)#validation_split=0.3
print("Modelo entrenado!")
#InputNet1  Datos de Entrada
#InputNet2OutNet1 Datos de Salida
#epochs  Repeticiones para validaciones
#batch_size  Puebras en paralelo
#verbose  Me mostrara como va avanzando
#validation_split Reservacion para validar datos

print("Presentación Gráfica...")
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Guardando el modelo...")
Net1Min.save('net1Max5Min.h5')
#####################################################################
print("Comenzando entrenamiento  la venta...")
InputNet2=pd.read_csv('EURUSDInputNet5Max.csv', delimiter=';',header=None)  #Lee un archivo de valores separados (csv).  Minimos del dia - Precios
OutNet2=pd.read_csv('EURUSDOutputNet5Max.csv', delimiter=';',header=None) #Lee un archivo como termina

Net2Min = Sequential()  #Un Sequential modelo es apropiado para una simple pila de capas donde cada capa tiene exactamente un tensor de entrada y un tensor de salida .
Net2Min.add(Dense(21, activation='relu', input_shape=(InputNet1.shape[1],))) #Dense: capas completamente conectadas con activación ReLU.  Ingresaron los 21 datos.
Net2Min.add(Dense(50))  #Dense 
Net2Min.add(Dense(50))  #Dense
Net2Min.add(Dense(60))  #Dense
Net2Min.add(Dense(1))  #Salida

Net2Min.compile(optimizer='adam', loss='mse', metrics=['mse']) #An optimizer is one of the two arguments required for compiling a Keras model:
print(Net2Min.summary())

print("Comenzando entrenamiento...")
historial = Net2Min.fit(InputNet2,OutNet2, epochs=40, batch_size=10,verbose=False,validation_split=0.3)#validation_split=0.3
print("Modelo entrenado!")

print("Presentación Gráfica...")
import matplotlib.pyplot as pls
pls.xlabel("# Epoca")
pls.ylabel("Magnitud de pérdida")
pls.plot(historial.history["loss"])
pls.show()

print("Guardando el modelo...")
Net1Min.save('netsMin5Min.h5')

print("Finalizado...")
input('Press ENTER to exit') 