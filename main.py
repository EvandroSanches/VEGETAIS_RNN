from keras.models import load_model, Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from statistics import stdev, mean
from pyod.models.knn import KNN
from pandasgui import show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

normalizador = MinMaxScaler()
timesteps = 30
epochs = 100
batch_size = 10

def CarregaDados():
    #Carregando dados
    dados = pd.read_csv('prices.csv')

    #Tratando dados
    dados["Price Dates"] = pd.to_datetime(dados['Price Dates'], format='%d-%m-%Y')
    #dados.index = dados["Price Dates"]
    dados = dados.drop("Price Dates", axis=1)
    dados = dados.drop("Tomato", axis=1)
    dados = dados.drop(index=148, axis=0)

    #Outliers(dados)

    #Definindo teste e treino
    train = dados[:]
    teste = dados[dados.shape[0] - 30:]

    #Normalizando dados
    train_normalizados = normalizador.fit_transform(train)
    teste_normalizados = normalizador.transform(teste)

    previsores_train = []
    preco_real_train = []

    previsores_teste = []

    #Estruturando dados para LSTM
    for i in range(timesteps, train_normalizados.shape[0]):
        previsores_train.append(train_normalizados[i - timesteps : i, :])
        preco_real_train.append(train_normalizados[i, :])

    for i in range(timesteps, teste_normalizados.shape[0]+1):
        previsores_teste.append(teste_normalizados[i - timesteps : i, :])

    previsores_train = np.asarray(previsores_train)
    preco_real_train = np.asarray(preco_real_train)

    previsores_teste = np.asarray(previsores_teste)

    return (previsores_train, preco_real_train), (previsores_teste, dados)


def CriaRede():
    #EStrutura da Rede Neural LSTM
    modelo = Sequential()

    modelo.add(LSTM(units=100, return_sequences=True, input_shape=(timesteps, 9)))
    modelo.add(Dropout(0.1))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.1))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.1))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.1))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.1))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.1))

    modelo.add(LSTM(units=80))
    modelo.add(Dropout(0.1))

    modelo.add(Dense(units=9, activation='linear'))

    modelo.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return modelo

def Treinamento():
    (previsores, preco_real), (_, _) = CarregaDados()

    #Definindo callbacks
    ers = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta= 1e-10)
    rlp = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1)
    mc = ModelCheckpoint(monitor='loss' ,filepath='Modelo.0.1', save_best_only=True, verbose=1)

    modelo = KerasRegressor(build_fn=CriaRede,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[ers, rlp, mc])

    #Treinando modelo com validação cruzada
    resultado = cross_val_score(estimator=modelo, X=previsores, y=preco_real, cv=10, scoring='neg_mean_absolute_error')

    #Visualizando resultados
    media = mean(resultado)
    desvio = stdev(resultado)

    plt.plot(resultado)
    plt.title('Relação de Perda\n'+'Média:'+str(media)+'\nDesvio Padrão:'+str(desvio))
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()


def Outliers(dados):
    dados_normalizados = normalizador.fit_transform(dados)

    detector = KNN()
    detector.fit(dados_normalizados)

    #Atribuindo classificações a variavel previsores
    previsores = detector.labels_

    outliers = []

    #Pegando indice dos valores classificados como outlier
    for i in range(len(previsores)):
        if previsores[i] == 1:
            outliers.append(i)

    dados = dados.iloc[outliers, :]

    show(dados)

def Previsao():
    #Carregando dados
    (previsores_treino, _), (previsores, dados) = CarregaDados()

    #Carregando modelo
    modelo = load_model('Modelo.0.1')

    #Fazendo predição e desnormalizando dados
    resultado = modelo.predict(previsores_treino)
    resultado = normalizador.inverse_transform(resultado)

    dados = dados[30:]
    dados = dados.reset_index()
    dados = dados.drop('index', axis=1)
    i = 0

    #Mostrando resultados de cada iten
    for coluna in dados.columns:
        plt.plot(resultado[:,i])
        plt.plot(dados[coluna])
        plt.title(str(coluna))
        plt.xlabel('Tempo')
        plt.ylabel('Valor')
        plt.legend(['Previsão', 'Preço real'])
        plt.show()
        i = i+1

Previsao()
