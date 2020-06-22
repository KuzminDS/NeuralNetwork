import os
import csv
import keras
from extract import extract_features
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def decoding(label):
	"""Расшифровка меток"""
	if label == 0:
		return 'classical'
	elif label == 1:
		return 'pop'
	else:
		return 'rock'


#Загружаем модель из файла
os.chdir('..\\Data\\model')
model = keras.models.load_model('model.h5')

os.chdir('..\\samples')
path = os.getcwd()

tracks = []
tracks_features = []

for filename in os.listdir(os.path.join(path)): #Заполняем список tracks данными, полученными из тестируемых треков
	songname = os.path.join(path, f'{filename}')
	features = extract_features(songname).split()
	tracks_features.append(np.array(features, dtype=float))
	tracks.append(filename)

scaler = StandardScaler()
tracks_features = scaler.fit_transform(np.array(tracks_features, dtype=float)) #Подготавливаем данные для модели

predictions = model.predict(tracks_features) #Формирование предсказаний

for item in enumerate(predictions): #Вывод предсказаний
	print(f'{tracks[item[0]]} - {decoding(np.argmax(item[1]))} : [classical - {int(item[1][0]*100)}% pop - {int(item[1][1]*100)}% rock - {int(item[1][2]*100)}%]')