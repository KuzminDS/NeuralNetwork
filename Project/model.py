import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import keras
from keras import layers
from keras import models


def prepare_data_from_csv(data):
	"""Подготовка данных для обучения"""
	data = data.drop(['filename'], axis=1) #Очищаем таблицу от ненужных данных

	genre_list = data.iloc[:, -1] #Получаем список жанров из последнего стобца
	encoder = LabelEncoder()
	y = encoder.fit_transform(genre_list) #Кодируем жанры

	scaler = StandardScaler()
	X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float)) #Из оставшихся данных формируем данные для обучения

	return train_test_split(X, y, test_size=0.2) #Разделяем обучаемые и тестируемые данные в соотношении 80%/20% соответственно


def create_model(X_train, X_test, y_train, y_test):
	"""Создание модели"""
	model = models.Sequential()
	model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(3, activation='softmax'))

	model.compile(optimizer='adam', 
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	return model


def print_plot(model, X_train, X_test, y_train, y_test):
	"""Выводим график обучения"""
	history = model.fit(X_train, 
					y_train,
					epochs=40,
					batch_size=32,
					validation_data=(X_test, y_test))

	test_loss, test_acc = model.evaluate(X_test, y_test)
	print('test_acc: ', test_acc)

	history_dict = history.history

	loss = history_dict['loss']
	val_loss = history_dict['val_loss']

	epochs = range(1, len(loss) + 1)

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
	plt.clf()


def save_model(model):
	"""Сохраняем модель в файле"""
	os.chdir('..\\Data\\model')
	model_file = 'model.h5'
	model.save(model_file)


os.chdir('..\\Data')
data_path = os.getcwd() # Получаем нужную директорию

data = pd.read_csv(os.path.join(data_path, 'data', 'data.csv')) #Считываем таблицу

X_train, X_test, y_train, y_test = prepare_data_from_csv(data)

test_model = create_model(X_train, X_test, y_train, y_test)

print_plot(test_model, X_train, X_test, y_train, y_test)

model = create_model(X_train, X_test, y_train, y_test)

model.fit(X_train, 
		  y_train,
		  epochs=100,
		  batch_size=32)

predictions = model.predict(X_test) #Выводим предсказания
for i in range(10):
	print(str(np.argmax(predictions[i]))+" "+str(y_test[i]))

test_loss, test_acc = model.evaluate(X_test, y_test) #Выводим точность обученной модели
print(f'test_acc: {test_acc*100}%') 

# save_model(model)