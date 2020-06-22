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
from keras.utils.vis_utils import plot_model

from IPython.display import SVG
from keras.utils import model_to_dot

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

os.chdir('..\\Data')
data_path = os.getcwd()

data = pd.read_csv(os.path.join(data_path, 'data', 'data.csv'))
print(data.head())

#Dropping unnecessary colums
data = data.drop(['filename'], axis=1)
print(data.head())

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)

#Normalizing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Creating a model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', 
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(X_train, y_train,
		  epochs=20,
		  batch_size=32)

plot_model(model, to_file='C:\\Python\\NN\\model_plot.png', show_shapes=True, show_layer_names=True)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc: ', test_acc)

predictions = model.predict(X_test)
for i in range(60):
	print(str(np.argmax(predictions[i]))+" "+str(y_test[i]))

#Saving model with weights and optimizer
os.chdir('..\\Data\\model')
path = os.getcwd()
model_file = 'model.h5'
model.save(model_file)