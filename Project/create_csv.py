import os
import csv
from extract import extract_features


header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
	header += f' mfcc{i}'
header += ' label'
header = header.split() #Создание заголовка таблицы

os.chdir('..\\Data')
data_path = os.getcwd() #Получаем доступ к нужной директории

file = open(os.path.join(data_path, 'data', 'data.csv'), 'w', newline='') #Создание файла на запись данных
with file:
	writer = csv.writer(file)
	writer.writerow(header)
genres = 'classical pop rock'.split()
for g in genres:
	for filename in os.listdir(os.path.join(data_path, f'genres\\{g}')): #Циклом проходим по всем трекам всех жанров
		songname = os.path.join(data_path, f'genres\\{g}\\{filename}')
		to_append = f'{filename} '
		to_append += extract_features(songname) #Вычисляем свойства музыкального трека
		to_append += f' {g}'
		file = open(os.path.join(data_path, 'data', 'data.csv'), 'a', newline='')
		with file:
			writer = csv.writer(file)
			writer.writerow(to_append.split()) #Записываем строку в таблицу
	print(f"Done for {g}")

