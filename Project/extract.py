import numpy as np
import librosa

def extract_features(songname):
	"""Возвращение ствойств музыкального трека"""
	y, sr = librosa.load(songname, mono=True, duration=30) #Загрузка музыкального трека
	chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) #Вычисление хроматограммы
	rms = librosa.feature.rms(y=y) #Вычисление среднеквадратичного значения каждого кадра
	spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr) #Вычисление спектральных центроидов
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr) #Вычисление ширины спектральной полосы
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr) #Вычисление частоты спада
	zcr = librosa.feature.zero_crossing_rate(y) #Вычисление частоты пересечения нуля
	mfcc = librosa.feature.mfcc(y=y, sr=sr) #Вычисление мел-кепстральных коэффициентов
	#Запись полученных значений в строку
	to_append = f'{np.mean(chroma_stft)} \
{np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} \
{np.mean(rolloff)} {np.mean(zcr)}'
	for elem in mfcc:
		to_append += f' {np.mean(elem)}'
	return to_append