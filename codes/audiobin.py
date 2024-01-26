#Standard libraries
import matplotlib.pyplot as plt
import numpy as np
#Sound libraries
import librosa
import librosa.display

def fft_signal(audio_path):
    audio, sr = librosa.load(audio_path)
    stft_signal = librosa.stft(audio, n_fft=1024, hop_length=512)
    stft_coeff = np.mean(stft_signal, axis=1)

    return audio, sr,  stft_signal, stft_coeff

def audio_importation():
    audio_objects = []

    for i in range(0,81):
        file_path = f'/Users/silver22/Documents/Audio_retrieval/My projects/registrazioni/{i}.wav'
        audio, sr, stft_signal, stft_coeff = fft_signal(file_path)
        audio_object = {'audio': audio, 'sr': sr, 'stft_signal': stft_signal, 'stft_coeff': stft_coeff}
        audio_objects.append(audio_object)

    return audio_objects


def audio_binarization(stft_coeff):
    binary = (stft_coeff > 0).astype(int)
    binary = 2*binary - 1

    return binary