import numpy as np
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
import audiobin
import Hopfield


"""
    PATTERNS to retrieve
"""

audio_object = audiobin.audio_importation()

num_patterns = 30 
I = np.arange(81)
np.random.shuffle(I)
I_new = I.copy()
patterns_bin = np.zeros((num_patterns,len(audio_object[0]['stft_coeff'])))

print("The set of stored numbers is given by")
print(I_new[0:30])
print(" ")

for i in range(0,num_patterns):
        stft_coeff = audio_object[I_new[i]]['stft_coeff']
        binary = audiobin.audio_binarization(stft_coeff)
        patterns_bin[i,:] = binary
    

"""
    MAIN
"""

model = Hopfield.HopfieldNetwork()
model.train_weights(patterns_bin)

print(" ")
print("Start to create corrupted file-test...")
rand_test = np.random.choice(range(30))

#critical randomness > 0.4
randomness = 0.3

print("We randomly choose to corrupt pattern = number",I_new[rand_test], "with randomness =", randomness)
test = patterns_bin[rand_test,:]

#Corruption
test_corrupted = model.get_corrupted(test,randomness)

#Prediction
predicted,magnetization = model.predict(test_corrupted,temperature=0.15)

result = model.pattern_similarity(patterns_bin,predicted)
result = I_new[result]
print(" ")
print("The corrupted pattern matches the number", result, "while the correct result is",
      I_new[rand_test])
print(" ")


""" REPRODUCING AUDIOS """
print("Reproducing original - corrupted - predicted audio")
print(" ")
#Reproducing original audio
stft_signal = audio_object[I_new[rand_test]]['stft_signal']
signal = librosa.istft(stft_signal, hop_length=512)
sd.play(signal, audio_object[I_new[rand_test]]['sr'])
sd.wait()

#Reproducing corrupted audio
signal_corrupted = np.zeros_like(stft_signal)
for i in range(signal_corrupted.shape[0]):
    if (test[i] != test_corrupted[i]):
        signal_corrupted[i,:] = (stft_signal[i,:]*(-1))**2
    else: signal_corrupted[i,:] = stft_signal[i,:]
audio_corrupted = librosa.istft(signal_corrupted, hop_length=512)
sd.play(audio_corrupted, audio_object[I_new[rand_test]]['sr'])
sd.wait()

#Reproducing retrieved audio
signal_predicted = np.zeros_like(stft_signal)
for i in range(signal_predicted.shape[0]):
    if (test[i] != predicted[i]):
        signal_predicted[i,:] = (stft_signal[i,:]*(-1))**2
    else: signal_predicted[i,:] = stft_signal[i,:]
audio_predicted = librosa.istft(signal_predicted, hop_length=512)
sd.play(audio_predicted, audio_object[I_new[rand_test]]['sr'])
sd.wait()

"""Plot magnetization """
fig, ax = plt.subplots(1,1,figsize=(10,7))
x = np.arange(0,len(magnetization[rand_test,:])+1)
ax.plot(x,np.insert(magnetization[rand_test,:], 0, 0),label="Mattis magnetization")
ax.set_xlabel("MC step")
plt.title("Mattis magnetization wrt test-pattern")
plt.tight_layout()
plt.show()



