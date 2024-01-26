import Hopfield
import audiobin
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
    PATTERNS to retrieve
"""

audio_object = audiobin.audio_importation()

num_patterns = 81  
I = np.arange(81)
np.random.shuffle(I)
I_new = I.copy()
patterns_bin_total_list = []

for iter in range(2, num_patterns+1):
    binary_list=[]
    patterns_bin = np.zeros((iter,len(audio_object[0]['stft_coeff'])))
    for i in range(0,iter):
        stft_coeff = audio_object[I_new[i]]['stft_coeff']
        binary = audiobin.audio_binarization(stft_coeff)
        patterns_bin[i,:] = binary
    
    patterns_bin_total_list.append(patterns_bin)


"""
    MAIN
"""

model = Hopfield.HopfieldNetwork()

T = T = np.linspace(0.01, 2, 80)
A = np.zeros(80)
for i in range(0,80):
    A[i] = (i+2)/513
magns = np.zeros((len(T),len(A)))

for iter in range(0,num_patterns-1):
    model.train_weights(patterns_bin_total_list[iter])
    print("We're using",iter+2, "patterns --- 1")
    rand_test = np.random.choice(range(model.num_patterns))
    randomness = 0.1
    test = patterns_bin[rand_test,:]
    test_corrupted = model.get_corrupted(test,randomness)

    #Predictions
    for t in tqdm(range(0,len(T))) :
        predicted,magnetization = model.predict(test_corrupted, temperature=T[t])
        magns[t,iter] = np.abs(magnetization[rand_test,-1])


plt.pcolormesh(A, T, magns, cmap='plasma')
plt.colorbar()
plt.grid(True, linestyle='dashed', linewidth=0.5)
plt.title('Phase Diagram with r = 0.1')
plt.tight_layout()
plt.savefig("/Users/silver22/Desktop/output_figura1.png")
plt.show()

