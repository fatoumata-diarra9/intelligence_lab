import numpy as np
import soundfile as sf
sr = 16000
duree = 1.0
t = np.linspace(0, duree, int(sr * duree), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * 440 * t)

print("nombre d'echantillons :", len(signal))
print("premieres valeurs :", signal[:5])

sf.write("data/test_la.wav", signal, sr)
