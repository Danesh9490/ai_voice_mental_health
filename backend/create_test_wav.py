import numpy as np
import soundfile as sf

# Create 3-second sine wave tone
sr = 22050
t = np.linspace(0, 3, sr * 3)
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

sf.write("test.wav", audio, sr)

print("test.wav created successfully")
