import numpy as np
import matplotlib.pyplot as plt

# 1. Read txt file
data = np.loadtxt("OFDM.txt")  
I = data[:, 0]
Q = data[:, 1]
complex_signal = I + 1j * Q

# 2. Draw the time domain waveform (Real and Imaginary)
plt.figure()
plt.plot(I, label='In-phase (I)')
plt.plot(Q, label='Quadrature (Q)')
plt.title('Time Domain Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# 3. IQ Constellation
plt.figure()
plt.scatter(I, Q, color='blue', s=10)
plt.title('IQ Constellation Diagram')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.axis('equal')
plt.show()

# 4. Frequency domain analysis (spectrum)
fft_data = np.fft.fft(complex_signal)
freq = np.fft.fftfreq(len(fft_data))
plt.figure()
plt.plot(freq, 20 * np.log10(np.abs(fft_data)))
plt.title('Frequency Domain (Spectrum)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()

# 5. plot the frequency graph (Spectrogram)
from scipy.signal import spectrogram

f, t, Sxx = spectrogram(complex_signal, fs=64e6, nperseg=256, noverlap=128)

plt.figure()
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram (Time-Frequency Representation)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.tight_layout()
plt.show()
