import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
import os

n = 500
Fs = 1000
F_max = 3

random_signal = np.random.normal(0, 10, n)

time = np.arange(n) / Fs

w = F_max / (Fs / 2)
sos = signal.butter(3, w, 'low', output='sos')

filtered_signal = signal.sosfiltfilt(sos, random_signal)

base_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

def plot_graph(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    save_path = os.path.join(figures_dir, filename + ".png")
    fig.savefig(save_path, dpi=600)
    plt.show()
    plt.close(fig)

plot_graph(time, filtered_signal,
           "Час (секунди)", "Амплітуда сигналу",
           f"Сигнал з максимальною частотою F_max = {F_max} Гц",
           "signal")

spectrum = fft.fft(filtered_signal)
spectrum_shifted = np.abs(fft.fftshift(spectrum))
freqs = fft.fftfreq(n, 1/Fs)
freqs_shifted = fft.fftshift(freqs)

plot_graph(freqs_shifted, spectrum_shifted,
           "Частота (Гц)", "Амплітуда спектру",
           f"Спектр сигналу з максимальною частотою F_max = {F_max} Гц",
           "spectrum")
