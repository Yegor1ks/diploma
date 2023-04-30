import matplotlib.pyplot as plt
from scipy.signal import hilbert
# from scipy.interpolate import interp1d
import numpy as np

A = 1000000
c = 1500
fs = 200000
fd = 5 * fs
d = c / (2 * fs)

z = -20 + np.random.uniform(-1.5, 1.5, 80)
x = np.linspace(0, 80, 8001)

zx = np.interp(x, np.arange(80), z)
# zx = interp1d(x, z)

plt.figure()
plt.grid(True)
plt.suptitle('zx')
plt.ylim((-25, 0))
plt.plot(x, zx)

T = np.sqrt(x[-1] ** 2 + 20 ** 2) / c
time = np.arange(0, T, 1 / fd)
signal = np.random.uniform(-1.5, 1.5, (2, len(time)))
r = np.sqrt(zx ** 2 + x ** 2)
ri = np.zeros(54975)

for i in np.arange(0, 54975, 1):
    ri = abs(r - (i + 1) / fd * c)
    m = np.min(ri)
    idx = np.argmin(ri)
    r_min = r[idx]
    alpha = np.arctan(zx[idx] / x[idx])
    tau = d / c * np.sin(alpha)
    if i / fd * c > 20:
        signal[0, i] = signal[0, i] + A / (r_min ** 2) * np.sin(2 * np.pi * fs * time[i])
        signal[1, i] = signal[1, i] + A / (r_min ** 2) * np.sin(2 * np.pi * fs * (time[i] - tau))

plt.figure()
plt.grid(True)
plt.suptitle('signal')
plt.plot(time, signal[0])

# plt.figure()
# plt.grid(True)
# plt.suptitle('ri')
# plt.plot(np.linspace(0, 80, 8001), ri)

s_left = hilbert(signal[0])
s_right = hilbert(signal[1])
dphi = np.angle(np.exp(1j * (np.angle(s_left) - np.angle(s_right))))

alpha = np.zeros(54975)
x_res = np.zeros(54975)
y_res = np.zeros(54975)

for i in np.arange(0, 54975, 1):
    alpha[i] = dphi[i] * c / (2 * np.pi * fs * d)
    x_res[i] = i / fd * c * np.cos(alpha[i])
    y_res[i] = i / fd * c * np.sin(alpha[i])

plt.figure()
plt.grid(True)
plt.suptitle('result')
plt.plot(x_res, y_res)
plt.plot(x, zx, color='r')

plt.show()
