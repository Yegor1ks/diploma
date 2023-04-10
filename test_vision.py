from math import floor
from numpy import deg2rad as rad

import matplotlib.pyplot as plt
import numpy as np
import scipy

# настройка вывода трёхмерных графиков
fig3d = plt.axes(projection='3d')
xGrid, yGrid = np.meshgrid(np.arange(200), np.arange(200))
fig3d.set(zlim=(-25, 0))
fig3d.set_xlabel("X")
fig3d.set_ylabel("Y")

# симуляция дна #

# массив дна со случайными значениями, мат ожиданием -20 и дисперсией 1.5
bot = -20 + 20 * np.random.uniform(-1.5, 1.5, (200, 200)) * np.random.uniform(-1, 1, (200, 200))

# Процедура сглаживания полученного массива дна относительно глубины 20 м.
# Применяется фильтр-интегратор по строкам и столбцам, ориентирующийся на текущие
# и предыдущие индексы, для этого значения первых(нулевых) строки и столбца равны -20
k = 0.08  # коэффициент фильтра-интегратора

for i in range(1, 200):
    for j in range(0, 200):
        bot[j][0] = -20
        bot[j][i] = bot[j][i - 1] * (1 - k) + bot[j][i] * k

for i in range(1, 200):
    for j in range(0, 200):
        bot[0][j] = -20
        bot[i][j] = bot[i - 1][j] * (1 - k) + bot[i][j] * k

fig3d.set_title("Карта дна")
fig3d.plot_surface(xGrid, yGrid, bot)

# формирование сигнала #

phi1 = -45  # граничные углы обзора
phi2 = 45
phiN = 20  # количество рассматриваемых углов рыскания

H = 20  # глубина

fd = 4000  # частота дискретизации
fs = 700  # частота сигнала
length = floor(140 / 1500 * fd)  # 373 отсчётов на 20 * 7 = 140 метров

s1 = np.zeros((phiN, length))  # буфер для сигналов
s2 = np.zeros((phiN, length))
rev1 = np.random.uniform(-1, 1, (phiN, length))  # буфер для реверберационной помехи
rev2 = np.random.uniform(-1, 1, (phiN, length))
rr = np.arange(1, length + 1) / fd * 1500
reverb = 1 / (rr ** 2)
for i in range(phiN):
    rev1[i][:] = rev1[i][:] * reverb
    rev2[i][:] = rev2[i][:] * reverb

bot1 = bot
zz = np.zeros((3, 6400))

# карта линий, о которым идёт ХН
for i in range(phiN):
    phi = phi1 + (phi2 - phi1) / phiN * (i + 1)
    for j in range(length):
        r = j / fd * 1500  # дистанция
        if r > H:
            dt = j / fd
            # координаты точек дна от которых отражается сигнал
            x = dt * 1500 * np.sin(rad(phi)) + 100
            y = dt * 1500 * np.cos(rad(phi)) + 1
            z = bot[floor(x)][floor(y)]
            bot1[floor(x)][floor(y)] = -25
            # угол наклона дна относительно соседних точек
            a1 = np.arctan((bot[floor(x)][floor(y + 1)] - bot[floor(x)][floor(y)]) / 1)
            # угол от антенны до точки на дне
            a2 = np.arctan(bot[floor(x)][floor(y + 1)] / y)
            # обратное отражение сигнала от дна
            A = np.sin(a1 - a2) ** 2
            # эхосигнал (с реверберацией) на двух элементах ПА
            tau = 0
            s1[i][j] = rev1[i][j] + A * 1000 / (rr[j] ** 2) * np.sin(2 * np.pi * fs * dt)
            s2[i][j] = rev2[i][j] + A * 1000 / (rr[j] ** 2) * np.sin(2 * np.pi * fs * (dt - tau))

fig3d.plot_surface(xGrid, yGrid, bot1, cmap='cividis')

ax2d, fig2d = plt.subplots()
fig2d.set_title("Приходящие сигналы")
fig2d.grid(True)
fig2d.set_xlabel("t")
fig2d.set_ylabel("S(t)")
for i in range(20):
    # fig2d.plot(np.arange(length), np.matrix(rev1[0]).T)
    fig2d.plot(np.arange(length), np.matrix(s1[i]).T)

# приём сигнала #

Fs1 = np.zeros((phiN, length), dtype=complex)  # буфер для спектра
Fs2 = np.zeros((phiN, length), dtype=complex)
for i in range(phiN):
    # Частотный спектр сигнала, БПФ
    Fs1[i] = scipy.fft.fft(s1[i])
    Fs2[i] = scipy.fft.fft(s2[i])
    # прямой спектральный анализ
    Fs1[i] = np.sqrt(Fs1[i].real ** 2 + Fs1[i].imag ** 2)
    Fs2[i] = np.sqrt(Fs2[i].real ** 2 + Fs2[i].imag ** 2)

    # пока что непонятно зачем
    # phi = phi1 + (phi2 - phi1) / phiN * i
    # for j in range(0, length, 100):
    #     r = j / fd * 1500  # дистанция
    #     t = j / fd
    #     x = t * 1500 * np.sin(rad(phi)) + 100
    #     y = t * 1500 * np.cos(rad(phi)) + 1
    #     z = bot[floor(x)][floor(y)]

plt.figure()
# FFs1 = np.zeros(length)
# FFs1 = np.matrix(Fs1[0][0:fd/2])
# plt.plot(np.arange(fd/2), FFs1)
plt.grid(True)
plt.plot(np.arange(0, fd, fd / length), np.real(Fs1[0] / max(Fs1[0])).T)
plt.suptitle('FFT (нормализованное)')

plt.show()

x = zz[:][0]
y = zz[:][1]
z = zz[:][2]

# построение картины дна по полученному сигналу #

xv = np.arange(min(x), max(x), 80)
yv = np.arange(min(y), max(y), 80)
X, Y = np.meshgrid(xv, yv)

# Z = scipy.interpolate.griddata(x.ravel(), y.ravel(), z.ravel(), X, Y)

# figure4 = plt.figure()
# ax = figure4.add_subplot(projection='3d')
# fig3d.plot_wireframe(X, Y, Z)
plt.show()
