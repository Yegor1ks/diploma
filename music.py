import numpy as np
import matplotlib.pyplot as plt

M = 10  # количество элементов решетки (сенсоров)
SNR = 10  # Отношение сигнал-шум (dB)
d = 3  # количество источников ЭМ волн
N = 50  # количество "замеров" (snapshots)

S = (np.sign(np.random.randn(d, N)) + 1j * np.sign(np.random.randn(d, N))) / np.sqrt(2)  # QPSK

W = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2) * 10 ** (-SNR / 20)  # AWGN
# Общая формула:
# sqrt(N0/2)*(G1 + jG2),
# где G1 и G2 - независимые гауссовские процессы.
# т.к. Es(энергия символа) для QPSK равна 1 Вт, спектральная мощность шума (noise spectral density):
# N0 = (Es/N)^(-1) = SNR^(-1) [Вт] (принимаем в данном примере, что SNR = Es/N0);
# или в логарифмическом масштабе:
# SNR_dB = 10log10(SNR) => N0_dB = -10log10(SNR) = -SNR_dB [дБ];
# Нам дано значение SNR в логарифмической шкале (т.е. в дБ), переводим в линейную:
# SNR = 10^(SNR_dB/10) => sqrt(N0) = (10^(-SNR_dB/10))^(1/2) = 10^(-SNR_dB/20)

mu_R = 2 * np.pi / M  # ширина главного лепестка ХН

cases = [[-1., 0, 1.], [-0.5, 0, 0.5], [-0.3, 0, 0.3]]
for idxm, c in enumerate(cases):
    # углы прихода (пространственные частоты):
    mu_1 = c[0] * mu_R
    mu_2 = c[1] * mu_R
    mu_3 = c[2] * mu_R

    # сканирующие вектора
    a_1 = np.exp(1j * mu_1 * np.arange(M))
    a_2 = np.exp(1j * mu_2 * np.arange(M))
    a_3 = np.exp(1j * mu_3 * np.arange(M))

    A = np.array([a_1, a_2, a_3]).T  # матрица сканирующих векторов
    X = np.dot(A, S) + W  # матрица принятых сигналов

    R = np.dot(X, np.matrix(X).H)

    U, Sigma, Vh = np.linalg.svd(X, full_matrices=True)
    U_0 = U[:, d:]  # шумовое подпространство

    thetas = np.arange(-90, 91) * (np.pi / 180)  # диапазон азимутов
    mus = np.pi * np.sin(thetas)  # диапазон пространственных частот
    a = np.empty((M, len(thetas)), dtype=complex)

    for idx, mu in enumerate(mus):
        a[:, idx] = np.exp(1j * mu * np.arange(M))

    # MVDR:
    S_MVDR = np.empty(len(thetas), dtype=complex)
    for idx in range(np.shape(a)[1]):
        a_idx = (a[:, idx]).reshape((M, 1))
        S_MVDR[idx] = 1 / (np.dot(np.matrix(a_idx).H, np.dot(np.linalg.pinv(R), a_idx)))

    # MUSIC:
    S_MUSIC = np.empty(len(thetas), dtype=complex)
    for idx in range(np.shape(a)[1]):
        a_idx = (a[:, idx]).reshape((M, 1))
        S_MUSIC[idx] = np.dot(np.matrix(a_idx).H, a_idx) \
                       / (np.dot(np.matrix(a_idx).H, np.dot(U_0, np.dot(np.matrix(U_0).H, a_idx))))

    plt.subplots(figsize=(8, 5), dpi=150)
    plt.semilogy(thetas * (180 / np.pi), np.real((S_MVDR / max(S_MVDR))), color='green', label='MVDR')
    plt.semilogy(thetas * (180 / np.pi), np.real((S_MUSIC / max(S_MUSIC))), color='red', label='MUSIC')
    plt.grid(color='r', linestyle='-', linewidth=0.2)
    plt.xlabel('Azimuth angles θ (degrees)')
    plt.ylabel('Power (pseudo)spectrum (normalized)')
    plt.legend()
    plt.title('Case #' + str(idxm + 1))
    plt.show()
