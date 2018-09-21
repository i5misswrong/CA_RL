import numpy as np
import matplotlib.pyplot as plt

k1 = 1
k2 = 2
k3 = 3
k4 = 4
k5 = 5
k6 = 6
k7 = 0
k8 = 0
k9 = 0
k10 = 0
k11 = 0
k12 = 0
k13 = 0
k14 = 0
k15 = 0
k16 = 0
k17 = 0
k18 = 0
k19 = 0
k20=0
k21=0

T = 10

delta_1 = 1
delta_2 = 1
delta_3 = 2
delta_4 = 3
delta_5 = 4
delta_6 = 5
delta_7 = 0
delta_8 = 0
delta_9=0
delta_10=0


def cule():
    x = np.linspace(0, 10, 100)
    y_ = (k1 * get_I_1(x) + k2 * get_J_1(x) + delta_1 * T + k8 * get_I_2(x) + 2 * k9 * get_I_1(x) * get_J_1(x)
           + k10 * np.power(get_J_1(x), 2) + k11 * get_J_2(x) + delta_3 * np.power(T, 2) + delta_4 * get_I_1(x) * T
           + delta_5 * get_J_1(x) * T) + (2 * k3 + 2 * k12 * get_I_1(x) + 2 * k12 * get_I_1(x) + 2 * k13 * get_J_1(
        x) + 2 * delta_6 * T + k4 * get_I_1(x) + k5 * get_J_1(x) + delta_2 * T + k15 * np.power(get_I_1(x),
                                                                                                2) + k16 * get_I_2(
        x) + 2 * k17 * get_I_1(x) * get_J_1(x) + k18 * np.power(get_J_1(x), 2) + k19 * get_J_2(x) + delta_7 * np.power(
        T, 2) + delta_8 * get_I_1(x) * T +delta_9*get_J_1(x)*T)*x+(3*k14+k6+k20*get_I_1(x)+k21*get_J_1(x)+delta_10*T)*np.power(x,2)

    print(x)
    print('-----------')
    print(y_)
    plt.plot(x, y_)
    plt.show()


def get_I_1(x):
    return x - 2 + 2 / (np.sqrt(1 + x))


def get_I_2(x):
    return np.power(x, 2) + 2 * np.power(((1 / (np.sqrt(1 + x))) - 1), 2)


def get_I_3(x):
    return np.power(x, 3) + 2 * np.power(((1 / (np.sqrt(1 + x))) - 1), 3)


def get_J_1(x):
    return x


def get_J_2(x):
    return np.power(x, 2)


if __name__ == '__main__':
    cule()
