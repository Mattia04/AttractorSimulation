import numpy as np


def __reshape(pos):
    if pos.ndim == 1:
        return pos
    return pos.reshape(3, -1)


def __returns(dxdt, dydt, dzdt, dims):
    if dims == 1:
        return dxdt, dydt, dzdt

    return np.vstack([dxdt, dydt, dzdt]).flatten()


def lorentz(t, pos, rho=28, sigma=10, beta=8 / 3):
    x, y, z = __reshape(pos)

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return __returns(dxdt, dydt, dzdt, pos.ndim)


def thomas(t, pos, b=0.19):
    x, y, z = __reshape(pos)
    dxdt = np.sin(y) - b * x
    dydt = np.sin(z) - b * y
    dzdt = np.sin(x) - b * z
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def langford(t, pos, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = __reshape(pos)
    dxdt = (z - b) * x - d * y
    dydt = (z - b) * y + d * x
    dzdt = c + a * z - z**3 / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def dadras(t, pos, a=3, b=2.7, c=1.7, d=2, e=9):
    x, y, z = __reshape(pos)
    dxdt = y - a * x + b * y * z
    dydt = c * y + z * (1 - x)
    dzdt = d * x * y - e * z
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def chen_lee(t, pos, alpha=5, beta=-10, delta=-0.38):
    x, y, z = __reshape(pos)
    dxdt = alpha * x - y * z
    dydt = beta * y + x * z
    dzdt = delta * z + (x * y) / 3
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def lorentz83(t, pos, a=0.95, b=7.91, f=4.83, g=4.66):
    x, y, z = __reshape(pos)
    dxdt = -a * x - y**2 - z**2 + a * f
    dydt = -y + x * y - b * z * x + g
    dzdt = -z + b * x * y + x * z
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def rossler(t, pos, a=0.2, b=0.2, c=5.7):
    x, y, z = __reshape(pos)
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def halvorsen(t, pos, a=1.89):
    x, y, z = __reshape(pos)
    dxdt = -a * x - 4 * y - 4 * z - y**2
    dydt = -a * y - 4 * z - 4 * x - z**2
    dzdt = -a * z - 4 * x - 4 * y - x**2
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def rabinovich_fabrikant(t, pos, alpha=0.14, gamma=0.10):
    x, y, z = __reshape(pos)
    dxdt = y * (z - 1 + x**2) + gamma * x
    dydt = x * (3 * z + 1 - x**2) + gamma * y
    dzdt = -2 * z * (alpha + x * y)
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def three_scroll(t, pos, a=32.48, b=45.84, c=1.18, d=0.13, e=0.57, f=14.7):
    x, y, z = __reshape(pos)
    dxdt = a * (y - x) + d * x * z
    dydt = b * x - x * z + f * y
    dzdt = c * z + y * x - e * x**2
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def sprott(t, pos, a=2.07, b=1.79):
    x, y, z = __reshape(pos)
    dxdt = y + x * (a * y + z)
    dydt = 1 - b * x**2 + y * z
    dzdt = x - x**2 - y**2
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def sprott_linz(t, pos, a=0.5):
    x, y, z = __reshape(pos)
    dxdt = y + z
    dydt = -x + a * y
    dzdt = x**2 - z
    return __returns(dxdt, dydt, dzdt, pos.ndim)


def four_wing(t, pos, a=0.2, b=0.01, c=-0.4):
    x, y, z = __reshape(pos)
    dxdt = a * x + y * z
    dydt = b * x + c * y - x * z
    dzdt = -z - x * y
    return __returns(dxdt, dydt, dzdt, pos.ndim)
