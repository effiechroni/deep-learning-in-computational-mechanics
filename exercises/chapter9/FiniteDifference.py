import numpy as np
from numba import jit

"""
Finite Difference Solver for the 1D wave equation
Adapted from: http://hplgit.github.io/INF5620/doc/notes/wave-sphinx/main_wave.html
"""


@jit(nopython=True)
def finiteDifferenceStep1D(u0, u1, f, c, dx, Nx, dt):
    u2 = np.zeros((Nx + 3))
    # central difference scheme with harmonic mean 
    u2[1:-1] = -u0[1:-1] + 2 * u1[1:-1] + \
               (dt / dx) ** 2 * ((0.5 / c[1:-1] ** 2 + 0.5 / c[2:] ** 2) ** (-1) * (u1[2:] - u1[1:-1]) - \
                                 (0.5 / c[:-2] ** 2 + 0.5 / c[1:-1] ** 2) ** (-1) * (u1[1:-1] - u1[:-2])) + \
               dt ** 2 * f[:]

    u2[0] = u2[2]
    u2[-1] = u2[-3]

    return u2


@jit(nopython=True)
def finiteDifference1D(u0, u1, f, c, dx, Nx, dt, N):
    Cx = np.max(c) * dt / dx
    if Cx ** 2 > 1:
        print('Warning: Courant number is larger than 1')

    U = np.zeros((Nx + 3, N + 1))
    U[:, 0] = u1

    for timestep in range(N):
        u2 = finiteDifferenceStep1D(u0, u1, f[:, timestep], c, dx, Nx, dt)
        u0[:], u1[:] = u1, u2

        U[:, timestep + 1] = u2

    return U


def getDerivativeInX(u, dx):
    dudx = (u[2:, :] - u[:-2, :]) / 2. / dx
    return dudx


def getAdjointSensitivity(u0, u1, c, dx, Nx, dt, N, u, um, sensorPositions):
    fadjoint = np.zeros((Nx + 1, N))
    fadjoint[sensorPositions, :] = um[sensorPositions, 1:] - u[1:-1][sensorPositions, 1:]
    fadjoint = np.flip(fadjoint, axis=1)

    cost = 0.5 * np.sum(dx * np.trapz(fadjoint[sensorPositions, :] ** 2, dx=dt, axis=1))

    uadjoint = finiteDifference1D(u0.copy(), u1.copy(), fadjoint, c, dx, Nx, dt, N)
    uadjoint = np.flip(uadjoint, axis=1)

    integrand = 2 * np.expand_dims(c[1:-1], 1) ** 2 * getDerivativeInX(uadjoint, dx) * getDerivativeInX(u, dx)
    gradient = 2 * np.trapz(integrand, dx=dt, axis=1) * dx

    return gradient, cost


def getAllAdjointSensitivities(u0, u1, f, c, dx, Nx, dt, N, um, sensorPositions):
    gradient = np.zeros(Nx + 1)
    cost = 0

    for i in range(len(f)):  # numberOfSources
        u = finiteDifference1D(u0.copy(), u1.copy(), f[i], c, dx, Nx, dt, N)

        result = getAdjointSensitivity(u0, u1, c, dx, Nx, dt, N, u, um[i], sensorPositions)
        gradient += result[0]
        cost += result[1]

    gradient /= len(f)
    cost /= len(f)

    return gradient, cost
