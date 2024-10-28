import numpy as np
import scipy
import time
import matplotlib.pyplot as plt


def shapeFunctions(xi, p):  # up to order 6
    N = np.zeros(p + 1)
    N[0] = 0.5 * (1 - xi)
    N[-1] = 0.5 * (1 + xi)  # should always be the last entry
    if p > 1:
        N[1] = 0.25 * np.sqrt(6) * (xi ** 2 - 1)
    if p > 2:
        N[2] = 0.25 * np.sqrt(10) * (xi ** 2 - 1) * xi
    if p > 3:
        N[3] = 0.0625 * np.sqrt(14) * (5 * xi ** 4 - 6 * xi ** 2 + 1)
    if p > 4:
        N[4] = 0.1875 * np.sqrt(2) * xi * (7 * xi ** 4 - 10 * xi ** 2 + 3)
    if p > 5:
        N[5] = 0.03125 * np.sqrt(22) * (21 * xi ** 6 - 35 * xi ** 4 + 15 * xi ** 2 - 1)
    return N


def shapeFunctionDerivatives(xi, p):
    dN = np.zeros(p + 1)
    dN[0] = -0.5
    dN[-1] = 0.5
    if p > 1:
        dN[1] = 0.25 * np.sqrt(6) * (2 * xi)
    if p > 2:
        dN[2] = 0.25 * np.sqrt(10) * (3 * xi ** 2 - 1)
    if p > 3:
        dN[3] = 0.0625 * np.sqrt(14) * (20 * xi ** 3 - 12 * xi)
    if p > 4:
        dN[4] = 0.1875 * np.sqrt(2) * (35 * xi ** 4 - 30 * xi ** 2 + 3)
    if p > 5:
        dN[5] = 0.03125 * np.sqrt(22) * (126 * xi ** 5 - 140 * xi ** 3 + 30 * xi)
    return dN


def strainDisplacementMatrix(xi, s, p):
    dN = shapeFunctionDerivatives(xi, p) * 2.0 / s
    B = dN
    return B


def localStiffnessMatrix(elementIndex, EA, s, p, integrationOrder):
    K = np.zeros((p + 1, p + 1))
    gp, gw = np.polynomial.legendre.leggauss(integrationOrder)

    for i in range(len(gp)):
        xi = gp[i]
        B = np.expand_dims(
            strainDisplacementMatrix(xi, s, p), 0
        )
        x = 0.5 * (xi + 1) * s + s * elementIndex
        K += np.transpose(B) @ B * gw[i] * EA(x)

    K *= s / 2.0  # determinant (Jacobian)
    return K


def eft(i, p):
    return np.linspace(i * p, (i + 1) * p, p + 1, dtype=int).tolist()


def globalStiffnessMatrix(EA, L, n, p, integrationOrder):
    K = scipy.sparse.dok_matrix((n * p + 1, n * p + 1))
    for i in range(n):
        Ke = localStiffnessMatrix(
            i, EA, L / n, p, integrationOrder
        )
        index = eft(i, p)
        K[np.ix_(index, index)] += Ke
    return K


def localForceVectorFromDistLoad(elementIndex, distLoad, s, p, integrationOrder):
    F = np.zeros(p + 1)
    gp, gw = np.polynomial.legendre.leggauss(integrationOrder)
    for i in range(len(gp)):
        xi = gp[i]
        N = shapeFunctions(xi, p)
        x = 0.5 * (xi + 1) * s + s * elementIndex
        F += N * distLoad(x) * gw[i]
    F *= s / 2.0 
    return F


def globalForceVectorFromDistLoad(distLoad, L, n, p, integrationOrder):
    F = np.zeros(n * p + 1)
    for i in range(n):
        Fe = localForceVectorFromDistLoad(i, distLoad, L / n, p, integrationOrder)
        index = eft(i, p)
        F[index] += Fe
    return F


def applyDirichletBCs(K, F, BCs):
    F -= K[:, BCs[0]] @ BCs[1]
    F[BCs[0]] = BCs[1]
    K[:, BCs[0]] = 0
    K[BCs[0], :] = 0
    K[BCs[0], BCs[0]] = 1
    return K, F


def getDisplacements(U, L, n, p, pointsPerElement, sampling="uniform"):
    s = L / n
    U_ = np.zeros(pointsPerElement * n)
    grid = np.zeros(pointsPerElement * n)
    if sampling == "uniform":
        points = np.linspace(-1, 1, pointsPerElement)
    elif sampling == "Gauss":
        points, _ = np.polynomial.legendre.leggauss(pointsPerElement)
    N = []
    for i in range(pointsPerElement):
        N.append(shapeFunctions(points[i], p))
    for i in range(n):
        Ue = U[eft(i, p)]
        for ixi in range(pointsPerElement):
            U_[i * pointsPerElement + ixi] = np.sum(N[ixi] * Ue)
            grid[i * pointsPerElement + ixi] = 0.5 * (points[ixi] + 1) * s + s * i
    return grid, U_


def getStrains(U, L, n, p, pointsPerElement, sampling="uniform"):
    s = L / n
    strain = np.zeros(pointsPerElement * n)
    grid = np.zeros(pointsPerElement * n)
    if sampling == "uniform":
        points = np.linspace(-1, 1, pointsPerElement)
    elif sampling == "Gauss":
        points, _ = np.polynomial.legendre.leggauss(pointsPerElement)
    points = np.linspace(-1, 1, pointsPerElement)
    dN = []
    for i in range(pointsPerElement):
        dN.append(shapeFunctionDerivatives(points[i], p))
    for i in range(n):
        Ue = U[eft(i, p)]
        for ixi in range(pointsPerElement):
            strain[i * pointsPerElement + ixi] = np.sum(dN[ixi] * Ue) / (s / 2.0)
            grid[i * pointsPerElement + ixi] = 0.5 * (points[ixi] + 1) * s + s * i
    return grid, strain


def getPotentialEnergy(U, L, n, p, pointsPerElement):
    integrationGrid, displacement = getDisplacements(
        U, L, n, p, pointsPerElement, sampling="Gauss"
    )
    integrationGrid, strain = getStrains(U, L, n, p, pointsPerElement, sampling="Gauss")
    _, gw = np.polynomial.legendre.leggauss(pointsPerElement)
    gw = np.tile(gw, n)

    s = L / n
    Jacobian = s / 2.0
    internalEnergy = 0.5 * np.sum(EA(integrationGrid) * strain ** 2 * gw) * Jacobian
    externalEnergy = np.sum(distLoad(integrationGrid) * displacement * gw) * Jacobian
    return internalEnergy, externalEnergy


# FE driver

DirichletBCs = np.array(
    [[0, -1], [0, 1]]
)  # dof indices  in first list, values in second
EA = lambda x: x ** 2 + 1.0
L = 1.5
distLoad = lambda x: -6 * x * np.pi * np.sin(3 * np.pi * x) - 9 * (
        x ** 2 + 1
) * np.pi ** 2 * np.cos(3 * np.pi * x)
uAnalytic = lambda x: (1.0 - np.cos(3.0 * np.pi * x))
analyticPotentialEnergy = 0.5 * 116.959701987868 - 86.329173615

n = 30
p = 1
integrationOrder = int(np.ceil(0.5 * (p - 1) ** 2 + 1))

start = time.perf_counter()
K = globalStiffnessMatrix(EA, L, n, p, integrationOrder)
F = globalForceVectorFromDistLoad(distLoad, L, n, p, integrationOrder)
K, F = applyDirichletBCs(K, F, DirichletBCs)

U = scipy.sparse.linalg.spsolve(K.tocsr(), F, use_umfpack=True)
end = time.perf_counter()
print("Elapsed Time: {:.2e} s".format(end - start))

pointsPerElement = 5
postprocessingGrid, postprocessingU = getDisplacements(U, L, n, p, pointsPerElement)

fig, ax = plt.subplots()
ax.plot(postprocessingGrid, postprocessingU, "k")
ax.plot(
    postprocessingGrid[::pointsPerElement], postprocessingU[::pointsPerElement], "ko"
)
ax.plot(postprocessingGrid[-1], postprocessingU[-1], "ko")
ax.plot(postprocessingGrid, uAnalytic(postprocessingGrid), "r:")
ax.grid()
plt.show()

# potential energy
internalEnergy, externalEnergy = getPotentialEnergy(U, L, n, p, pointsPerElement)
potentialEnergy = internalEnergy - externalEnergy
relativeError = (
        np.abs((potentialEnergy - analyticPotentialEnergy) / analyticPotentialEnergy) * 100
)
string = "Final relative error of the potential energy: {:.2f} %"
print(string.format(relativeError))

# post-processing
fig, ax = plt.subplots()
ax.set_xlabel("$x$")
ax.set_ylabel("$u$")
ax.plot(postprocessingGrid, postprocessingU, "k", linewidth=3, label="FE prediction")
ax.plot(
    postprocessingGrid[::pointsPerElement], postprocessingU[::pointsPerElement], "ko"
)
ax.plot(postprocessingGrid[-1], postprocessingU[-1], "ko")
ax.plot(
    postprocessingGrid,
    uAnalytic(postprocessingGrid),
    "r:",
    linewidth=3,
    label="analytical solution",
)
ax.grid()
plt.legend(loc="upper right")
fig.tight_layout()
plt.show()