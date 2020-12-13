# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:55:28 2020

@author: Peilin Yang
"""

import utils
import numpy as np
from numba import njit

bmax=50 
amax=4000
kmax=1
nZ=3
nB=50 
rho_z=0.966
sigma_z=0.92
nA=70
nK=50

b_grid = utils.agrid(amax=bmax, n=nB)
a_grid = utils.agrid(amax=amax, n=nA)
k_grid = utils.agrid(amax=kmax, n=nK)
e_grid, pi, Pi = utils.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

tax=0.5
eis=2
w=0.6
beta=0.99
@njit
def step3(lhs, rhs, Wb, a_grid, eis, nZ, nB, nA):
    ap_endo = np.empty((nZ, nB, nA))
    Wb_endo = np.empty((nZ, nB, nA))
    for iz in range(nZ):
        for ibp in range(nB):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ibp, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ibp, ia] = 0
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, 0]
                elif iap == nA:
                    ap_endo[iz, ibp, ia] = a_grid[iap]
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap]
                else:
                    y0 = lhs[iz, ibp, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ibp, iap] - rhs[iap, ia]
                    ap_endo[iz, ibp, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap - 1] + (
                                ap_endo[iz, ibp, ia] - a_grid[iap - 1]) * (
                                Wb[iz, ibp, iap] - Wb[iz, ibp, iap - 1]) / (a_grid[iap] - a_grid[iap - 1])
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo

def step4(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, b', a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + bbb - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) > 0, 'b(bp) is not increasing'
    # assert np.min(np.diff(ap_endo, axis=1)) > 0, 'ap(bp) is not increasing'
    i, pi = utils.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    ap = utils.apply_coord(i, pi, ap_endo.swapaxes(1, 2)).swapaxes(1, 2)
    bp = utils.apply_coord(i, pi, b_grid).swapaxes(1, 2)
    return bp, ap

def step5(lhs, rhs, Wb, a_grid, k_grid, eis, nZ, nK, nA):
    ap_endo = np.empty((nZ, nK, nA))
    Wb_endo = np.empty((nZ, nK, nA))
    for iz in range(nZ):
        for ik in range(nK):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ik, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ik, ia] = 0
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, 0]
                elif iap == nA:
                    ap_endo[iz, ik, ia] = a_grid[iap]
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, iap]
                else:
                    y0 = lhs[iz, ik, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ik, iap] - rhs[iap, ia]
                    ap_endo[iz, ik, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * (
                            Wb[iz, 0, iap - 1] + (ap_endo[iz, ik, ia] - a_grid[iap - 1]) *
                            (Wb[iz, 0, iap] - Wb[iz, 0, iap - 1]) / (a_grid[iap] - a_grid[iap - 1]))
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo

def step6(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, k, a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + b_grid[0] - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) < 0, 'b(kappa) is not decreasing'
    # assert np.min(np.diff(ap_endo, axis=1)) < 0, 'ap(kappa) is not decreasing'
    ap = utils.interpolate_y(b_endo[:, ::-1, :].swapaxes(1, 2), b_grid, 
                             ap_endo[:, ::-1, :].swapaxes(1, 2)).swapaxes(1, 2)
    return ap

def Psi1_fun(ap, a, ra, chi0, chi1, chi2):
    return np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)
def Psi2_fun(ap, a, ra, chi0, chi1, chi2):
    Psi1 = np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)
    return -(1 + ra) * (Psi1 + chi1 * (chi2 - 1) / chi2 * (np.abs(ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** chi2)
def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid
z_grid = income(e_grid, tax, w, 1)
Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))


# step 2: Wb(z, b', a') and Wa(z, b', a')
Wb = (Vb.T @ (beta * Pi.T)).T
Wa = (Va.T @ (beta * Pi.T)).T

# step 3: a'(z, b', a) for UNCONSTRAINED
lhs_unc = Wa / Wb
Psi1 = Psi1_fun(a_grid[:, np.newaxis], a_grid[np.newaxis, :], ra, chi0, chi1, chi2)

# Get the Polciy function a(a') if unconstraint opt
a_endo_unc, c_endo_unc = step3(lhs_unc, 1 + Psi1, Wb, a_grid, eis, nZ, nB, nA)

# step 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED
b_unc, a_unc = step4(a_endo_unc, c_endo_unc, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

# step 5: a'(z, kappa, a) for CONSTRAINED
lhs_con = lhs_unc[:, 0, :]
lhs_con = lhs_con[:, np.newaxis, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
a_endo_con, c_endo_con = step5(lhs_con, 1 + Psi1, Wb, a_grid, k_grid, eis, nZ, nK, nA)

# step 6: a'(z, b, a) for CONSTRAINED
a_con = step6(a_endo_con, c_endo_con, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)


# step 7a: put policy functions together
# Policy Function: a and b
a, b = a_unc.copy(), b_unc.copy()
b[b <= b_grid[0]] = b_grid[0]
a[b <= b_grid[0]] = a_con[b <= b_grid[0]]

zzz = z_grid[:, np.newaxis, np.newaxis]
bbb = b_grid[np.newaxis, :, np.newaxis]
aaa = a_grid[np.newaxis, np.newaxis, :]
c = zzz + (1 + ra) * aaa + (1 + rb) * bbb - Psi_fun(a, aaa, ra, chi0, chi1, chi2) - a - b
uc = c ** (-1 / eis)
u = e_grid[:, np.newaxis, np.newaxis] * uc

# step 7b: update guesses
Psi2 = Psi2_fun(a, aaa, ra, chi0, chi1, chi2)
Va = (1 + ra - Psi2) * uc
Vb = (1 + rb) * uc
