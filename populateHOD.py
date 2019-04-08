"""
Populate a halo catalogue with galaxies using HOD parameters.

For now I will only use the simple 5 parameter HOD.
"""

import scipy.special
import scipy.optimize
import numpy as np
# import time


def populate(halo_mass, hod_par):

    """
    halo_mass is a numpy array with halo masses.
    hod_par is a list of HOD parameters in the following order:
    (log10(Mcut), log10(M1), sigma, kappa, alpha)

    Returns two arrays one for centrals one for satellites.
    The centrals array has 1 (there is a central) or 0 (no central).
    The satellites array has the number of satellites.
    """

    Mcut, M1, sigma, kappa, alpha = hod_par
    # Central galaxies
    NcenProb: float = 0.5*scipy.special.erfc((Mcut - np.log10(halo_mass)) / np.sqrt(2) / sigma)
    Ncen = np.random.poisson(NcenProb)
    # Can only have one central galaxy. Turn Ncen > 1 to 1
    Ncen[np.where(Ncen > 0)] = 1

    # Satellite galaxies
    NsatProb = NcenProb * ((halo_mass - kappa * 10 ** Mcut) / 10 ** M1) ** alpha
    NsatProb[np.where(np.isnan(NsatProb))] = 0
    NsatProb[np.where(NsatProb < 1)] = 0
    Nsat = np.random.poisson(NsatProb)

    return Ncen, Nsat


def place_satellites(halos, n_sat):
    """
    halos - halo mass, virial radius, scale radius, x, y, z
    Nsat - Number of satellites I need to place
    Both of the above are arrays.

    returns x, y, z coordinates of satellite galaxies assuming NFW profile.
    """
    # Rs is in the second column and it is in kpc so convert to Mpc
    M, Rv, Rs, xh, yh, zh = halos.T
    Rv /= 1000.0
    Rs /= 1000.0
    # Total number of satellites
    tot_sat = int(np.sum(n_sat))
    # Total number of halos
    nhalo = np.size(Rs)

    # Random direction
    x = np.random.randn(tot_sat)
    y = np.random.randn(tot_sat)
    z = np.random.randn(tot_sat)
    # Make it unit length
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    distance = np.zeros(tot_sat)
    # I need to generate distribuiton of distances according to NFW profile
    # I will generate uniform numbers - eta - and then transform to - r - in
    # such a way that r follows NFW.
    # I don't want r to be larger than virial radius which implies that eta can
    # not be larger than a certain number.
    # I will figure out what that number should be here.
    eta_max = np.log(1 + Rv) + 1.0/(1 + Rv) - 1
    # Concentration parameter is usually not much larger than 10
    eta = np.random.uniform(0, eta_max)
    for i in range(tot_sat):
        def func(r):
            np.log(1 + r) + 1.0/(1 + r) - 1 - eta[i]

        rstar = scipy.optimize.fsolve(func, np.ndarray([1]))
        distance[i] = Rs[i]*rstar
    dx = np.transpose(x*distance)
    dy = np.transpose(y*distance)
    dz = np.transpose(z*distance)
    # Now pull all x, y, z of satellites
    xsat, ysat, zsat = np.zeros((3, tot_sat))
    counter = 0
    for i in range(nhalo):
        for j in range(n_sat[i]):
            xsat[counter] = xh[i] + dx[counter]
            ysat[counter] = yh[i] + dy[counter]
            zsat[counter] = zh[i] + dz[counter]
            counter += 1

    return xsat, ysat, zsat 


def populate_hod(halos, hod_par):
    """
    Take halos and populate them with galaxies according to hod_par
    parameters.
    halo array must be a numpy array with 5 columns:
    halomass (Mpc/h), Rs (kpc/h), x, y, z (all Mpc/h comoving)
    returns x, y, z of galaxies.
    """
    Mhalos = halos[:, 0]
    Ncen, Nsat = populate(Mhalos, hod_par)
    xcen, ycen, zcen = np.transpose(halos[Ncen, -3:])
    xsat, ysat, zsat = place_satellites(halos, Nsat)
    xyzall = np.hstack(([xcen, ycen, zcen], [xsat, ysat, zsat]))
    return xyzall


"""
if __name__ == '__main__':
    start_time = time.time()
    halos = np.load('/mnt/data1/MDhalos.npy')
    print(time.time() - start_time)
    print(halos)
    HODpar = (13.08, 14.06, 0.98, 1.13, 0.9)
    xyzall = populate_hod(halos, HODpar)
    print(xyzall)
    print(time.time() - start_time)
"""
