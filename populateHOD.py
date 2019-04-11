"""
Populate a halo catalogue with galaxies using HOD parameters.

For now I will only use the simple 5 parameter HOD.

Want to have two options. In-phase for when I want changes in cosmology to be
coherently reflected and Out-of-phase for when I want random phases.
This is all related to the Abacus idea.
"""

import scipy.special
import scipy.optimize
import numpy as np
# import time


def populate(halo_mass, hod_par):
    """
    :param halo_mass: Numpy array of halo masses
    :param hod_par: 5 HOD parameters in the following order -
    (log10(Mcut), log10(M1), sigma, kappa, alpha)
    :return: Ncen - Array that gives mean number of central galaxies in halos; Nsat - array that gives mean number of
    satellite galaxies in halos.
    """

    Mcut, M1, sigma, kappa, alpha = hod_par
    # Central galaxies
    n_cen_prob = 0.5*scipy.special.erfc((Mcut - np.log10(halo_mass)) / np.sqrt(2) / sigma)
    n_cen_prob[np.where(n_cen_prob > 1)] = 1

    # Satellite galaxies
    NsatProb = n_cen_prob * ((halo_mass - kappa * 10 ** Mcut) / 10 ** M1) ** alpha
    NsatProb[np.where(np.isnan(NsatProb))] = 0

    return n_cen_prob, NsatProb


def place_centrals(halos, n_cen_prob, seed='None'):
    """
    :param halos: array with 6 columns - halo mass, concentration radius, virial radius, x, y, z
    :param n_cen_prob: array of probabilities for ahalo to host a central galaxy
    :param seed: random seed to make sure I can rerun with the same phase. This is optional if equal to
    'None' a random seed is used.
    :return: Array of x, y, z of central galaxies
    """

    if seed != 'None':
        np.random.seed(seed)

    Ncen = np.random.poisson(n_cen_prob)
    # Can only have one central galaxy. Turn Ncen > 1 to 1
    Ncen[np.where(Ncen > 1)] = 1
    
    return np.transpose(halos[Ncen, -3:])


def place_satellites(halos, n_sat_prob, seed='None'):
    """
    :param halos: array with 6 columns - halo mass, concentration radius, virial radius, x, y, z
    :param n_sat_prob: array of average number of satellites per halo
    :param seed: random seed to make sure I can rerun with the same phase. This is optional if equal to
    'None' a random seed will be used.
    :return: x, y, z of satellite galaxies
    """

    if seed != 'None':
        np.random.seed(seed)

    # This is actual number of satellites not mean.
    n_sat = np.random.poisson(n_sat_prob)
    tot_n_sat = np.sum(n_sat)
    x_sat, y_sat, z_sat = np.zeros(3, tot_n_sat)

    # Rs is in the second column and it is in kpc so convert to Mpc. Same for Rv.
    M, Rv, Rs, xh, yh, zh = halos.T
    Rv /= 1000.0
    Rs /= 1000.0
    # Total number of halos
    nhalo = np.size(Rs)

    # Coordinates of possible satellites with respect to the center.
    # I will create twice the number of average satellites just in case.
    # I need to do this to ensure in phase change in cosmology.
    num_sat = int(n_sat_prob)*2
    # Random direction
    dx_sat = np.random.randn(num_sat)
    dy_sat = np.random.randn(num_sat)
    dz_sat = np.random.randn(num_sat)
    # Make it unit length
    norm = np.sqrt(dx_sat**2 + dy_sat**2 + dz_sat**2)
    dx_sat /= norm
    dy_sat /= norm
    dz_sat /= norm
    distance = np.zeros(num_sat)
    # I need to generate distribuiton of distances according to NFW profile
    # I will generate uniform numbers - eta - and then transform to - r - in
    # such a way that r follows NFW.
    # I don't want r to be larger than virial radius which implies that eta can
    # not be larger than a certain number.
    # I will figure out what that number should be here.
    eta_max = np.log(1 + Rv) + 1.0/(1 + Rv) - 1
    # Concentration parameter is usually not much larger than 10
    eta = np.random.uniform(0, eta_max)
    for i in range(num_sat):
        def func(r):
            np.log(1 + r) + 1.0/(1 + r) - 1 - eta[i]

        rstar = scipy.optimize.fsolve(func, np.ndarray([1]))
        distance[i] = Rs[i]*rstar
    dx_sat = np.transpose(dx_sat*distance)
    dy_sat = np.transpose(dy_sat*distance)
    dz_sat = np.transpose(dz_sat*distance)
    # I need to be careful with this loop to ensure matched phase
    counter = 0
    for i in range(nhalo):
        # This is not entirely correct. If I get more than twice the number of satellites than the mean I
        # will just ignore them. This has a very low likelihood of happening however so I do not think it
        # will affect the results much.
        for j in range(np.max(n_sat[i], int(2*n_sat_prob[i]))):
            x_sat[counter] = xh[i] + dx_sat[counter]
            y_sat[counter] = yh[i] + dy_sat[counter]
            z_sat[counter] = zh[i] + dz_sat[counter]
            counter += 1
        # Make sure we move to the next satellite. This is necessary to maintain matched phase.
        counter += 2*n_sat_prob[i] - n_sat[i]

    return x_sat, y_sat, z_sat


def populate_hod(halos, hod_par, seed):
    """
    :param halos: array of halo properties - six columns - mass, concentration radius, virial radius, x, y, z,
    :param hod_par: HOD parameters in the order - (log10(Mcut), log10(M1), sigma, kappa, alpha)
    :param seed: random seed in case I want matched phases. If equal to 'None' random seed is used.
    :return: x, y, z of both centrals and satellites.
    """
    mass_halos = halos[:, 0]
    n_cen_prob, n_sat_prob = populate(mass_halos, hod_par)
    xcen, ycen, zcen = place_centrals(halos, n_cen_prob, seed)
    xsat, ysat, zsat = place_satellites(halos, n_sat_prob, seed)
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
    # xyzall = populate_hod(halos, HODpar, 400)
    print(xyzall)
    print(time.time() - start_time)
"""
