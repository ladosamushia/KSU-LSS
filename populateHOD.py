"""
Populate a halo catalogue with galaxies using HOD parameters.

For now I will only use the simple 5 parameter HOD.

Want to write the code in such a way that I can run it "in phase" so that changes in cosmology are
coherently reflected. This is all related to the Abacus idea.
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
    n_cen_prob = 0.5 * scipy.special.erfc((Mcut - np.log10(halo_mass)) / np.sqrt(2) / sigma)

    # Satellite galaxies
    n_sat_prob = n_cen_prob * ((halo_mass - kappa * 10 ** Mcut) / 10 ** M1) ** alpha
    n_sat_prob[np.where(np.isnan(n_sat_prob))] = 0

    return n_cen_prob, n_sat_prob


def place_centrals(halos, n_cen_prob):
    """
    :param halos: array with 6 columns - halo mass, concentration radius, virial radius, x, y, z
    :param n_cen_prob: array of probabilities for a halo to host a central galaxy
    :return: Array of x, y, z of central galaxies (only one central per halo)
    """

    Ncen = np.random.poisson(n_cen_prob)
    # Can only have one central galaxy.

    return np.transpose(halos[Ncen != 0, -3:])


def nfw_cdf(r, eta):
    """
    :r: distance from the center of the halo
    :eta: probability
    :return: value of r for which probability is equal to eta (for NFW profile)

    This function computes a CDF of NFW radial profile of halos and returns the value of radisu for which the CDF is
    equal to a certain value.
    """
    return np.log(1 + r) + 1.0 / (1 + r) - 1 - eta


def satellite_xyz(r_virial, r_s):
    """
    :param r_virial: Virial radius of a host halo
    :param r_s: Concentration radius of a host halo
    :return: Randomly generated dx, dy, dz (with respect to the host halo) of a satellite following NFW.
    """
    # Random direction
    dxyz_sat = np.random.randn(3)
    # Normalise to unity
    norm = np.sqrt(dxyz_sat ** 2)
    dxyz_sat /= norm
    # I need to generate distribuiton of distances according to NFW profile
    # I will generate uniform numbers - eta - and then transform to - r - in
    # such a way that r follows NFW.
    # I don't want r to be larger than virial radius which implies that eta can
    # not be larger than a certain number.
    eta_max = np.log(1 + r_virial) + 1.0 / (1 + r_virial) - 1
    eta = np.random.uniform(0, eta_max)
    rstar = scipy.optimize.fsolve(nfw_cdf, np.ndarray([1]), args=eta)
    distance = r_s * rstar
    dxyz_sat *= distance

    return dxyz_sat


def place_satellites(halos, n_sat_prob, satellites):
    """
    :param halos: array with 6 columns - halo mass, concentration radius, virial radius, x, y, z
    :param n_sat_prob: array of average number of satellites per halo
    :param satellites: satellites that have been previously placed (for in-phase)
    :return: x, y, z of satellite galaxies
    """

    # This is actual number of satellites not mean.
    n_sat = np.random.poisson(n_sat_prob)
    tot_n_sat = np.sum(n_sat)
    # I will hold satellite x, y, z in here
    xyz_sat = np.zeros((3, tot_n_sat))
    # Rs is in the second column and it is in kpc so convert to Mpc. Same for Rv.
    # Carefull here, these are not copies. chancing Rv will change halos.
    M, Rv, Rs, xh, yh, zh = halos.T
    # Total number of halos
    nhalo = np.size(Rs)

    # Number of satellites needed in this specific halo. Some of them may have been pregenerated (in-phase)
    sat_num = np.array([len(sat) if sat is not None else 0 for sat in satellites])
    # How many more I need to create
    need_more = n_sat - sat_num
    # Add satellites as necessary
    for i in range(nhalo):
        if need_more[i] > 0:
            for j in range(need_more[i]):
                xyz = satellite_xyz(Rv[i]/1000, Rs[i]/1000)
                xyz += [xh[i], yh[i], zh[i]]
                if sat_num[i] == 0:
                    satellites[i] = [xyz]
                else:
                    satellites[i].append(xyz)
                sat_num[i] += 1

    counter = 0
    for i in range(nhalo):
        for j in range(n_sat[i]):
            xyz_sat[:, counter] = satellites[i][j]
            counter += 1
    return xyz_sat


def populate_hod(halos, hod_par, satellites):
    """
    :param halos: array of halo properties - six columns - mass, concentration radius, virial radius, x, y, z,
    :param hod_par: HOD parameters in the order - (log10(Mcut), log10(M1), sigma, kappa, alpha)
    :param satellites: satellites that already have been generated (in-phase). List of lists of np.array([x,y,z]).
    :return: x, y, z of both centrals and satellites.
    """
    mass_halos = halos[:, 0]
    n_cen_prob, n_sat_prob = populate(mass_halos, hod_par)
    xcen, ycen, zcen = place_centrals(halos, n_cen_prob)
    xsat, ysat, zsat = place_satellites(halos, n_sat_prob, satellites)
    xyzall = np.hstack(([xcen, ycen, zcen], [xsat, ysat, zsat]))
    return xyzall


"""
if __name__ == '__main__':
    halos = np.load('/Users/Lado/Downloads/halos_small.npy')
    HODpar = (13.08, 14.06, 0.98, 1.13, 0.9)
    Ncen, Nsat = populate(halos[:, 0], HODpar)
    xc, yc, zc = place_centrals(halos, Ncen)
    satellites = [[] for i in range(np.size(Ncen))]
    xyzall = populate_hod(halos, HODpar, satellites)
"""
