'''
Populate a halo catalogue with galaxies using HOD parameters.

For now I will only use the simple 5 parameter HOD. 
'''

import scipy.special
import scipy.optimize
import numpy as np
import time

def populate(Mhalos, HODpar):
    '''
    Mhalos is a numpy array with halo masses.
    HODpar is a list of HOD parameters in the following order:
    (log10(Mcut), log10(M1), sigma, kappa, alpha)
    
    Returns two arrays one for centrals one for satellites. 
    The centrals array has 1 (there is a central) or 0 (no central).
    The satellites array has the number of satellites.
    '''

    Mcut, M1, sigma, kappa, alpha = HODpar
    # Central galaxies
    NcenProb = 0.5*scipy.special.erfc((Mcut - np.log10(Mhalos))/np.sqrt(2)/sigma)
    Ncen = np.random.poisson(NcenProb)
    # Can only have one central galaxy. Turn Ncen > 1 to 1
    Ncen[np.where(Ncen > 0)] = 1

    # Satellite galaxies
    NsatProb = NcenProb*((Mhalos - kappa*10**Mcut)/10**M1)**alpha
    NsatProb[np.where(np.isnan(NsatProb))] = 0
    NsatProb[np.where(NsatProb < 1)] = 0
    Nsat = np.random.poisson(NsatProb)

    return Ncen, Nsat

def place_satellites(halos, Nsat):
    '''
    halos - halo mass, virial radius, x, y, z
    Nsat - Number of satellites I need to place
    Both of the above are arrays.

    returns x, y, z coordinates of satellite galaxies assuming NFW profile.
    '''
    # Rs is in the second column and it is in kpc so convert to Mpc
    M, Rs, xh, yh, zh = halos.T
    Rs /= 1000.0
    # Total number of satellites
    TotSat = np.sum(Nsat)
    # Total number of halos
    Nhalo = np.size(Rs)

    # Random direction
    x = np.random.randn(TotSat)
    y = np.random.randn(TotSat)
    z = np.random.randn(TotSat)
    # Make it unit length
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    distance = np.zeros(TotSat)
    # Concentration parameter is usually not much larger than 10
    eta = np.random.uniform(0, 2, TotSat)
    for i in range(TotSat):
        func = lambda x: np.log(1+x) + 1.0/(1+x) - 1 - eta[i]
        rstar = scipy.optimize.fsolve(func, 1)
        distance[i] = Rs[i]**3*rstar
    dx = np.transpose(x*distance)
    dy = np.transpose(y*distance)
    dz = np.transpose(z*distance)
    # Now pull all x, y, z of satellites
    xsat, ysat, zsat = np.zeros((3, TotSat))
    counter = 0
    for i in range(Nhalo):
        for j in range(Nsat[i]):
            xsat[counter] = xh[i] + dx[counter]
            ysat[counter] = yh[i] + dy[counter]
            zsat[counter] = zh[i] + dz[counter]
            counter += 1

    return xsat, ysat, zsat 


def populateHOD(halos, HODpar):
    '''
    Take halos and populate them with galaxies according to HODpar
    parameters.
    halo array must be a numpy array with 5 columns:
    halomass (Mpc/h), Rs (kpc/h), x, y, z (all Mpc/h comoving)
    returns x, y, z of galaxies.
    '''
    Mhalos = halos[:,0]
    Ncen, Nsat = populate(Mhalos, HODpar)
    xcen, ycen, zcen = np.transpose(halos[Ncen, 2:])
    xsat, ysat, zsat = place_satellites(halos, Nsat)
    xyzall = np.hstack(([xcen, ycen, zcen], [xsat, ysat, zsat]))
    return xyzall

if __name__ == '__main__':
    start_time = time.time()
    halos = np.load('/mnt/data1/MDhalos.npy')
    print(time.time() - start_time)
    print(halos)
    HODpar = (13.08, 14.06, 0.98, 1.13, 0.9)
    xyzall = populateHOD(halos, HODpar)
    print(xyzall)
    print(time.time() - start_time)
