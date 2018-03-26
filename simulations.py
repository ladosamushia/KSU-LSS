'''
functions for processing simulations.
'''

from sys import argv
import numpy as np
import pickle

def grid_data(xyz,Nbins):
    '''
    Make a density grid NbinsxNbinsxNbins out of xyz coordinates.
    xyz -- 3xN array of x, y, z coordinates.
    '''
    grid, _ = np.histogramdd(xyz,bins=Nbins)
    return grid

def Fourier(grid):
    '''
    Fourier transform a grid of number of particles
    grid -- NxNxN grid of number of particles.
    '''
    g_ave = np.mean(grid)
    delta = (grid - g_ave)/g_ave
    delta = np.fft.rfftn(delta)
    return delta

def Pk(gridn,gridk,L,kmax,Nk):
    '''
    Compute P(k) from a delta(k) grid.
    gridn -- delta(r) grid
    gridk -- delta(k) grid
    L -- size of the cube
    kmax -- maximum wavenumber we want to go to
    Nk -- number of bins from 0 to kmax
    '''
    # Power spectrum binning
    kbinedges = np.linspace(0,kmax,Nk+1)
    Pk = np.zeros(Nk)

    Ngrid = np.shape(gridn)
    print('Ngrid:',Ngrid)
    gridSize = Ngrid[0]
    Ntot = np.sum(gridn)
    print('Ntot:',Ntot)
    dL = L/gridSize

    # Normalize F[n] to get F[delta]
    deltak = gridk

    # Wavenumbers
    kx = np.fft.fftfreq(gridSize,dL)
    ky = np.fft.fftfreq(gridSize,dL)
    kz = np.fft.rfftfreq(gridSize,dL)
    kx, ky, kz = np.meshgrid(kx,ky,kz)
    kk = np.sqrt(kx**2 + ky**2 + kz**2)*2*np.pi
    print('Nyquist:',kk.max())

    # Gridding window
    print('Correcting for gridding effects')
    WNGP = np.sinc(L*kx/2/gridSize)    
    WNGP *= np.sinc(L*ky/2/gridSize)
    WNGP *= np.sinc(L*kz/2/gridSize)
    deltak /= WNGP

    # Free up some memory
    kx = []
    ky = []
    kz = []
    WNGP = []

    # Compute power
    deltak = np.absolute(deltak)**2*L**3/gridSize**6
    deltak[0,0,0] = 0

    # Compute average Pk
    for i in range(Nk):
        inbin = np.logical_and(kk > kbinedges[i], kk < kbinedges[i+1])
        Pk[i] = np.sum(deltak[inbin])/np.sum(inbin)

    # Subtract shot noise
    SN = L**3/Ntot
    print('Shot noise:',SN)
    Pk -= SN

    # Centers of k bins
    kbin = (kbinedges[:-1] + kbinedges[1:])/2

    return [kbin, Pk]

def Bk(gridn,gridk,L,kmax,Nk):
    '''
    Compute B(k) from a delta(k) grid.
    This does not subtract shot-noise or correct for sampling effect.
    gridn -- delta(r) grid
    gridk -- delta(k) grid
    L -- size of the cube
    kmax -- maximum wavenumber we want to go to
    Nk -- number of bins from 0 to kmax
    '''
    # Binning
    kbinedges = np.linspace(0,kmax,Nk+1)
    # Centers of k bins
    kbin = (kbinedges[:-1] + kbinedges[1:])/2

    Ngrid = np.shape(gridn)
    print('Ngrid:',Ngrid)
    gridSize = Ngrid[0]
    Ntot = np.sum(gridn)
    print('Ntot:',Ntot)
    dL = L/gridSize

    # Normalize F[n] to get F[delta]
    deltak = gridk

    # Wavenumbers
    kx = np.fft.fftfreq(gridSize,dL)
    ky = np.fft.fftfreq(gridSize,dL)
    kz = np.fft.rfftfreq(gridSize,dL)
    kx, ky, kz = np.meshgrid(kx,ky,kz)
    kk = np.sqrt(kx**2 + ky**2 + kz**2)*2*np.pi
    print('Nyquist:',kk.max())

    # Free up some memory
    kx = []
    ky = []
    kz = []

    # Precompute the length of Bk data vector
    Bksize = 0
    for i1 in range(Nk):
        for i2 in range(i1,Nk):
            for i3 in range(min(i1+i2+1,Nk)):
                Bksize += 1
    Bk = np.zeros(Bksize)
    ktriplet = np.zeros((3,Bksize))

    # Compute average Bk in bins of k1, k2, k3
    counter = 0
    for i1 in range(Nk):
        kmin = kbinedges[i1]
        kmax = kbinedges[i1+1]
        not_inbin = np.logical_and(kk<kmin,kk>kmax)
        deltak = np.copy(gridk)
        deltak[not_inbin] = 0
        deltar1 = np.fft.irfftn(deltak)
        k1 = kbin[i1]
        for i2 in range(i1,Nk):
            kmin = kbinedges[i2]
            kmax = kbinedges[i2+1]
            not_inbin = np.logical_and(kk<kmin,kk>kmax)
            deltak = np.copy(gridk)
            deltak[not_inbin] = 0
            deltar2 = np.fft.irfftn(deltak)
            k2 = kbin[i2]
            # To make sure the triangular condition is satisfied
            for i3 in range(i2,min(i1+i2+1,Nk)):
                print(i1,i2,i3)
                k3 = kbin[i3]
                kmin = kbinedges[i3]
                kmax = kbinedges[i3+1]
                not_inbin = np.logical_and(kk<kmin,kk>kmax)
                deltak = np.copy(gridk)
                deltak[not_inbin] = 0
                deltar3 = np.fft.irfftn(deltak)

                Bisp = np.sum(deltar1*deltar2*deltar3)
                dk = kbinedges[1] - kbinedges[0]
                Bisp *= L**6/gridSize**12/(8*np.pi**2*k1*k2*k3*dk**3)

                Bk[counter] = Bisp
                ktriplet[:,counter] = [k1,k2,k3]
                counter += 1

    return ktriplet, Bk
