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
    kx = np.fft.fftfreq(Ngrid[0],dL)
    ky = np.fft.fftfreq(Ngrid[0],dL)
    kz = np.fft.rfftfreq(Ngrid[0],dL)
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
        print(i,np.sum(inbin),Pk[i])

    # Subtract shot noise
    SN = L**3/Ntot
    print('Shot noise:',SN)
    Pk -= SN

    # Centers of k bins
    kbin = (kbinedges[:-1] + kbinedges[1:])/2

    return [kbin, Pk]
