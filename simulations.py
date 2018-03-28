'''

lado.samushia.office@gmail.com

Functions for processing simulations.
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

def grid_weighted(xyz,weight,Nbins):
    '''
    Make a density grid NbinsxNbinsxNbins out of xyz coordinates.
    xyz -- 3xN array of x, y, z coordinates.
    '''
    grid, _ = np.histogramdd(xyz,bins=Nbins,weights=weight)
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

def Fourier_alt(grid):
    '''
    Fourier transform a grid of number of particles
    grid -- NxNxN grid of number of particles.
    This does the complex (not real) FFT.
    '''
    g_ave = np.mean(grid)
    delta = (grid - g_ave)/g_ave
    delta = np.fft.fftn(delta)
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

    deltak = np.copy(gridk)

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

def Pk_alt(gridn,gridk,L,kmax,Nk):
    '''
    This is an alternative Pk estimator that relies on bunch of FFTs. I am
    using it mainly as a prep for the bispectrum estimator.
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

    # Wavenumbers
    kx = np.fft.fftfreq(gridSize,dL)
    ky = np.fft.fftfreq(gridSize,dL)
    kz = np.fft.fftfreq(gridSize,dL)
    kx, ky, kz = np.meshgrid(kx,ky,kz)
    kk = np.sqrt(kx**2 + ky**2 + kz**2)*2*np.pi
    print('Nyquist:',kk.max())

    # Free up some memory
    kx = []
    ky = []
    kz = []

    Pk = np.zeros(Nk)

    # Compute Pk
    for i in range(Nk):
        kmin = kbinedges[i]
        kmax = kbinedges[i+1]
        k = kbin[i]
        not_inbin = np.logical_or(kk<kmin,kk>=kmax)
        deltak = np.copy(gridk)
        deltak[not_inbin] = 0
        deltak[0,0,0] = 0
        deltar = np.fft.irfftn(deltak)
        dk = kbin[1] - kbin[0]
        Npair = np.fft.irfftn(np.logical_not(not_inbin))
        Npair = np.sum(Npair**2)*gridSize**3
        Pk[i] = np.sum(deltar**2)*L**3/gridSize**3/Npair

    return Pk
    

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

    # Wavenumbers
    kx = np.fft.fftfreq(gridSize,dL)
    ky = np.fft.fftfreq(gridSize,dL)
    kz = np.fft.fftfreq(gridSize,dL)
    kx, ky, kz = np.meshgrid(kx,ky,kz)
    kk = np.sqrt(kx**2 + ky**2 + kz**2)*2*np.pi
    print('Nyquist:',kk.max())

    # Free up some memory
    kx = []
    ky = []
    kz = []

    # Precompute all Fourier transforms
    print('Computing Fourier transforms of spherical shells')
    deltas = np.zeros((Nk,gridSize,gridSize,gridSize))
    Ntri = np.zeros((Nk,gridSize,gridSize,gridSize))
    for i in range(Nk):
        kmin = kbinedges[i]
        kmax = kbinedges[i+1]
        not_inbin = np.logical_or(kk<kmin,kk>=kmax)
        Ntri[i,:,:,:] = np.real(np.fft.ifftn(np.logical_not(not_inbin)))
        deltak = np.copy(gridk)
        deltak[not_inbin] = 0
        deltas[i,:,:,:] = np.real(np.fft.ifftn(deltak))

    # Precompute number of Bk bins
    Bksize = 0
    for i1 in range(Nk):
        for i2 in range(i1,Nk):
            for i3 in range(i2,min(i1+i2+1,Nk)):
                Bksize += 1
    print('Number of bispectrum bins:',Bksize)
    Bk = np.zeros(Bksize)
    ktriplet = np.zeros((3,Bksize))

    # Compute average Bk in bins of k1, k2, k3
    counter = 0
    print('Computing Bispectrum in bins')
    for i1 in range(Nk):
        for i2 in range(i1,Nk):
            # To make sure the triangular condition is satisfied
            for i3 in range(i2,min(i1+i2+1,Nk)):
                deltaall = deltas[i1,:,:,:]*deltas[i2,:,:,:]*deltas[i3,:,:,:]
                Bisp = np.sum(deltaall)
                Ntriall = np.sum(Ntri[i1,:,:,:]*Ntri[i2,:,:,:]*Ntri[i3,:,:,:])
                Ntriall *= gridSize**6
                Bisp *= L**6/gridSize**3/Ntriall
                Bk[counter] = Bisp
                ktriplet[:,counter] = [kbin[i1],kbin[i2],kbin[i3]]
                counter += 1

    return ktriplet, Bk
