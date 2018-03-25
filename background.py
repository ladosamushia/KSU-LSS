'''
Functions computing some cosmological quantities for the background expansion.
E.g. DA, H, f, G, etc.
'''
import numpy as np
from scipy.integrate import quad

def H(z,H0,Om,Ol,w0=-1,wa=0):
    Ok = 1 - Om - Ol
    DE =  Ol*(1+z)**(3*(1+w0+wa))*np.exp(-3*wa*z/(1+z))
    return H0*np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + DE)

def DA(z,H0,Om,Ol,w0=-1,wa=0):
    da = quad(lambda x: 1/H(x,H0,Om,Ol,w0,wa),0,z)[0]
    da *= 2997.9/(1+z)
    return da
