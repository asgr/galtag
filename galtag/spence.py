"""
dilogarithmic/Spence function 

Utility that is only needed internally by the code.

"""

from __future__ import division

__author__ = "Prajwal R Kafle <pkafauthor@gmail.com>"

import scipy.special as ss
import numpy as np

def lirange1(x):
    return np.sum([(-1)**i*x**i/i**2 for i in np.arange(1,11,1)], axis=0)

def lirange2(x):
    ai = np.array([0, 1, 5,  1, 131, 661,  1327,  1163,    148969, 447047.])
    bi = np.array([1., 4, 24, 6, 960, 5760, 13440, 13440,  1935360, 6451200])
    return -np.pi**2/12 + np.sum([(np.log(2)/i - ai[i-1]/bi[i-1])*(1-x)**i for i in np.arange(1,11,1)], axis=0)

def lirange3(x, source):
    if source == 'correct':
        return -np.pi**2/6 - 0.5*np.log(x)**2 - np.sum([(-1)**i*x**(-i)/i**2 for i in np.arange(1,11,1)], axis=0)
    else:
        return -np.pi**2/6 - np.log(x)**2 + np.sum([(-1)**i*x**(-i)/i**2 for i in np.arange(1,11,1)], axis=0)

def li_negative(x, source='correct'):
    x = np.asarray(x)
    y = np.empty_like(x)

    cut = (x<0.35) 
    y[cut] = lirange1(x[cut])

    cut = (x>=0.35) & (x<1.95)
    y[cut] = lirange2(x[cut])

    cut = (x>=1.95)
    y[cut] = lirange3(x[cut], source)

    return y
    
if __name__=="__main__":
    import matplotlib.pyplot as plt

    r = np.logspace(-3, 2., 5000)
    rvir = 350 # kpc
    c = 10.
    y = c*r/rvir 

    plt.plot(y, li_negative(y, source='correct'), 'k', label='Recomputed')
    plt.plot(y, li_negative(y, source='incorrect'), 'r--', label='Duarte et al. 2015')
    plt.legend()
    plt.xlabel('y')
    plt.ylabel(r'Li$_{2}$(-x)')
    plt.tight_layout()
    plt.minorticks_on()
    plt.show()
