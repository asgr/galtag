from __future__ import division

__author__ = "Prajwal R Kafle <pkafauthor@gmail.com>"

import numpy as np
import cosmolopy as cp

""" 
Galaxy formation and evolution utility code snippets 

"""

def vel(z, c = cp.cc.c_light_cm_s/1e5):
    """
    Purpose
    -------
    Calculate velocity from redshift using
       $v = c \frac{(1+z)^2 - 1}{(1+z)^2+1}$

    Parameters
    ------
    z : redshift
    c : speed of light in km/s
    Returns
    -------
    v: velocity #km/s
    """
    # return z*c/(1+z)
    return c*((1+z)**2-1)/((1+z)**2+1)


def smhm_behroozi(logMstar, z,  Mstar00 = 10.72, Mstar0a = 0.55,
                  M10 = 12.35, M1a = 0.28, beta0 = 0.44,
                  abeta  = 0.18, delta0 = 0.57, adelta = 0.17,
                  gamma0 = 1.56, agamma = 2.51 ):
    """
    Purpose
    -------
    Stellar mass - halo mass relation (Eqn 21) of Behroozi et al. 2010
    
    Parameters
    ----------
    args:

    logMstar : Stellar mass of a galaxy (in log10) 

    z :redshift of a galaxy

    kwargs:
    #Table 2 of Behroozi et al. Free mu,k and 0<z<1 case
    Mstar00 = 10.72, Mstar0a = 0.55, M10 = 12.35, 
    M1a = 0.28, beta0 = 0.44, abeta  = 0.18, delta0 = 0.57,
    adelta = 0.17, gamma0 = 1.56, agamma = 2.51
    
    Returns
    -------
    Dark matter halo mass in log10

    Notes
    -----
    Only works for logMstar = [8.8, 11.6] range.

    """

    # Warning
    if np.any(logMstar<8.8):
        print "Warning: LogMstar value less than Behroozi et al. 2010 Figure 2 range [8.8, 11.6]."
    elif np.any(logMstar>11.6):
        print "Warning: LogMstar value greater than Behroozi et al. 2010 Figure 2 range [8.8, 11.6]."
    
    #scale factor
    a = 1/(1+z)
    
    #Equation 21 of Behroozi et al
    logM1     = M10 + M1a*(a-1)
    logMstar0 = Mstar00 + Mstar0a*(a-1)
    beta      = beta0 + abeta*(a-1)
    delta     = delta0 + adelta*(a-1)
    gamma     = gamma0 + agamma*(a-1) 
    MstarbyMstar0 = 10**( logMstar - logMstar0 ) 
    return logM1 + beta*np.log10( MstarbyMstar0 ) +\
           (MstarbyMstar0**delta)/(1+(MstarbyMstar0)**-gamma) - 0.5


def veldisp_gapper(z, sigerr=None, c = cp.cc.c_light_cm_s/1e5):
    """
    Purpose
    --------
    Computes velocity dispersion of a galaxy group using the gapper method.

    Currently, this returns veldisp for a group with any occupancy, i.e., no lower limit in the occupancy set yet.

    Parameters
    ----------
    z: array_like
       Redshift of galaxies in a group

    sigerr: array_like
       Default value is None.
       Velocity error of galaxies in a group

    c : speed of light in km/s 

    Returns
    -------
    veldisp : velocity dispersion

    velerr : error in velocity dispersion 
    """
    
    #Velocities are first sorted in an increasing order
    vels = np.sort(c*z)/(1 + np.median(z))
    N    = len(vels)
    gaps = vels[1:] - vels[:-1]
    i    = np.arange(0, N - 1, 1)
    j    = i + 1 #0 index doesn't have any physical meaning
    wts  = j*(N - j)
    siggap = np.sum(gaps*wts)*np.sqrt(np.pi)/(N*(N-1.))
    veldispraw = np.sqrt((N*siggap**2)/(N-1.))

    if np.asarray(sigerr).any():
        sigerr  = np.sqrt(np.mean(sigerr**2))
        veldisp = np.sqrt(np.maximum(0, veldispraw**2-sigerr**2) )
        velerr  = sigerr
        veldispout = veldisp, velerr
    else:
        veldispout = veldispraw

    return veldispout


def angsep(phi, theta, deg=True):
    """
    Purpose
    -------
    Using Haversine calculates angular distance 2 points in sphere, i.e.,
    between (RA1, DEC1) and (RA2, DEC2).

    Parameters
    ----------
    phi : [ RA1 , RA2 ] in Degrees

    theta : [ Dec1, Dec2 ] in Degrees

    RA1, RA2, Dec1, Dec2 are scalars or numpy
    
    Returns
    -------
    Angular separation between 2 points in sphere, i.e.,
    between (RA1, DEC1) and (RA2, DEC2) in radians.
    
    See also Haversine
    """
    ra1, ra2   = phi
    dec1, dec2 = theta

    if deg==True:
        ra1, ra2   = np.radians(ra1), np.radians(ra2)
        dec1, dec2 = np.radians(dec1), np.radians(dec2)
    
    sin = np.sin
    cos = np.cos
    return np.arccos( sin(dec1)*sin(dec2)+cos(dec1)*cos(dec2)*cos(ra1-ra2) )


def zmax_func(Mabs, mapp_lim, cosmo=None):
    """
    Purpose
    -------
    Computes maximum observed redshift at an absolute magnitude of 
    a galaxy from an apparent mag limited survey.

    Parameters
    ----------
    Mabs : absolute magnitude of a galaxy
    mapp_lim : apparent magnitude limit of a galaxy

    Returns
    -------
    zmax : maximum observed redshift
    """
    # Distance modulus with a constant apparent magnitude limit of a survey
    DM = mapp_lim-Mabs
    distfunc, redfunc = cp.distance.quick_distance_function(cp.distance.luminosity_distance, return_inverse=True, **cosmo)
    zmax = redfunc(10**(DM/5.-5.))
    return zmax


def ke_corr(z, kcorrvals=[0.20848, 1.0226, 0.52366, 3.5902, 2.3843],
            zp=0.2, zref=0., Qzref=1.75):
    """
    Purpose
    -------
    Calculates K and E corrections at a given redshift

    Parameters
    ----------
    z : galaxy observed redshift
    
    Notes
    -----
    kwargs values are adopted from Robotham et al. 2011.
    Also, see Blanton and Roweis 2007.
    Be reminded, kcorrvals are function of zp and zref.
    """
    z_zp = z - zp
    z_zref = z - zref
    kcorr = kcorrvals[0] + kcorrvals[1]*z_zp + kcorrvals[2]*z_zp**2 + \
            kcorrvals[3]*z_zp**3 + kcorrvals[4]*z_zp**4
    ecorr = Qzref*z_zref
    return kcorr, ecorr


def modality(v):
    """
    Purpose
    -------
    In galaxy group studies modality is used as a normality discriminator.

    Inputs
    ------
    v: array_like
       velocity 
   
    Returns
    -------
    modality : array_like
       0.33 represents gaussian whereas 0.55 means uniform distributions
    """
    
    s = st.skew(vel, bias=False)
    k = st.kurtosis(vel, bias=False)
    m = (1+s**2)/(3+k**2)
    return s, k, m


def nmad(zspec, zphoto, cfr_thres=15):
    """
    Purpose
    -------
    Calculate scaled_bias, normalised median absolute deviation
    and a percentage of catastrophic outliers. 

    Used in photometric-redshift studies.

    Inputs
    ------
    zspec: array_like
           spectroscopic redshift
    zphoto: array_like
           photometric redshift
    cfr_thres: 15
           catastrophic failure rate threshold value in percentage

    Returns
    -------
    scaled_bias : array_like
         scaled_bias = (zphoto - zspec)/(1 + zspec)
    
    nmad : float
         nmad = Median(np.abs(|scaled_bias - median(scaled_bias)|))
    
    std : float
         standard deviation of the scaled_bias

    cfr : float (in %)
         catastrophic failure rate = |scaled_bias| > cfr_thres
 
    """
    cfr_thres_infrac = cfr_thres/100.
    zdiff = zphoto - zspec
    
    scaled_bias = zdiff/(1 + zspec)
    nmad = 1.4826*np.median( np.abs(scaled_bias - np.median(scaled_bias)))

    std = np.std(scaled_bias)
    
    cfr = np.abs(scaled_bias)>(cfr_thres_infrac)
    cfr = 100*scaled_bias[cfr].shape[0]/scaled_bias.shape[0]
    return scaled_bias, nmad, std, cfr
