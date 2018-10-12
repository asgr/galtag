from __future__ import division

__author__ = "Prajwal R Kafle <pkafauthor@gmail.com>"

import numpy as np
import cosmolopy as cp

""" 
Halo utility code snippets 

"""
constants = {'h':1., 'omega_M_0':0.3, 'omega_lambda_0':0.7,
             'G': cp.constants.G_const_Mpc_Msun_s*cp.constants.Mpc_km**2}
# G in Mpc/Msun/(km/s)^2

class Vir():
    
    def __init__(self, z, delvir=200, dim=3, scale='comoving', empirical_factor=None, consts=constants):
        """
        Purpose
        --------
        Given group velocity dispersion and central galaxy redshift 
        calculate halo mass (mhalo)
        
        Parameters
        ----------
        z: array_like 
           redshift of the group-center
    
        delvir : 200  
                 density threshold
        dim: 3 (physical space) or 2 (projected space)
               When input is veldisp, dim = 3 means veldisp must be intrinsic and out put rvir is 3d and viceversa.

        scale : 'comoving' or 'angular'
               When input is veldisp and output rvir, scale = comoving yields rvir in comoving space.
               When input is rvir, and scale = 'comoving' then input rvir to be provided in comoving space.
        
        const : {'h':, 'omega_M_0':, 'omega_lambda_0':, 'G'}
        
        empirical_factor : None| True
                           if empirical_factor is True an additional empirical factor of 1/sqrt{2}
                           multiplied to the \sigma_v. Chris Power suggests this is a better prescription 
                           of DM halo.
 
        Returns
        -------
        veldisp : velocity dispersion in km/s
        mhalo : log10 halo virial mass in Msun
        rhalo : halo 3D virial radius in Mpc

        Notes
        -----
        Velocity dispersion of an observed galaxy group is 2D case.
        So, the dim flag must be used accordingly.
        
        Derived by solving equation 3 and 4 of Kafle et al. 2016.
        
        ASGR Celestial package assumes GM200/r200 = 1/2 (Sqrt(alpha)*veldisp)^2,
        that is an extra 1/2 factor. Hence, his code has 32 in denom. 
        To include this use 'empiricial_factor' kwarg.
        
        Also, see https://en.wikipedia.org/wiki/Virial_theorem
        
        Equation 65 of Peter Shneider's Extragalactic book.
        """
        self.empirical_factor = empirical_factor

        # Dimension
        self.dim = dim
        
        # Scale
        if scale == 'comoving':
            self.scale = 1 + z
        else:
            self.scale = 1.

            
        # Consts
        self.G = consts['G']    #4.302e-9 : Grav constant in Mpc Msun^-1 (km/s)^2
        H = 100*consts['h']*np.sqrt(consts['omega_M_0']*(1+z)**3 +
                                    consts['omega_lambda_0'] )
        rhocrit = 3 * H**2/(8 * np.pi * self.G)
        self.rhofactor = (4/3)*np.pi*delvir*rhocrit
        
        
    def veldisp2vir(self, veldisp):
        """
        Input
        -----
        veldisp : array_like
                  veldispersion 
        
                  velocity dispersion must be "intrinsic" velocity dispersion of the group and 
                  not of the line-of-sight. If los vel. dispersion to be used make dim=2.

        Returns
        -------
        rvir : halo virial radius in Mpc
        log10(mvir) : halo logarithmic virial mass in Msun
        """
        
        if self.empirical_factor == True:
            veldisp = veldisp/np.sqrt(2)

        if np.any(veldisp<=0):
            print 'Velocity dispersion is zero.'
        
        if self.dim == 2:
            veldisp = np.sqrt(3) * veldisp
    
        # Halo virial mass and radius
        mvir = veldisp**3/np.sqrt(self.G**3 * self.rhofactor)
        rvir = self.scale*(mvir/self.rhofactor)**(1/3)

        if self.dim == 2:
            rvir = rvir/1.37
            
        return rvir, np.log10(mvir)
    

    def rvir2veldisp(self, rvir):
        """
        Input
        -----
        rvir : array_like
               halo virial radius in Mpc unit
        Returns
        -------
        veldisp : in km/s
        log10(mvir) : in Msun unit
        """
        
        if self.dim == 2:
            rvir = rvir*1.37
            
        rvir = rvir/self.scale
        mvir = self.rhofactor*rvir**3

        veldisp = np.sqrt(self.G*mvir/rvir)
        
        if self.empirical_factor == True:
            veldisp = np.sqrt(2)*veldisp

        if self.dim == 2:
            veldisp = veldisp/np.sqrt(3)
            
        return veldisp

    
    def mvir2veldisp(self, mvir):
        """
        Input
        -----
        mvir : array_like
              halo mass in Msun unit

        Returns
        ------
        veldisp : in km/s
        rvir : in Mpc
        """
        
        rvir = (mvir/self.rhofactor)**(1/3)
        mvir = self.rhofactor*rvir**3

        veldisp = np.sqrt(self.G*mvir/rvir)
        
        if self.empirical_factor == True:
            veldisp = np.sqrt(2)*veldisp

        if self.dim == 2:
            veldisp = veldisp/np.sqrt(3)

        return veldisp
    
    def mvir2rvir(self, mvir):
        """
        Input
        -----
        mvir : halo mass in Msun unit
        
        Returns
        -------
        rvir : halo virial radius in Mpc
        """
        
        rvir = self.scale*(mvir/self.rhofactor)**(1/3)

        if self.dim == 2:
            rvir = rvir/1.37
        
        return rvir

    def rvir2mvir(self, rvir):
        """
        Input
        -----
        rvir : halo virial radius in Mpc

        Returns
        -------
        mvir : halo virial mass in log10 and Msun units
        """
        
        rvir = rvir/self.scale
        
        if self.dim == 2:
            rvir = rvir*1.37
            
        mvir = self.rhofactor*rvir**3
        return np.log10(mvir)


def cmvir( m200, z, ref='duffy08', consts=constants):
    """
    Purpose
    -------
    Given virial mass calculate concentration based on theoretical relations

    Inputs
    ------
    m200 : virial mass in MSun unit

    z : redshift

    ref  : duffy08 | maccio08 | king2011

           duffy08 relation is taken from Table 1 Duffy et al. 2008. 4th row
           maccio08 relation is taken from Equation 10 of Maccio et al. 2008 for WMAP5 cosmology
            
    Outputs
    -------
    c : concentration
    """

    if ref == 'duffy08':
        Mpivot = 2e12/consts['h']
        A200 = 6.71
        B200 = -0.091
        C200 = -0.44
        
        conc = A200 * (m200/Mpivot)**B200 * (1 + z)**C200
    
    elif ref == 'maccio08':
        # WMAP 1
        # conc = 10**(0.917 - 0.104*np.log10(m200*consts['h']/1e12))
        # WMAP 3
        # conc = 10**(0.769 - 0.083*np.log10(m200*consts['h']/1e12))

        # WMAP 5
        conc = 10**(0.830 - 0.098*np.log10(m200*consts['h']/1e12))

    return conc
