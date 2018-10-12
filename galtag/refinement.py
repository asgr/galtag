"""
Duarte and Mamon et al. 2015 model.
"""

from __future__ import division

__author__ = "Prajwal R Kafle <pkafauthor@gmail.com>"

import numpy as np
import spence as li
import galevol
import pandas 
from scipy.interpolate import interp1d

def _g(x):
    return np.log(1+x) - x/(1+x)

class gal2grp():
    
    def __init__(self, R, zgal, grp, constants):
        """
        Purpose
        -------
        To calculate assignment probability of a galaxy to a group. 

        Scalar code
        
        Inputs
        ------
        R : projected distance of a galaxy from a group in Mpc

        zgal : redshift grid of a galaxy to be assigned

        grp : list of a group coordinates 
              [m200, v200, r200, c200, rs, z}
               
              m200 group virial mass in Msun
              r200 group virial radius (3D) in Mpc.
        """
        
        self.G = constants['G']
        self.n200 = 1.    # This is arbitrary as it cancels out. Refer to Duarte et al. 2015
        
        self.m200, self.v200, self.r200, self.c200, self.rs, zgrp = grp
        
        self.R = R    #angular_separation*dang

        #v = LOS velocity v of a galaxy relative to a central group galaxy
        self.v = constants['c']*(zgal - zgrp)/(1 + zgrp)

    def rsigma_sqr(self, r):
        """
        Group-centric radial velocity dispersion 

        Equation A4 of Duarte et al. 2015
        """
    
        y = self.c200*r/self.r200
        yplus1 = y + 1

        term0 = self.c200/(6*y*yplus1)
        term1 = _g(self.c200)

        term2 = 6*(y*yplus1)**2*li.li_negative(y)
        term3 = 6*y**4*np.arctanh(1/(2*y + 1))
        term4 = 3*y**2*(2*y + 1)*np.log(y)
        term5 = (y*yplus1)**2*(np.pi**2 + 3*np.log(yplus1)**2)
        term6 = 3*(2*y**2-1)*np.log(yplus1)
        term7 = 3*y*yplus1*(3*y + 1)

        
        termsum = term2 + term3 - term4 + term5 - term6 - term7
        
        coeff = self.G*self.m200/self.r200

        rsigsqr = coeff*(term0/term1)*termsum

        if (rsigsqr<0).any():
            if (termsum<0).any():
                error_str = 'termsum < 0'
            else:
                error_str = 'term1 or term0 = 0'
                
            raise ValueError('%s: rsig_sqr can not be negative'%error_str)
        
        return rsigsqr

    
    def anisotropy(self, r):
        """
        Velocity anisotropy of a group 

        Equation 18
        """
        
        return 0.5*r/(r + self.rs)

    
    def zsigma_sqr(self, r, R):
        """
        Equation 13 
        """
        return (1 - self.anisotropy(r)*(R/r)**2)*self.rsigma_sqr(r)

    
    def hfunc(self, r, R, v, veps):
        """
        line-of-sight velocity distribution of a group

        Equation 12
        """
        zsigsqr = self.zsigma_sqr(r, R) + veps**2
        coeff =  1/np.sqrt(2*np.pi*zsigsqr)
        return coeff*np.exp(-0.5*v**2/zsigsqr)

    
    def rho(self, r):
        """
        NFW density 
        """
        x = r/self.r200
        coeff = self.n200/(4*np.pi*self.r200**3)
        rhokernel = 1/(_g(self.c200)*x*(x + 1/self.c200)**2)
        return coeff*rhokernel

    
    def ghfunc(self, rnum, veps):
        """
        Halo density in projected phase space for one galaxy (v,R)
        
        Equation 11
        """
        # In r
        r = np.linspace(self.R, self.r200, rnum)[:,None]
        integrand = r*self.rho(r)*self.hfunc(r, self.R, self.v, veps)

        return 2*np.trapz(integrand, r, axis=0)

        # In u, but scalar
        # Variable transformation to r = R coshu
        # u = np.linspace(0., np.arccosh(self.r200/self.R), rnum)
        # r = self.R*np.cosh(u[:,None])
        # integrand = r*self.rho(r)*self.hfunc(r, self.R, self.v, veps)
        # return 2*np.trapz(integrand, u, axis=0)
        

    def gifunc(self):
        """
        Interloper density in projected phase space 
        
        Equation 19, 20
        """
        
        # Eqn 24, 25, 26
        x = self.R/self.r200
        Ax = 10**(-1.092 - 0.01922*x**3 + 0.1829*x**6)
        sigx = 0.6695 - 0.1004*x**2
        B = 0.0067
        
        coeff = self.n200/(self.r200**2*self.v200)

        # gkcore can be very large for which exp function will be 0
        # This part can be optimised
        gkcore = 0.5*(self.v/self.v200/sigx)**2
        gkernel = Ax*np.exp(-gkcore) + B 
        return coeff*gkernel

    
    def total_probab(self, rnum, veps):
        """
        Probability of a galaxy to be in a group.

        Equation 8 

        rnum is grid in u where r = R Cosh(u)
        veps = error in los-velocity or galaxy redshift
        """

        # In case we desire to run code per galaxy per group basis
        if self.R>self.r200:
            # Outside virial radius 0 probab
            return 0., 0.
        else:
            # For in between galaxies calculate a probab
            ghterm = self.ghfunc(rnum, veps)
            giterm = self.gifunc()
            return ghterm, ghterm/(ghterm + giterm)

        
class refinement( ):
    
    def __init__(self, galdf, grpdf, dmatrix, constants):
        """
        Purpose
        -------
        To calculate assignment probability of all galaxies to potential groups

        
        Inputs
        ------
        galdf: galaxy dataframe [galaxy X columns]
              
              rows
              ----
              Each row must be a unique galaxy

              columns
              -------
              p(z) : redshift distribution in zgrid

                     Minimum required column headers are strings of 
                     redshift grid

                     zgrid = np.arange(0.005, 1.5, 0.01)
                     zgrid_string = zgrid.astype(str)
              
              Extra columns (e.g CATAID, RA, Dec, Rpetro etc) 
              are welcome too, as code won't touch them. 

        grpdf: group datafram [group X columns]

              rows
              ----
              Each row must be a unique group

              columns
              -------
              Minimum required column headers are [m200, r200, CenZ]
              
              m200 in Msun; r200 is 3D virial radius in Mpc
              z is group centre radius

        dmatrix: [galaxy X group]
                 distance (in Mpc) sparse  matrix (in scipy.sparse CSR format) 
                 created from rmatrix.py  

        constants: dictionary of constantsants such as
                   G: Gravitation constant in Mpc/Msun/(km/s)^2
                   c: speed of light in km/s

        Returns
        -------
        refgaldf : [galaxy X columns]
                   
                   same as input galdf with value-added data, including, 
                   refined redshift, probability 

        Note
        ----
        hardcoded : ref='duffy08'
        """
        self.galdf = galdf
        self.grpdf = grpdf
        self.dmat = dmatrix
        self.constants = constants
        
        self.ngal, ngrp = dmatrix.shape

        print 'Ngal, Ngrp: [%d, %d]'%(self.ngal, ngrp)
        
        # Checking dataframes and distance matrix have same dimension
        assert galdf.shape[0] == self.ngal
        assert self.grpdf.shape[0]== ngrp

        # Adding few more group virial properties
        # vcircular in km/s; rs in Mpc and c is concentration
        self.grpdf['v200'] = np.sqrt(constants['G'] * 
                                     (self.grpdf['m200'])/self.grpdf['r200'])    
        self.grpdf['c200'] = galevol.halo.cmvir(self.grpdf['m200'].values,   
                                                self.grpdf['CenZ'], ref='duffy08')
        self.grpdf['rs'] = self.grpdf['r200']/grpdf['c200']

        
    def g2G(self, zgrid, rnum, veps=0.0, zrange=[0., 2.]):
        """
        For loop over galaxies for each set of groups where distance matrix is dense

        rnum = 1000 # how fine r grid to be 
        veps = 0.0 
        zrange = [zmin, zmax] 
                 maximum or minimum values of photo-z of the overall sample
        
        """
        zmin, zmax = zrange
        
        zgrid_string = zgrid.astype(str)
        
        for i, row in self.galdf.iterrows():

            pzgal = row[zgrid_string].values
            pzgal = interp1d(zgrid, pzgal)

            
            # Find trim grpdf where dense galaxy-group distance matrix is dense
            dense_grp = self.dmat[i].indices

            trim_grpdf = self.grpdf.loc[dense_grp].reset_index()

            # For a subset of potential group add R column 
            trim_grpdf['R'] = self.dmat[i].data

            df = pandas.DataFrame(columns=['zexpfull', 'zexphalo', 'ptotfull', 'ptothalo',
                                           'HaloID', 'HaloZ', 'm200', 'r200'])
    
            for j, grp in trim_grpdf.iterrows():                
                grpcoords = [grp['m200'], grp['v200'], grp['r200'],
                             grp['c200'], grp['rs'], grp['CenZ']  ]
                                
                # High res zgrid near CenZ and coarse in the wings
                zleft = np.maximum(zmin, grp['CenZ']-0.05)
                zright = np.minimum(grp['CenZ']+0.05, zmax)
                
                finezgrid = np.hstack(( np.linspace(zmin, zleft, 5),
                                        np.linspace(zleft, zright, 400),
                                        np.linspace(zright, zmax, 5) ))

                gal2grp_inst = gal2grp(grp['R'], finezgrid, grpcoords, self.constants)
                phalo, pmaggie = gal2grp_inst.total_probab(rnum, veps)

                pref = pmaggie*pzgal(finezgrid)
                ptot = np.trapz(pref, finezgrid)

                if ptot < 1e-15:
                    # If the probability to go in a group is < 1e-15
                    # no need to complete this loop, hence, skip.
                    continue
                
                # <z> refined redshift
                zexp = np.trapz(finezgrid*pref/ptot, finezgrid)
                
                # ---------- halo calculation -------------
                # <z> using phalo probability only
                pref2 = phalo*pzgal( finezgrid )
                ptot2 = np.trapz(pref2, finezgrid )
                zexp2 = np.trapz(finezgrid*pref2/ptot2, finezgrid) 
                
                # Save list of potential Groups for a galaxy
                df.loc[j, 'zexpfull'] = zexp 
                df.loc[j, 'ptotfull'] = ptot
                
                df.loc[j, 'zexphalo'] = zexp2 
                df.loc[j, 'ptothalo'] = ptot2 

                df.loc[j, 'HaloID'] = grp['HaloID'] 
                df.loc[j, 'HaloZ'] = grp['CenZ'] 
                df.loc[j, 'HaloR'] = grp['R'] 
                df.loc[j, 'm200'] = grp['m200'] 
                df.loc[j, 'r200'] = grp['r200'] 
            
            df.sort_values('ptotfull', ascending=False, inplace=True)

            if df.empty:
                df.loc[0, 'zexpfull'] = -9999.99 
                df.loc[0, 'ptotfull'] = -9999.99
                df.loc[0, 'zexphalo'] = -9999.99
                df.loc[0, 'ptothalo'] = -9999.99 
                df.loc[0, 'HaloID'] = -9999 
                
            # Updating galaxy catalogue with refined redshift,
            # probabilities etc
            self.galdf.loc[i, 'zexpfull'] = df.iloc[0]['zexpfull']
            self.galdf.loc[i, 'zexphalo'] = df.iloc[0]['zexphalo']
            self.galdf.loc[i, 'ptotfull'] = df.iloc[0]['ptotfull']
            self.galdf.loc[i, 'ptothalo'] = df.iloc[0]['ptothalo']
            self.galdf.loc[i, 'HaloID'] = df.iloc[0]['HaloID']

            if np.remainder(i, 2000)==0:
                print '%1.2f'%(100*i/self.ngal)
            
        return self.galdf
        


#import pdb
#pdb.set_trace()

"""
# ---------- full calculation -------------
gal2grp_inst = gal2grp(grp['R'], zgrid, grpcoords, self.constants)

# phalo = only halo probability
# pmaggie = phalo/(phalo+pinterloper) probability
phalo, pmaggie = gal2grp_inst.total_probab(rnum, veps)

# p(R, v) X p(zg)
pref = pmaggie*pzgal

# Total probability
ptot = np.trapz(pref, zgrid)    


if ptot < 1e-10:
# If the probability to go in a group is < 1e-6
# no need to complete this loop, hence, skip.
continue

# <z> refined redshift
zexp = np.trapz(zgrid*pref/ptot, zgrid)

# import pdb
# pdb.set_trace()

# ---------- halo calculation -------------
# <z> using phalo probability only
pref2 = phalo*pzgal
ptot2 = np.trapz(pref2, zgrid)
zexp2 = np.trapz(zgrid*pref2/ptot2, zgrid) 
"""
