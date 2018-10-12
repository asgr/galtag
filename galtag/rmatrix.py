"""
Creates galaxy X group-central galaxy sparse distance matrix

For a pair of photoz-galaxy and group catalogues, 
the distance matrix needs to be calculated just once.

"""

from __future__ import division

__author__ = "Prajwal R Kafle <pkafauthor@gmail.com>"

import cosmolopy as cp
from astropy.coordinates import angle_utilities 
import scipy.sparse as ss
import numpy as np

def distmap(galaxy, central, constants):
    """
    Purpose
    -------
    Given galaxy and group central galaxy coordinates, calculates distance matrix between them.

    For distance R > r200 of the group, matrix is sparse. 

    Inputs
    ------
    galaxy: list 
            [ra, dec] angles in radians
    
    central: list
            [ra, dec, z, r200] 
            angles in radians, r200 in Mpc
    
    Outputs
    -------
    smatrix : sparse distance (Mpc) matrix in csc format [galaxy X group] dimension
    """
    ragal, decgal = galaxy
    racen, deccen, zcen, rvir = central

    # Angular distance to the group centre
    # in Mpc
    dangcen = cp.distance.angular_diameter_distance(zcen, 
                                                    z0=0, **constants)
    ngrp = racen.shape[0]
    ngal = ragal.shape[0]

    print 'NCentral, NGalaxy: [%d, %d]'%(ngrp, ngal)
    
    smatrix = ss.lil_matrix((ngrp, ngal))
                            
    for j, (decj, raj, dangj, rvirj) in enumerate(zip(deccen, racen, dangcen, rvir)):

        # I don't record galaxy-group pairs that are far from
        # 2sqr deg on sky
        ang = np.abs((decj - decgal)*(raj - ragal))
        angcut = ang<(0.05*(np.pi/180)**2)    #Only record angular separation < 0.025 sqr degree
        
        # Angular separation of galaxies from centrals
        angsep = angle_utilities.angular_separation(decj, raj, 
                                                    decgal[angcut], ragal[angcut])
        Rsep = angsep*dangcen[j]

        # Only keeping information for the galaxies within virial cone of the group.        
        Rsep[Rsep>rvirj] = 0

        Rj = np.zeros(ngal)       
        Rj[angcut] = Rsep.copy()

        smatrix[j] = Rj

        if np.remainder(j, 2000)==0:
            print '%1.2f'%(100*j/ngrp),

    print 'Density of sparse matrix: %1.1f per centage'%(100*smatrix.getnnz()/(smatrix.shape[0]*smatrix.shape[1]))

    # Transposing to get [galaxy X group] dimesion
    # and csc format to make it compatible to sparse.save_npz
    smatrix = smatrix.tocsc().transpose(copy=True)
    return smatrix
