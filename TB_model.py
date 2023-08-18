import TBG_lattice
import numpy as np
from scipy.linalg import expm

class TB_model(TBG_lattice):
    '''
    Define the tight-binding model using:
    theta: twist angle in deg, a: lattice constant, L: interlayer distance
    h: hopping function
    nn: use nearest-neighbor for intralayer
    '''
    def __init__(self, theta, a, L, h, nn=True):
        super().init(theta, a, L)
        self.h = h
        self.nn = nn
        self.H_tb = None


    def make_tight_binding_H(self, l_x, l_y):
        # make the tight-binding Hamiltonian 
        pos_x, pos_y, pos_z = self.position_mapping(l_x, l_y)
        N = pos_x.size # total number of atoms
        t = 3.048

        H = np.zeros([N, N], dtype='complex')
        for i in range(N):
            for j in range(N):
                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]
                dz = pos_z[i] - pos_z[j]
                d_tot = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if self.nn and dz == 0:
                    if abs(np.sqrt(dx**2 + dy**2)-self.d)<0.0001 and i!=j:
                        H[i,j] = -t
                else:
                    H[i,j] = self.h(d_tot)

        self.H_tb = H

    
    def tb_propagator(self, t):
        return expm(-1j * t * self.H_tb)
