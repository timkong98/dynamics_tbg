from TBG_lattice import TBG
import numpy as np
from tqdm import tqdm
from scipy.linalg import expm

class TB_model(TBG):
    '''
    Define the tight-binding model using:
    theta: twist angle in deg, a: lattice constant, L: interlayer distance
    h: hopping function
    nn: use nearest-neighbor for intralayer
    '''
    def __init__(self, theta, a, L, hopping_type):
        super().__init__(theta, a, L)
        if hopping_type == 'SK':
            self.h = self.h_SK
        else:
            self.h = self.h_default
        self.H_tb = None

    def h_default(self, rx, ry, rz):
        if rz == 0 and np.abs(rx**2 + ry**2 - self.d**2)< 0.01:
            h = -3.048
        elif rz != 0:
            r = np.sqrt(rx**2 + ry**2 + rz**2)
            h_0 = 83.13493133107325
            h = h_0 * np.exp(-r)
        else:
            h = 0
        
        return h if np.abs(h)>1e-16 else 0


    def h_SK(self, rx, ry, rz):
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        a0 = self.a / np.sqrt(3)
        
        if r == 0:
            h = 0 # onsite
        else:
            h = -2.7 * np.exp(-(r - a0)/(0.319*a0)) * (1 - (rz / r)**2) + 0.48 * np.exp(-(r - self.L)/(0.319*a0)) * ((rz / r)**2)
        return h if np.abs(h)>1e-16 else 0


    def make_tight_binding_H(self, l_x, l_y):
        # make the tight-binding Hamiltonian 
        pos_x, pos_y, pos_z = self.position_mapping(l_x, l_y, flatten=True)
        pos_x = pos_x.flatten()
        N = pos_x.size # total number of atoms

        H = np.zeros([N, N], dtype='complex')
        for i in range(N):
            for j in range(N):
                rx = pos_x[i] - pos_x[j]
                ry = pos_y[i] - pos_y[j]
                rz = pos_z[i] - pos_z[j]
                H[i,j] = self.h(rx, ry, rz)

        self.H_tb = H

    
    def tb_propagator(self, t):
        return expm(-1j * t * self.H_tb)
