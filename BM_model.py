import TBG_lattice
import numpy as np

class BM_model(TBG_lattice):
    def __init__(self, theta, a, L, w0, w1, N_freq, N_supercell):
        super().__init__(theta, a, L)

        self.N_freq = N_freq
        self.N_supercell = N_supercell

        self.w0=w0
        self.w1=w1

        # planewave frequencies
        freq = np.linspace(-N_freq, N_freq, 2*N_freq + 1)
        mm, nn = np.meshgrid(freq, freq)
        self.freqs = np.stack((mm.flatten(), nn.flatten()))
        self.dim = self.freqs.shape[1]
        self.dim_H = 4 * self.dim

        # k_list: coordinates of k in momentum space
        # E_k: for each k, the 2*dim eigenvalues
        # V_k: for each k and E_k, the 2*dim dimension eigenvectors
        self.Nk = N_supercell ** 2 
        self.k_list = np.zeros([self.NK, 2])
        self.E_k = np.zeros([self.Nk, self.dim_H])
        self.V_k = np.zeros([self.N_tot, self.dim_H, self.dim_H], dtype=complex) 

        self.V_decomposed = None


    def H_BM(self, k):
        H = np.zeros([self.dim_H, self.dim_H], dtype=complex)
        v = 6.6
        
        kx, ky = k  
        s1x, s1y = self.s1
        s2x, s2y = self.s2
        s3x, s3y = self.s3
        b1x, b1y = self.b_m1
        b2x, b2y = self.b_m2
        
        for a in range(self.dim):
            l1, l2 = self.freqs[:, a]
            for b in range(self.dim):
                n1, n2 = self.freqs[:, b]
                
                
                # intra layer terms
                H12 = v*(kx+n1*b1x+n2*b2x - (ky+n1*b1y+n2*b2y)*1j) if l1 == n1 and l2 == n2 else 0
                H21 = v*(kx+n1*b1x+n2*b2x + (ky+n1*b1y+n2*b2y)*1j) if l1 == n1 and l2 == n2 else 0
                
                H34 = v*(s1x+kx+n1*b1x+n2*b2x - (s1y+ky+n1*b1y+n2*b2y)*1j) if l1 == n1 and l2 == n2 else 0
                H43 = v*(s1x+kx+n1*b1x+n2*b2x + (s1y+ky+n1*b1y+n2*b2y)*1j) if l1 == n1 and l2 == n2 else 0
                
    
                # interlayer terms
                c1 = 1 if l1 == n1 and l2 == n2 else 0
                c2 = 1 if l1 == n1 and l2 == n2-1 else 0
                c3 = 1 if l2 == n2 and l1 == n1+1 else 0
                
                d1 = 1 if l1 == n1 and l2 == n2 else 0
                d2 = 1 if l1 == n1 and l2 == n2+1 else 0
                d3 = 1 if l2 == n2 and l1 == n1-1 else 0
                
                w_coef = np.array([[self.w0, self.w1], [self.w1, self.w0]])
                M1 = w_coef*(self.T1*c1 + self.T2*c2 + self.T3*c3)
                M2 = w_coef*(np.conj(self.T1.T)*d1 + np.conj(self.T2.T)*d2 + np.conj(self.T3.T)*d3)

                
                [[H13, H14], [H23, H24]] = M1 
                [[H31, H32], [H41, H42]] = M2
                
                H[4*a: 4*a+4, 4*b:4*b+4] = np.array([[0, H12, H13, H14],
                                                    [H21, 0, H23, H24],
                                                    [H31, H32, 0, H34],
                                                    [H41, H42, H43, 0]])
        return H
    

    def make_H(self):
        u_supercell = np.linspace(0, self.N_supercell-1, self.N_supercell) / self.N_supercell
        ux, uy = np.meshgrid(u_supercell, u_supercell)
        ux = ux.flatten()
        uy = uy.flatten()
        
        for i in range(self.Nk):
            kx = self.b_m1[0]*ux[i] + self.b_m2[0]*uy[i]
            ky = self.b_m1[1]*ux[i] + self.b_m2[1]*uy[i]
            k = np.array([kx, ky])
            vals, vecs = np.linalg.eigh(self.H_BM(k))
    
            self.k_list[i, :] = k
            self.E_k[i, :] = vals.real
            self.V_k[i, :, :] = np.where(np.abs(vecs.real)<1e-12, 0, vecs)


    def eigenvec_to_function(self, vector, k, x, y):
        psi1, psi2, psi3, psi4 = 0, 0, 0, 0
        kx, ky = k
        s1x, s1y = self.s1
        b1x, b1y = self.b_m1
        b2x, b2y = self.b_m2
        
        for i in range(self.dim):
            c1, c2, c3, c4 = vector[list(range(4*i, 4*i+4))]
            n1, n2 = self.freqs[:, i]
            # plane wave with potential of layer 1
            psi_l1 = np.exp((kx*x+ky*y)*1j) *  np.exp((n1*(b1x*x+b1y*y)+n2*(b2x*x+b2y*y))*1j)
            # layer 2
            psi_l2 = psi_l1 * np.exp((s1x*x+s1y*y)*1j)
            
            psi1 = psi1 + c1 * psi_l1
            psi2 = psi2 + c2 * psi_l1 
            psi3 = psi3 + c3 * psi_l2 
            psi4 = psi4 + c4 * psi_l2 
            
        return psi1, psi2, psi3, psi4
    

    def int_supercell(self, f, g):
        N = 1
        # a fine mesh for the supercell
        meshsize = 100
        meshx = np.linspace(-N/2, N/2, meshsize+1)
        meshx = meshx[0:meshsize]
        mxx, myy = np.meshgrid(meshx, meshx)
        
        mx = self.a_m1[0]*mxx + self.a_m2[0]*myy
        my = self.a_m1[1]*mxx + self.a_m2[1]*myy
        
        area = np.cross(self.a_m1, self.a_m2) * (N / meshsize)**2
        s = np.multiply(np.conj(f(mx, my)), g(mx, my)) * area
        return np.sum(s)
    

    def decompose_function(self, f):
        # input initial condition f_init = (f1, f2, f3, f4)
        # update V_decomposed, a vector representation of f
        # using eigenvectors in V_k 
        
        for i in range(self.Nk):
            k = self.k_list[i, :]
            vecs = self.V_k[i, :, :]

            for j in range(self.dim_H):
                vec = vecs[:, j]
                psi = lambda x,y: self.eigenvec_to_function(vec, k, x, y)
                
                a1 = self.int_supercell(psi, f)
                a2 = self.Nk * np.cross(self.a_m1, self.a_m2) # eigenvectors from H is normalized
                a = a1 / a2
                self.V_decomposed[i, :, j] = vec * a if np.abs(a) >1e-8 else 0


    def bm_propagator(self, x, y, t):
        z1a, z1b, z2a, z2b = 0, 0, 0, 0
        
        for i in range(self.Nk):
            k = self.k_list[i, :]
            for j in range(self.dim_H):
                E = self.E_k[i, j]
                V = self.V_decomposed[i, :, j]
                
                psi = lambda x,y: self.eigenvec_to_function(V, k, x, y)
                z = psi(x,y)
                z1a = z1a + z[0]*np.exp(-E*t*1j)
                z1b = z1b + z[1]*np.exp(-E*t*1j)
                z2a = z2a + z[2]*np.exp(-E*t*1j)
                z2b = z2b + z[3]*np.exp(-E*t*1j)
            
        return (z1a, z1b, z2a, z2b)
    

    def map_BM_sol(self, z, pos_x, pos_y):
        # map the BM solution to values on the lattice by multiplying a phase
        z1a, z1b, z2a, z2b = z
        N = z1a.size

        z_with_phase = np.zeros(x.size, dtype='complex')
        k1x, k1y = self.K_1
        k2x, k2y = self.K_2
        for i in range(x.size):
            x = pos_x[i]
            y = pos_y[i]
            if i < N:
                z_with_phase[i] = z1a[i] * np.exp((k1x*x+k1y*y)*1j)
            elif i < 2*N:
                z_with_phase[i] = z1b[i] * np.exp((k1x*x+k1y*y)*1j)
            elif i < 3*N:
                z_with_phase[i] = z2a[i] * np.exp((k2x*x+k2y*y)*1j)
            else:           
                z_with_phase[i] = z2b[i] * np.exp((k2x*x+k2y*y)*1j)
        return z_with_phase
        
        



