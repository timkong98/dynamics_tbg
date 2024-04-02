from TBG_lattice import TBG
import numpy as np
from tqdm import tqdm
import scipy
import time

class BM_model(TBG):
    def __init__(self, theta, a, L, order, N_freq, N_supercell, hopping_func, wAA=1, wAB=1, terms=(True,True,True,True)):
        super().__init__(theta, a, L)

        self.N_freq = N_freq
        self.N_supercell = N_supercell

        self.w0=wAA
        self.w1=wAB

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
        self.k_list = np.zeros([self.Nk, 2])
        self.E_k = np.zeros([self.Nk, self.dim_H])
        self.V_k = np.zeros([self.Nk, self.dim_H, self.dim_H], dtype=complex) 

        self.V_decomposed = None
        self.order = order

        self.term_rot, self.term_quad, self.term_local, self.term_hop = terms 

        self.NN1 = list([(0,0), (-1,0), (0,1)])
        self.NN2= list([(1,1), (-1,1), (-1,-1)])
        self.NN3= list([(-2,-1), (-2,0), (0,2), (1,2), (1,0), (0,-1)])


        self.inter_terms, self.inter_kdep_terms = self.calc_coef(hopping_func)

        if hopping_func == 'default':
            self.monolayer = self.monolayer_default
        elif hopping_func == 'SK':
            self.monolayer = self.monolayer_SK

    @staticmethod
    def gradient(f, X, h=1e-6):
        D1 = (f(X + np.array([h,0])) - f(X - np.array([h,0])))/(2*h)
        D2 = (f(X + np.array([0,h])) - f(X - np.array([0,h])))/(2*h)
        return np.array([D1, D2])
    
    @staticmethod
    def Hankel_transform(f, K):
        # The fourier transform of a radial function can be express in terms of Hankel transform
        # \hat F(k) = 2*\pi*\int_0^\infty J_0(kr)F(r)r dr
        J_0 = lambda x: scipy.special.jv(0, x)
        k = np.linalg.norm(K)
        h = scipy.integrate.quad(lambda x: J_0(k*x)*f(np.array([x,0]))*x, 0, np.inf)
        return  2 * np.pi * h[0]

    
    def calc_coef(self, hopping_type):
        inter = dict()
        inter_k = dict()
        if hopping_type == 'default':
            h_hat = self.h_hat_default
            Dh_hat = self.Dh_hat_default
        elif hopping_type == 'SK':
            h_hat = self.h_hat_SK
            Dh_hat = self.Dh_hat_SK
        
        for hop in self.NN1 + self.NN2:
            inter[hop] = h_hat(self.K_2 + self.B_2 @ hop) / np.linalg.det(self.A) 
            inter_k[hop] = Dh_hat(self.K_2 + self.B_2 @ hop) / np.linalg.det(self.A) 

        return inter, inter_k
            

    def monolayer_default(self, k, layer, order):
        t_0 = -3.048
        v1 = -np.sqrt(3) * self.a * t_0 / 2
        v2 = self.a**2 * t_0 / 8
        if layer == 1:
            theta_i = -self.theta
        else:
            theta_i = self.theta
        fk = v1 * k @ [1, -1j]
        if order == 2 and self.term_quad:
            fk = fk + v2 * k.T @ np.array([[1, 1j],[1j, -1]]) @ k
        if self.term_rot:
            fk = fk * np.exp(theta_i / 2 * 1j)

        return np.array([[0, fk],[fk.conj(), 0]])
    
    def monolayer_SK(self, k, layer, order):
        # cut-off 3a
        d_AA = np.array([1, np.sqrt(3), 2, np.sqrt(7), 3]) * self.a
        d_AB = np.array([1, 2, np.sqrt(7), np.sqrt(13), 4, np.sqrt(19), 5, np.sqrt(28)]) / np.sqrt(3) * self.a

        h = lambda r: -2.7 * np.exp(-(r - self.d)/(0.319*self.d))

        s2_coef = np.array([1, -6, 4, 14, -18])
        t1_coef = np.array([-1, 2, 1, -5, -4, 7, 5, 2])
        t2_coef = np.array([1, 4, -13, -1, 16, 11, 25, -52])

        v2_AA = 3*self.a**2 / 4 * h(d_AA) @ s2_coef
        v1_AB = np.sqrt(3)/2 * self.a * h(d_AB) @ t1_coef
        v2_AB = self.a**2 / 8 * h(d_AB) @ t2_coef
        
        if layer == 1:
            theta_i = -self.theta
        else:
            theta_i = self.theta
        f_AA = 0
        f_AB = v1_AB * k @ [1, -1j]
        if order == 2 and self.term_quad:
            f_AA = f_AA + v2_AA * k.T @ np.array([[1, 0],[0, 1]]) @ k
            f_AB = f_AB + v2_AB * k.T @ np.array([[1, 1j],[1j, -1]]) @ k
        if self.term_rot:
            f_AA = f_AA * np.exp(theta_i / 2 * 1j)
            f_AB = f_AB * np.exp(theta_i / 2 * 1j)

        return np.array([[f_AA, f_AB],[f_AB.conj(), f_AA]])


    def h_default(self, X):
        # h: interlayer hopping function
        h_0 = 83.135
        alpha_0 = 1
        return h_0 * np.exp(-alpha_0 * np.sqrt(X.T @ X + self.L**2))


    def h_hat_default(self, xi):
        # \hat h: Fourier transform of interlayer hopping
        h_0 = 83.135
        alpha_0 = 1
        return 2*np.pi*h_0*alpha_0 * np.exp(-self.L * np.sqrt(xi.T @ xi + alpha_0 **2)) * (1 + self.L*np.sqrt(xi.T @ xi + alpha_0 **2)) / (xi.T @ xi + alpha_0**2)**(3/2)

    def Dh_hat_default(self, xi):
        # Derivative of \hat h, i=0 x-direction, i=1 y-direction
        h_0 = 83.135
        alpha_0 = 1
        s =  xi.T @ xi + alpha_0**2
        ans = -self.L / np.sqrt(s) * self.h_hat_default(xi)
        ans = ans - 3 / s * self.h_hat_default(xi) 
        ans = ans + self.L * (2*np.pi*h_0*alpha_0) * np.exp(-self.L * np.sqrt(xi.T @ xi + alpha_0**2)) / (s**2)
        return ans*xi

    def h_SK(self, X):
        V_pi = -2.7
        V_sigma = 0.48
        Delta = 0.319
        r = np.sqrt(X.T @ X + self.L**2)
        h = V_pi * np.exp(-(r - self.d)/(Delta*self.d)) * (1 - (self.L / r)**2) + V_sigma * np.exp(-(r - self.L)/(Delta*self.d)) * ((self.L / r)**2)
        return h
    
    def h_hat_SK(self, xi):
        return self.Hankel_transform(self.h_SK, xi)
    
    def Dh_hat_SK(self, xi):
        return self.gradient(self.h_hat_SK, xi)



    def H_BM(self, k, order):
        H_intra = np.zeros([self.dim_H, self.dim_H], dtype=complex)
        H_inter = np.zeros([self.dim_H, self.dim_H], dtype=complex)
        zero_block = np.zeros([2,2])
        
        for a in range(self.dim):
            l = self.freqs[:, a]
            for b in range(self.dim):
                n = self.freqs[:, b]

                # initialize the blocks
                H11, H12, H21, H22 = np.zeros([4,2,2])

                # intra layer terms
                if (l==n).all():
                    k_now_1 = k + self.B_m @ n
                    k_now_2 = k_now_1 + self.s1
                    H11 = self.monolayer(k_now_1, 1, order)
                    H22 = self.monolayer(k_now_2, 2, order)       
                
                H_intra[4*a:4*a+4, 4*b:4*b+4] = np.block([[H11, zero_block], [zero_block, H22]])

                # interlayer terms
                hops = self.NN1
                if order == 2 and self.term_hop:
                    hops = hops + self.NN2 + self.NN3

                for hop in hops:
                    if (l + hop == n).all():
                        h_val = self.inter_terms[hop]
                        H12 = h_val * self.T_matrix(self.B_2 @ hop)


                if order == 2 and self.term_local:
                    for hop in self.NN1:
                        if (l + hop == n).all():
                            K_G = self.K_2 + self.B_2 @ hop 
                            k_rel = k + self.B_m @ n + self.s1

                            grad_h = self.inter_kdep_terms[hop]
                            Dh_val = grad_h @ k_rel
                            
                            #print(n, np.linalg.norm(k_rel), np.linalg.norm(Dh_val))

                            H12 = H12 + Dh_val * self.T_matrix(self.B_2 @ hop)

                            
                H12 = H12 * np.array([[self.w0, self.w1], [self.w1, self.w0]])
                H_inter[4*a: 4*a+4, 4*b:4*b+4] = np.block([[zero_block, H12],[zero_block, zero_block]])
        
        H2 = H_inter + H_inter.conj().T
        #H1 = H_intra + H_intra.conj().T
        return H_intra + H2
    

    def make_H(self):
        u_supercell = np.linspace((-self.N_supercell)/2, (self.N_supercell-2)/2, self.N_supercell) / self.N_supercell
        #u_supercell = np.linspace((-self.N_supercell+1)/2, (self.N_supercell-1)/2, self.N_supercell) / self.N_supercell

        ux, uy = np.meshgrid(u_supercell, u_supercell)
        u = np.stack((ux.flatten(), uy.flatten()))
        
        for i in tqdm(range(self.Nk), desc='Solving for k'):
            k = self.B_m @ u[:, i]
            vals, vecs = np.linalg.eigh(self.H_BM(k, self.order))
    
            self.k_list[i, :] = k
            self.E_k[i, :] = vals
            self.V_k[i, :, :] = vecs

    def make_H_sym(self):
        u_supercell = np.linspace((-self.N_supercell+1)/2, (self.N_supercell-1)/2, self.N_supercell) / self.N_supercell

        ux, uy = np.meshgrid(u_supercell, u_supercell)
        u = np.stack((ux.flatten(), uy.flatten()))
        
        for i in tqdm(range(self.Nk), desc='Solving for k'):
            k = self.B_m @ u[:, i]
            vals, vecs = np.linalg.eigh(self.H_BM(k, self.order))
    
            self.k_list[i, :] = k
            self.E_k[i, :] = vals
            self.V_k[i, :, :] = vecs
    
    def make_H_old(self):
        u_supercell = np.linspace(0, (self.N_supercell-1), self.N_supercell) / self.N_supercell

        ux, uy = np.meshgrid(u_supercell, u_supercell)
        u = np.stack((ux.flatten(), uy.flatten()))
        
        for i in tqdm(range(self.Nk), desc='Solving for k'):
            k = self.B_m @ u[:, i]
            vals, vecs = np.linalg.eigh(self.H_BM(k, self.order))
    
            self.k_list[i, :] = k
            self.E_k[i, :] = vals
            self.V_k[i, :, :] = vecs


    def eigenvec_to_function(self, vec, k, x, y):
        if x.ndim == 1:
            x = np.array([x,x,x,x])
            y = np.array([y,y,y,y])
        psi = np.zeros_like(x, dtype='complex')
        
        for index in range(4):
            v = vec[index::4]
            X = np.array([x[index],y[index]])

            if index < 2:
                wave_number = (self.freqs.T @ (self.B_m.T @ X)) + k @ X 
            else:
                wave_number = (self.freqs.T @ (self.B_m.T @ X)) + (k + self.s1) @ X
            
            phi = np.exp(wave_number * 1j)
                
            psi[index] = np.sum(phi.T * v, axis=1)
        
        return psi

    
    def supercell_mesh(self, N, mesh_size):
        # a fine mesh for the supercell
        mesh = np.linspace(-N/2, N/2, mesh_size+1)
        mesh = mesh[0:mesh_size]
        mxx, myy = np.meshgrid(mesh, mesh)
        
        mx = self.A_m[0,0]*mxx + self.A_m[0,1]*myy
        my = self.A_m[1,0]*mxx + self.A_m[1,1]*myy
        mx = mx.flatten()
        my = my.flatten()
        
        weight = np.linalg.det(self.A_m) * (N / mesh_size) ** 2
        return mx, my, weight
    
    

    def decompose_function(self, f):
        # input initial condition f_init = (f1, f2, f3, f4)
        # update V_decomposed, a vector representation of f
        # using eigenvectors in V_k 
        self.V_decomposed = np.zeros_like(self.V_k)
        normalization = self.Nk * np.linalg.det(self.A_m) # eigenvectors from H is normalized


        mx, my, weight = self.supercell_mesh(N=np.sqrt(self.Nk), mesh_size=int(7* np.sqrt(self.Nk))) #N =  self.Nk
        z = f(mx, my)

        for i in tqdm(range(self.Nk), desc='Decomposing f with new integral'):
            k = self.k_list[i, :]
            vecs = self.V_k[i, :, :]

            for j in range(self.dim_H):
                vec = vecs[:, j]

                psi = self.eigenvec_to_function(vec, k, mx, my)
                integral = np.einsum('ij,ij->', np.conj(psi), z) * weight
                s = integral / normalization
                self.V_decomposed[i, :, j] = vec * s


    def bm_propagator(self, x, y, t):
        z = np.zeros_like(x, dtype='complex')
        
        for i in tqdm(range(self.Nk), desc=f'Running t={t}'):
            k = self.k_list[i, :]
            for j in range(self.dim_H):
                E = self.E_k[i, j]
                V = self.V_decomposed[i, :, j]
                
                psi = self.eigenvec_to_function(V, k, x, y)
                z += psi * np.exp(-E*t*1j)
            
        return z
    

    def map_BM_sol(self, z, x, y):
        # map the BM solution to values on the lattice by multiplying a phase
        z_out = np.zeros_like(z)
        for index in range(4):
            X = np.array([x[index],y[index]])

            if index < 2:
                z_out[index] = z[index] * np.exp(self.K_1 @ X *1j)
            else:           
                z_out[index] = z[index] * np.exp(self.K_2 @ X *1j)

        return z_out
            


def bm_run_gaussian(bm_model, times, coef, sigma, l_trunc, dir, name):
    # Setting up the model
    pos_x, pos_y, _ = bm_model.position_mapping(l_x=l_trunc, l_y=l_trunc)

    # Generate and save initial condition
    gaussian = lambda x, y:  np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
    c1, c2, c3, c4 = coef
    f = lambda x,y: np.array([c1*gaussian(x,y), c2*gaussian(x,y), c3*gaussian(x,y), c4*gaussian(x,y)])
    WP = bm_model.map_wavepacket_func(f, pos_x, pos_y)
    np.save(f'{dir}/init_{name}.npy', WP)

    bm_model.decompose_function(f)

    V_evo = np.zeros((times.size, WP.shape[0], WP.shape[1]), dtype='complex')
    for i, t in enumerate(times):
        Vi = bm_model.bm_propagator(pos_x, pos_y, t)
        V_evo[i] = bm_model.map_BM_sol(Vi, pos_x, pos_y)

    np.save(f'{dir}/bm_{name}.npy', V_evo)



def bm_run_band(bm_model, times, k, num_band, sigma, l_trunc, dir, name):
    # Setting up the model
    pos_x, pos_y, _ = bm_model.position_mapping(l_x=l_trunc, l_y=l_trunc)

    # Generate and save initial condition
    gaussian = lambda x, y:  np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))

    H = bm_model.H_BM(k, order=1)
    E, vec= np.linalg.eigh(H)
    i = int(bm_model.dim_H / 2) + num_band

    psi = lambda x,y: bm_model.eigenvec_to_function(vec[:,i], k, x, y)    
    f = lambda x,y: psi(x,y) * gaussian(x,y)
    WP = bm_model.map_wavepacket_func(f, pos_x, pos_y)
    np.save(f'{dir}/init_{name}.npy', WP)

    bm_model.decompose_function(f)

    V_evo = np.zeros([times.size, WP.shape[0], WP.shape[1]], dtype='complex')
    for i, t in enumerate(times):
        Vi = bm_model.bm_propagator(pos_x, pos_y, t)
        V_evo[i] = bm_model.map_BM_sol(Vi, pos_x, pos_y)

    np.save(f'{dir}/bm_{name}.npy', V_evo)
