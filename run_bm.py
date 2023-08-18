import numpy as np
import TBG_lattice
import BM_model

theta = 1.05

N_freq = 5
N_supercell = 4
l_trunc = 40

# tbg lattices
bm_model = BM_model(theta, a=2.5, L=3.5, w0=0.11, w1=0.11, N_freq=N_freq, N_supercell=N_supercell)
bm_model.make_H()

pos_x, pos_y, pos_z = BM_model.position_mapping(l_trunc, l_trunc)

'''
Initial conditions 
1 and 2 are Gaussian distributions
3 and 4 are wave-packets on certain bands
'''
# ====== Generate initial condition I ====== # 
sigma = 10
f1_1 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
f1_2 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
f1_3 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
f1_4 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
f1 = lambda x,y: np.array([f1_1(x,y), f1_2(x,y), f1_3(x,y), f1_4(x,y)])
V1 = bm_model.map_wavepacket_func(f1)
np.save('data/init_1.npy', V1)

# ====== Generate initial condition II ====== # 
f2_1 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
f2_2 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma)) *1j
f2_3 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma)) *1j
f2_4 = lambda x,y: np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
f2 = lambda x,y: np.array([f2_1(x,y), f2_2(x,y), f2_3(x,y), f2_4(x,y)])
V2 = bm_model.map_wavepacket_func(f2)
np.save('data/init_2.npy', V2)

# ====== Generate initial condition III ====== # 
k1 = np.array([0, -0.002])
H1 = bm_model.H_BM(k1)
E1, V1= np.linalg.eig(H1)
E1 = E1.real
i = np.argpartition(E1, int(bm_model.dim_H/2))[int(bm_model.dim_H/2)]
psi1 = lambda x,y: bm_model.eigenvec_to_function(V1[:,i], k1, x, y)
gaussian = lambda x, y:  np.exp(-((x)**2+(y)**2)/(2*sigma*sigma))
f3 = lambda x,y: np.array(psi1(x,y)) * gaussian(x,y)
V3 = bm_model.map_wavepacket_func(f3)
np.save('data/init_3.npy', V3)


# ====== Generate initial condition IV ====== # 
k2 = np.array([0.01, -0.0275])
H2 = bm_model.H_BM(k2)
E2, V2= np.linalg.eig(H2)
E2 = E2.real
i = np.argpartition(E2, int(bm_model.dim_H/2))[int(bm_model.dim_H/2)]
psi2 = lambda x,y: bm_model.eigenvec_to_function(V2[:,i], k2, x, y)
f4 = lambda x,y: np.array(psi2(x,y)) * gaussian(x,y)
V4 = bm_model.map_wavepacket_func(f4)
np.save('data/init_4.npy', V4)


# ====== Chooose one init and run propagator ====== #
f = f1
times = np.linspace(0, 5, 6)
bm_model.decompose_function(f)

V_time_evo = np.zeros([times.size, V1.size], dtype='complex')
for i, t in enumerate(times):
    z = bm_model.bm_propagator(pos_x, pos_y, t)
    V_time_evo[i, :] = bm_model.map_BM_sol(z, pos_x, pos_y)

np.save(f'data/bm_1.npy', V_time_evo)



