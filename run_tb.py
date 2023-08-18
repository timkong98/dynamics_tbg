import numpy as np
import TBG_lattice
import TB_model


# lattice coefficients
theta = 1.05
L = 3.5
a = 2.5

# hopping function coefficients
gamma_0 = 1
h_0 = 83.13493133107325
h = lambda x: h_0 * np.exp(-gamma_0 * x)

# truncation
l_trunc = 40

tb_model = TB_model(theta, a, L, h, nn=True)
tb_model.make_tight_binding_H(l_trunc, l_trunc)
tb_prop = tb_model.tb_propagator(t=1)

names = np.array([1, 2, 3, 4])
times = np.linspace(1, 5, 5)

for name in names:
    V = np.load(f'data/init_{name}.npy')
    #print('loaded successfully')
    V_time_evo = np.zeros([times.size, V.size], dtype='complex')
    for i, t in enumerate(times):
        V = tb_prop @ V
        V_time_evo[i, :] = V

    np.save(f'data/tb_{name}.npy', V_time_evo)
   

    

