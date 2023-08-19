import numpy as np
from TBG_lattice import TBG
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from cycler import cycler

# plot the vectors into 2 plots
def plot_vec(pos_x, pos_y, V, R, title, save=True):
    
    fig, ((ax1, ax2)) = plt.subplots(1,2, figsize=(15,6))
    
    value = np.abs(V)
    value = value / np.linalg.norm(V)
    vmax = np.max(value)

    cm1 = LinearSegmentedColormap.from_list('', ['white', "#e64b35"])
    N = int(pos_x.size / 4)
    cf1 = ax1.scatter(pos_x[:2*N], pos_y[:2*N], c=value[:2*N], vmin=0, vmax=vmax, s=10, cmap=cm1)
    cf2 = ax2.scatter(pos_x[2*N:], pos_y[2*N:], c=value[2*N:], vmin=0, vmax=vmax, s=10, cmap=cm1)
    
    ax1.set_xlim(-R,R)
    ax1.set_ylim(-R,R)
    ax2.set_xlim(-R,R)
    ax2.set_ylim(-R,R)
    fig.colorbar(cf1, ax=ax1, format='%.2f')
    fig.colorbar(cf2, ax=ax2, format='%.2f')
    ax1.set_title(title + ', layer 1')
    ax2.set_title(title + ', layer 2')
    ax1.set_aspect(1)
    ax2.set_aspect(1)
    # plt.show()
    if save:
        # plt.savefig("images/' + title + '.pdf", bbox_inches='tight')
        fig.savefig('images/' + title + '.jpg', bbox_inches='tight', dpi=400)
    return


tbg = TBG(theta=1.05, a=2.5, L=3.5)
l_trunc = 40
pos_x, pos_y, _ = tbg.position_mapping(l_trunc, l_trunc)
R = l_trunc * np.sqrt(3) * tbg.a/2 #R is the truncated  
V_BM = np.load('data/bm_2.npy')
V_TB = np.load('data/tb_4.npy')

for t in range(1,6):
    plot_vec(pos_x, pos_y, V_TB[t-1,:], R, title=f'TB_run_4_T={t}', save=True)


