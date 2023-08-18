import numpy as np
import TBG_lattice
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from cycler import cycler

# plot the vectors into 2 plots
def plot_vec(pos_x, pos_y, V, R, title, save=True):

    line_cycler = (cycler(color=["#e64b35","#4dbbd5","#00a087","#3c5488","#f39b7f",
                                "#8491b4","#91d1c2","#dc0000","#ff6148","#b09c85"]) +
                    cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]))

    marker_cycler = (cycler(color=["#e64b35","#4dbbd5","#00a087","#3c5488","#f39b7f",
                                "#8491b4","#b09c85","#dc0000"]) +
                    cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-.", ":"]) +
                    cycler(marker=["x", "+", ".", "x", "+", ".", "x", "+"]))

    plt.rc("axes", prop_cycle=marker_cycler)
    plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.75)
    plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)
    plt.rcParams['text.usetex'] = True
    plt.rc('text.latex', preamble=r'\usepackage{bm}')
    plt.rcParams.update({'font.size': 15})
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    fig, ((ax1, ax2)) = plt.subplots(1,2, figsize=(15,8))
    value = np.abs(V)
    value = value / np.max(value)
    cm1 = LinearSegmentedColormap.from_list('', ['white', "#4dbbd5"])
    N = pos_x.size / 4
    cf1 = ax1.scatter(pos_x[:2*N], pos_y[:2*N], c=value[:2*N], vmin=0, vmax=1, s=10, cmap=cm1)
    cf2 = ax2.scatter(pos_x[2*N:], pos_y[2*N:], c=value[2*N:], vmin=0, vmax=1, s=10, cmap=cm1)
    
    ax1.set_xlim(-R,R)
    ax1.set_ylim(-R,R)
    ax2.set_xlim(-R,R)
    ax2.set_ylim(-R,R)
    fig.colorbar(cf1, ax=ax1)
    fig.colorbar(cf2, ax=ax2)
    ax1.set_title('Layer 1')
    ax2.set_title('Layer 2')
    ax1.set_aspect(1)
    ax2.set_aspect(1)
    plt.show()
    if save:
        plt.savefig("images/' + title + '.pdf", bbox_inches='tight')
        plt.savefig("images/' + title + '.jpg", bbox_inches='tight', dpi=400)
    return


tbg = TBG_lattice(theta=1.05, a=2.5, L=3.5)
l_trunc = 40
pos_x, pos_y, _ = tbg.position_mapping(l_trunc, l_trunc)
R = l_trunc * np.sqrt(3) * tbg.a/2 #R is the truncated  
V_BM = np.load('data/bm_1.npy')
V_TB = np.load('data/tb_1.npy')

for t in range(6):
    plot_vec(pos_x, pos_y, V_BM[t:], R, title=f'bm_1_{t}', save=False)


