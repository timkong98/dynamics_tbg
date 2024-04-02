import numpy as np

# the 2d rotation matrix
def rot(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

class TBG:
    '''Define the TBG geometry using
    theta: twist angle in degrees, a: lattice constant, L: interlayer distance
    '''
    def __init__(self, theta, a, L):
        self.theta_degree = theta
        self.a = a
        self.L = L
        
        self.theta = self.theta_degree / 180 * np.pi
        self.d = self.a / np.sqrt(3)

        # monolayer lattice and shifts
        a1, a2 = self.a / 2 * np.array([1, np.sqrt(3)]), self.a / 2 * np.array([-1, np.sqrt(3)])
        self.A = np.column_stack((a1, a2))
        self.B = 2*np.pi * np.linalg.inv(self.A).T
        self.tau = np.array([[0,0], [0, self.d]])


        # K points for monolayer
        self.K = self.B @ [1/3, -1/3]
        self.k_d = np.linalg.norm(self.K)

        # K points for bilayer
        self.K_1 = rot(-self.theta/2) @ self.K
        self.K_2 = rot(self.theta/2) @ self.K

        # bilayer lattice, first number is layer
        
        self.A_1 = rot(-self.theta / 2) @ self.A
        self.A_2 = rot(self.theta / 2) @ self.A

        self.B_1 = rot(-self.theta / 2) @ self.B
        self.B_2 = rot(self.theta / 2) @ self.B

        self.tau_1 = rot(-self.theta / 2) @ self.tau
        self.tau_2 = rot(self.theta / 2) @ self.tau

        # distance between K points of different layers
        self.k_theta = 2*self.k_d*np.sin(self.theta/2)

        # momentum hops
        self.s1 = self.k_theta * np.array([0, -1])
        self.s2 = self.k_theta * np.array([np.sqrt(3)/2, 1/2])
        self.s3 = self.k_theta * np.array([-np.sqrt(3)/2, 1/2])

        # moire unit cell lattice
        if theta != 0:
            self.B_m = self.B_1 - self.B_2
            self.A_m = 2*np.pi*np.linalg.inv(self.B_m).T

        
    # define the T matrix for the phase
    def T_matrix(self, G):
        T = np.array([[G @ (self.tau_2[:,i] - self.tau_2[:,j]) for i in range(2)] for j in range(2)])
        return np.exp(-1j * T)
        

    # input: (l_x * l_y) number of cells in a truncated system
    # output: pos_x, pos_y, pos_z: (x,y,z) coordinates of lattice points
    def position_mapping(self, l_x, l_y, flatten=False):
        
        N_x = 2 * l_x + 1  # number of unit cells
        N_y = 2 * l_y + 1
        N = N_x * N_y

        # using meshgrid to define lattice indices
        n_list_x = np.linspace(-l_x, l_x, N_x)
        n_list_y = np.linspace(-l_y, l_y, N_y)

        n_mesh_1, n_mesh_2 = np.meshgrid(n_list_x, n_list_y)
        
        # suppose direction 1 coincides a_x1, direction 2 coincides a_x2
        # the index of m-th entry is (n1[m], n2[m])
        n = np.array([n_mesh_1.flatten(), n_mesh_2.flatten()])
        #n2 = n_mesh_2.flatten()

        # give the physical position of any index
        pos_x = np.zeros((4, N))
        pos_y = np.zeros((4, N))
        pos_z = np.zeros((4, N))

        for index in range(4):
            if index == 0:
                pos = (self.A_1 @ n).T +  self.tau_1[:,0]
                pos_x[index] = pos[:,0]
                pos_y[index] = pos[:,1]
                pos_z[index] = self.L
            
            elif index == 1:
                pos = (self.A_1 @ n).T +  self.tau_1[:,1]
                pos_x[index] = pos[:,0]
                pos_y[index] = pos[:,1]
                pos_z[index] = self.L

            elif index == 2:
                pos = (self.A_2 @ n).T +  self.tau_2[:,0]
                pos_x[index] = pos[:,0]
                pos_y[index] = pos[:,1]
                pos_z[index] = 0

            else:
                pos = (self.A_2 @ n).T +  self.tau_2[:,1]
                pos_x[index] = pos[:,0]
                pos_y[index] = pos[:,1]
                pos_z[index] = 0

        if flatten:
            pos_x = pos_x.flatten()
            pos_y = pos_y.flatten()
            pos_z = pos_z.flatten()

        return pos_x, pos_y, pos_z
    

    

    def map_wavepacket_func(self, f, pos_x, pos_y):
        '''
        map a wave-packet function f = (f1a, f1b, f2a, f2b) to values
        [ f1a(X) * exp(1j*K1*X),
          f1b(X) * exp(1j*K1*X),
          f2a(X) * exp(1j*K2*X).
          f2b(X) * exp(1j*K2*X)]
        '''
        V = np.zeros_like(pos_x, dtype='complex')

        for index in range(4):
            X = pos_x[index]
            Y = pos_y[index]
            z = f(X,Y)[index]
            if index < 2:
                V[index] = z * np.exp(self.K_1 @ np.array([X, Y]) * 1j)
            else:
                V[index] = z * np.exp(self.K_2 @ np.array([X, Y]) * 1j)

        return V
