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
        self.a1, self.a2 = self.a / 2 * np.array([1, np.sqrt(3)]), self.a / 2 * np.array([-1, np.sqrt(3)])
        self.b1, self.b2 = 4*np.pi/(3*self.d)*np.array([np.sqrt(3) / 2, 1 / 2]), 4*np.pi/(3*self.d)*np.array([-np.sqrt(3)/2,1/2])
        self.tau_A, self.tau_B = np.array([0, 0]), np.array([0, self.d])

        # K points for monolayer
        self.K = 4 * np.pi / (3 * self.a) * np.array([1, 0])
        self.k_d = np.linalg.norm(self.K)

        # K points for bilayer
        self.K_1 = rot(-self.theta/2) @ self.K
        self.K_2 = rot(self.theta/2) @ self.K

        # bilayer lattice, first number is layer
        self.a_11, self.a_12 = rot(-self.theta / 2) @ self.a1, rot(-self.theta / 2) @ self.a2
        self.a_21, self.a_22 = rot(self.theta / 2) @ self.a1, rot(self.theta / 2) @ self.a2
        
        self.A_1 = np.column_stack((self.a_11,self.a_12))
        self.A_2 = np.column_stack((self.a_21,self.a_22))

        self.b_11, self.b_12 = rot(-self.theta / 2) @ self.b1, rot(-self.theta / 2) @ self.b2
        self.b_21, self.b_22 = rot(self.theta / 2) @ self.b1, rot(self.theta / 2) @ self.b2
        
        self.B_1 = np.column_stack((self.b_11,self.b_12))
        self.B_2 = np.column_stack((self.b_21,self.b_22))

        self.tau_1A, self.tau_1B = rot(-self.theta / 2) @ self.tau_A, rot(-self.theta / 2) @ self.tau_B
        self.tau_2A, self.tau_2B = rot(self.theta / 2) @ self.tau_A, rot(self.theta / 2) @ self.tau_B

        # distance between K points of different layers
        self.k_theta = 2*self.k_d*np.sin(self.theta/2)

        # momentum hops
        self.s1 = self.k_theta * np.array([0, -1])
        self.s2 = self.k_theta * np.array([np.sqrt(3)/2, 1/2])
        self.s3 = self.k_theta * np.array([-np.sqrt(3)/2, 1/2])

        # moire unit cell lattice
        self.a_m1 = 4*np.pi/(3*self.k_theta) * np.array([np.sqrt(3)/2, -1/2])
        self.a_m2 = 4*np.pi/(3*self.k_theta) * np.array([np.sqrt(3)/2, 1/2])
        
        self.A_m = np.array([self.a_m1, self.a_m2]).transpose()
        
        self.b_m1 = np.sqrt(3)*self.k_theta * np.array([1/2, -np.sqrt(3)/2])
        self.b_m2 = np.sqrt(3)*self.k_theta * np.array([1/2, np.sqrt(3)/2])

        self.T1 = np.array([[1,1],[1,1]])
        self.T2 = np.exp(1j * np.array([[self.b_12 @ self.tau_1A - self.b_22 @ self.tau_2A, 
                                         self.b_12 @ self.tau_1A - self.b_22 @ self.tau_2B],
                                        [self.b_12 @ self.tau_1B - self.b_22 @ self.tau_2A,
                                         self.b_12 @ self.tau_1B - self.b_22 @ self.tau_2B]]))
        self.T3 = np.exp(1j * np.array([[-self.b_11 @ self.tau_1A + self.b_21 @ self.tau_2A,
                                         -self.b_11 @ self.tau_1A + self.b_21 @ self.tau_2B],
                                        [-self.b_11 @ self.tau_1B + self.b_21 @ self.tau_2A,
                                         -self.b_11 @ self.tau_1B + self.b_21 @ self.tau_2B]]))


    # input: (l_x * l_y) number of cells in a truncated system
    # output: pos_x, pos_y, pos_z: (x,y,z) coordinates of lattice points
    def position_mapping(self, l_x, l_y):
        
        N_x = 2 * l_x + 1  # number of unit cells
        N_y = 2 * l_y + 1
        N = N_x * N_y

        # using meshgrid to define lattice indices
        n_list_x = np.linspace(-l_x, l_x, N_x)
        n_list_y = np.linspace(-l_y, l_y, N_y)

        n_mesh_1, n_mesh_2 = np.meshgrid(n_list_x, n_list_y)
        
        # suppose direction 1 coincides a_x1, direction 2 coincides a_x2
        # the index of m-th entry is (n1[m], n2[m])
        n1 = n_mesh_1.flatten()
        n2 = n_mesh_2.flatten()

        # give the physical position of any index
        pos_x = np.zeros(4 * N)
        pos_y = np.zeros(4 * N)
        pos_z = np.zeros(4 * N)

        for i in range(4 * N):
            if i < N:
                # layer 1 site A
                pos = n1[i] * self.a_11 + n2[i] * self.a_12 + self.tau_1A
                pos_x[i] = pos[0]
                pos_y[i] = pos[1]
                pos_z[i] = self.L

            elif i < 2 * N:
                # layer 1 site B
                j = i % N
                pos = n1[j] * self.a_11 + n2[j] * self.a_12 + self.tau_1B
                pos_x[i] = pos[0]
                pos_y[i] = pos[1]
                pos_z[i] = self.L

            elif i < 3 * N:
                # layer 2 site A
                j = i % N
                pos = n1[j] * self.a_21 + n2[j] * self.a_22 + self.tau_2A
                pos_x[i] = pos[0]
                pos_y[i] = pos[1]
                pos_z[i] = 0

            else:
                # layer 2 site B
                j = i % N
                pos = n1[j] * self.a_21 + n2[j] * self.a_22 + self.tau_2B
                pos_x[i] = pos[0]
                pos_y[i] = pos[1]
                pos_z[i] = 0

        return pos_x, pos_y, pos_z
    

    def map_wavepacket_func(self, f, pos_x, pos_y):
        '''
        map a wave-packet function f = (f1a, f1b, f2a, f2b) to values
        [ f1a(X) * exp(1j*K1*X),
          f1b(X) * exp(1j*K1*X),
          f2a(X) * exp(1j*K2*X).
          f2b(X) * exp(1j*K2*X)]
        '''
        k1x, k1y = self.K_1
        k2x, k2y = self.K_2
        N = pos_x.size
        v = np.zeros(N, dtype='complex')

        for i in range(N):
            x = pos_x[i]
            y = pos_y[i]
            z1a, z1b, z2a, z2b = f(x,y)
            if i < N / 4:
                v[i] = z1a * np.exp((k1x*x+k1y*y)*1j)
            elif i < 2*N / 4:
                v[i] = z1b * np.exp((k1x*x+k1y*y)*1j)
            elif i < 3*N / 4:
                v[i] = z2a * np.exp((k2x*x+k2y*y)*1j)
            else:           
                v[i] = z2b * np.exp((k2x*x+k2y*y)*1j)
        
        return v
