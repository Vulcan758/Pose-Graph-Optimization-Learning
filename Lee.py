import numpy as np

def to_hat(vector): #converts w to w^
    if vector.shape[0] == 3: #direct for low dim stuff
        x, y, z = vector
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    m = vector.shape[0]
    n = int((1 + np.sqrt(1 + 8 * m)) / 2) # invert m = n(n-1)/2 --> this is the n value in n by n of matrix
        
    if n * (n - 1) // 2 != m: # this expression outputs the vector shape required, (e.g. 3x3 --> 3, 4x4 --> 6, 5x5 --> 10, ..)
        raise ValueError(f"length {m} is not n(n-1)/2 for any integer n")

    r, c = np.triu_indices(n, k=1)

    sign_mask = (-1.0) ** (r + c)

    skew = np.zeros((n, n), dtype=vector.dtype)        
    skew[r, c] = vector[::-1] * sign_mask
    return skew - skew.T
    
def to_vee(hat_vec):
    if hat_vec.shape == (3, 3): #direct 
        return np.array([hat_vec[2, 1], hat_vec[0, 2], hat_vec[1, 0]])
    
    is_square(hat_vec)
    r_ = hat_vec.shape[0]
    
    #this part does the reverse of to_hat
    r, c = np.triu_indices(r_, k=1)
    vector = hat_vec[r, c]
    sign_mask = (-1.0) ** (c + r)
    vector = (sign_mask * vector)[::-1]
    
    return vector

def is_square(matrix):
    r_, c_ = matrix.shape
    if r_ != c_:
        raise ValueError("Matrix is not square")
    return True
    
def is_valid_hat(hat_vec):
    is_square(hat_vec)
    n = hat_vec.shape[0]
    return np.allclose(hat_vec + hat_vec.T, np.zeros((n, n))) #skew symmetric ma|trices summed with its transpose is supposed to equal 0

class SO_n():
    def __init__(self, R: np.ndarray, verbose = False, validate=True):
        self.verbose = verbose
        is_square(R)
        r = R.shape[0]

        if r > 3 and self.verbose:
            print("Exp and log map will not work for this dimenison")
        
        if validate and (not np.allclose(R @ R.T, np.eye(r)) or not np.isclose(np.linalg.det(R), 1.0)):
            raise ValueError("Invalid rotation matrix")
        if self.verbose:
            print(f"Rotation matrix is valid and is in SO({r})")
        self.R, self.dims = R, r
        self.threshold = 0.001

    def inverse(self):
        return self.R.T

    def log_map(self):
        if self.dims == 2:
            return so_n(np.array([np.arctan2(self.R[1, 0], self.R[0, 0])]))

        theta = np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1.0, 1.0))
        v = to_vee(self.R - self.inverse())
        
        if theta < self.threshold:
            phi = 0.5 * (1 + ((theta ** 2) / 6)) * v
        
        elif np.pi - theta < self.threshold:
            B = ((self.R + self.R.T) / 2 - np.cos(theta) * np.eye(3)) / (1 - np.cos(theta))  # = a a^T
            k = np.argmax(np.diag(B))
            a = B[:, k] / np.sqrt(B[k, k])
            if a @ v < 0:      # v = 2 sin(theta) a, so a must point the same way as v
                a = -a
            phi = theta * a
            
        else:
            phi = (theta / (2 * np.sin(theta))) * v
        
        return so_n(phi) 

class so_n():
    def __init__(self, phi: np.ndarray, verbose = False):
        self.verbose = verbose
        self.m = phi.shape[0]
        self.n = int((1 + np.sqrt(1 + 8 * self.m)) / 2) # invert m = n(n-1)/2 --> this is the n value in n by n of matrix
        self.threshold = 0.001

        if (self.n > 3 or self.m > 6) and self.verbose:
            print("Exp and log map will not work for this dimensions")
        if phi.ndim != 1:
            raise ValueError("Invalid vector shape")
        self.hat = self.to_hat(phi)
        self.vee = phi #this is the vector
        if self.verbose:
            print(self.hat) #this is the skew

        self.theta = np.linalg.norm(self.vee)

    def is_valid_hat(self, hat_vec):
        is_square(hat_vec)
        n = hat_vec.shape[0]
        return np.allclose(hat_vec + hat_vec.T, np.zeros((n, n))) #skew symmetric matrices summed with its transpose is supposed to equal 0

    def compute_left_jacobians(self, right = False):
        #this function was only made so that right jacobian can be computed easily rather than hardcoding in the init method
        vector = self.vee
        if right:
            vector = -self.vee
        W = self.to_hat(vector)
        theta = np.linalg.norm(vector)
        n = W.shape[0]
        
        if theta < self.threshold:
            c_term, s_term = (1/2) - (((theta) ** 2) / 24), (1/6) - (((theta) ** 2) / 120)
            inverse_cs_term = (1/12) - (((theta) ** 2) / 720)
        else:
            c_term, s_term = ((1 - np.cos(theta)) / theta ** 2), ((theta - np.sin(theta)) / theta ** 3)
            inverse_cs_term = (1 / theta ** 2) - ((1 + np.cos(theta)) / (2 * theta * np.sin(theta)))
        
        I = np.eye(n)
        
        Jl = I + c_term * W + s_term * (W @ W)
        Jl_inv = I - 0.5 * W + inverse_cs_term * (W @ W)

        return Jl, Jl_inv

    def exp_map(self):
        n = self.hat.shape[0]
        if self.theta < self.threshold:
            c_term, s_term = 0.5 - ((self.theta ** 2) / 24), 1 - ((self.theta ** 2) / 6)
        else:
            c_term, s_term = ((1 - np.cos(self.theta)) / self.theta ** 2), ((np.sin(self.theta) / self.theta))
        
        I = np.eye(n)
        
        R = I + (s_term * self.hat) + (c_term * (self.hat @ self.hat)) #Rodrigues formula
        
        # if n > 4:
        #     from scipy.linalg import expm
        #     R = expm(hat_vec)
        #     if n > 50:
        #         from scipy.sparse.linalg import expm
        #         R = expm(hat_vec)
     
        return SO_n(R, validate=False)

    def to_hat(self, vector): #converts w to w^
        return to_hat(vector)
        
    def to_vee(self, hat_vec):
        return to_vee(hat_vec)

class se3():
    def __init__(self, twist):
        self.vee = twist #[rho, phi]
        self.phi = so_n(twist[3:]) #angular velocities
        self.rho = twist[:3] #translational velocities
        self.threshold = 0.001
        
        self.hat = np.zeros((4, 4))
        self.hat[:3, :3] = self.phi.hat
        self.hat[:3, 3] = self.rho
        
    def exp_map(self):
        R = self.phi.exp_map() #returns SO_n()
        Jl_phi, Jl_inv_phi = self.phi.compute_left_jacobians()
        t = Jl_phi @ self.rho
        return SE3(R, t)

    def ad(self):
        return np.block([[self.phi.hat, to_hat(self.rho)], #solver doesnt use this so i wont be changing it
                         [np.zeros((3, 3)), self.phi.hat]])
        
    def compute_left_jacobians(self, right=False):
        # first compute the Q matrix
        twist = self.vee
        if right:
            twist = -self.vee
        phi = so_n(twist[3:]) if right else self.phi
        rho = twist[:3]
        P = to_hat(rho)
        S = phi.hat
        theta = phi.theta

        if theta < self.threshold:
            c1 = (1/6) - ((theta ** 2)/120)
            c2 = (1/24) - ((theta ** 2)/720)
            c3 = (1/120) - ((theta ** 2)/2520)
        else:
            c1 = (theta - np.sin(theta))/(theta ** 3)
            c2 = ((theta ** 2) + 2 * np.cos(theta) - 2)/(2 * (theta ** 4))
            c3 = (2 * theta - 3 * np.sin(theta) + theta * np.cos(theta))/(2 * (theta ** 5))

        Q = 0.5 * P + c1 * (S @ P + P @ S + S @ P @ S) + c2 * (S @ S @ P + P @ S @ S - 3 * (S @ P @ S)) + c3 * (S @ P @ S @ S + S @ S @ P @ S)
        # end Q matrix compute
    
        J, J_inv = phi.compute_left_jacobians()

        Jl = np.zeros((6, 6));     Jl[:3, :3] = J;         Jl[:3, 3:] = Q;                      Jl[3:, 3:] = J
        Jl_inv = np.zeros((6, 6)); Jl_inv[:3, :3] = J_inv; Jl_inv[:3, 3:] = -J_inv @ Q @ J_inv; Jl_inv[3:, 3:] = J_inv

        return Jl, Jl_inv

        
class SE3():
    def __init__(self, rotation, translation):
        if isinstance(rotation, SO_n):
            self.rotation = rotation
        else:
            self.rotation = self.quat_to_rot(rotation)   # this is quat (x, y, z, w)        
            
        self.t_flat = translation #flat vector form
        self.R = self.rotation.R
        self.t = self.t_flat.reshape(3, 1) #column vector form (flat vector transposed
        
        self.X = np.eye(4)
        self.X[:3, :3] = self.R
        self.X[:3, 3] = self.t_flat

    def __matmul__(self, other):
        if isinstance(other, SE3): #for pose on pose
            R = self.R @ other.R
            t = self.R @ other.t_flat + self.t_flat
            return SE3(SO_n(R, validate=False), t)
        if isinstance(other, np.ndarray): #for pose on point
            if other.shape == (3,): 
                return self.R @ other + self.t_flat
            if other.ndim == 2 and other.shape[0] == 3:  
                return self.R @ other + self.t           
        return NotImplemented
        
    def log_map(self):
        phi = self.rotation.log_map()
        Jl_phi, Jl_inv_phi = phi.compute_left_jacobians()
        rho = Jl_inv_phi @ self.t_flat
        twist = np.concatenate([rho, phi.vee])
        return se3(twist)

    def Log(self):
        twist = self.log_map()
        return twist.vee

    def boxplus(self, delta):
        return self @ se3(delta).exp_map()

    def boxminus(self, other):
        D = other.inverse() @ self.X
        return SE3(SO_n(D[:3, :3], validate=False), D[:3, 3]).Log()

    def inverse(self):
        Rt = self.R.T
        Xi = np.eye(4)
        Xi[:3, :3] = Rt
        Xi[:3, 3] = -Rt @ self.t_flat
        return Xi

    def inverse_pose(self):
        Rt = self.rotation.inverse()
        return SE3(SO_n(Rt, validate=False), -Rt @ self.t_flat)
        
    def Ad(self):
        A = np.zeros((6, 6))
        A[:3, :3] = self.R
        A[:3, 3:] = to_hat(self.t_flat) @ self.R
        A[3:, 3:] = self.R
        return A

    def Ad_inv(self):
        Rt = self.R.T
        A = np.zeros((6, 6))
        A[:3, :3] = Rt
        A[:3, 3:] = -Rt @ to_hat(self.t_flat)
        A[3:, 3:] = Rt
        return A
    
    def quat_to_rot(self, q):
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("zero quaternion")
        q = np.asarray(q, dtype=float)
        q = q / n
        x, y, z, w = q
        R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                      [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                      [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])
        return SO_n(R, validate=False)


class se2():
    def __init__(self, twist):
        self.vee = twist
        self.phi = so_n(twist[2:])
        self.rho = twist[:2]
        self.threshold = 0.001

        self.hat = np.zeros((3, 3))
        self.hat[:2, :2] = self.phi.hat
        self.hat[:2, 2] = self.rho

    def exp_map(self):
        R = self.phi.exp_map()
        Jl_phi, Jl_inv_phi = self.phi.compute_left_jacobians()
        t = Jl_phi @ self.rho
        return SE2(R, t)

    def ad(self):
        A = np.zeros((3, 3))
        A[:2, :2] = self.phi.hat
        A[0, 2] = self.rho[1]
        A[1, 2] = -self.rho[0]
        return A

    def compute_left_jacobians(self, right=False):
        twist = self.vee
        if right:
            twist = -self.vee
        x, y, theta = twist

        if abs(theta) < self.threshold:
            a = 1 - (theta ** 2) / 6
            b = theta / 2 - (theta ** 3) / 24
            q0 = y / 2 + x * theta / 6 - y * (theta ** 2) / 24
            q1 = -x / 2 + y * theta / 6 + x * (theta ** 2) / 24
            a_inv = 1 - (theta ** 2) / 12
        else:
            s, c = np.sin(theta), np.cos(theta)
            a = s / theta
            b = (1 - c) / theta
            q0 = (theta * x + y * (1 - c) - x * s) / (theta ** 2)
            q1 = (theta * y - y * s - x * (1 - c)) / (theta ** 2)
            a_inv = theta * s / (2 * (1 - c))

        V = np.array([[a, -b], [b, a]])
        V_inv = np.array([[a_inv, theta / 2], [-theta / 2, a_inv]])
        q = np.array([q0, q1])

        Jl = np.eye(3)
        Jl[:2, :2] = V
        Jl[:2, 2] = q

        Jl_inv = np.eye(3)
        Jl_inv[:2, :2] = V_inv
        Jl_inv[:2, 2] = -V_inv @ q

        return Jl, Jl_inv


class SE2():
    def __init__(self, rotation, translation):
        if isinstance(rotation, SO_n):
            self.rotation = rotation
        else:
            self.rotation = self.angle_to_rot(rotation)

        self.t_flat = np.asarray(translation, dtype=float)
        self.R = self.rotation.R
        self.t = self.t_flat.reshape(2, 1)
        self.theta = np.arctan2(self.R[1, 0], self.R[0, 0])

        self.X = np.eye(3)
        self.X[:2, :2] = self.R
        self.X[:2, 2] = self.t_flat

    def __matmul__(self, other):
        if isinstance(other, SE2):
            R = self.R @ other.R
            t = self.R @ other.t_flat + self.t_flat
            return SE2(SO_n(R, validate=False), t)
        if isinstance(other, np.ndarray):
            if other.shape == (2,):
                return self.R @ other + self.t_flat
            if other.ndim == 2 and other.shape[0] == 2:
                return self.R @ other + self.t
        return NotImplemented

    def log_map(self):
        phi = self.rotation.log_map()
        Jl_phi, Jl_inv_phi = phi.compute_left_jacobians()
        rho = Jl_inv_phi @ self.t_flat
        twist = np.concatenate([rho, phi.vee])
        return se2(twist)

    def Log(self):
        twist = self.log_map()
        return twist.vee

    def boxplus(self, delta):
        return self @ se2(delta).exp_map()

    def boxminus(self, other):
        return (other.inverse_pose() @ self).Log()

    def inverse(self):
        Rt = self.R.T
        Xi = np.eye(3)
        Xi[:2, :2] = Rt
        Xi[:2, 2] = -Rt @ self.t_flat
        return Xi

    def inverse_pose(self):
        Rt = self.R.T
        return SE2(SO_n(Rt, validate=False), -Rt @ self.t_flat)

    def Ad(self):
        A = np.eye(3)
        A[:2, :2] = self.R
        A[0, 2] = self.t_flat[1]
        A[1, 2] = -self.t_flat[0]
        return A

    def Ad_inv(self):
        return self.inverse_pose().Ad()

    def angle_to_rot(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return SO_n(R, validate=False)
