from Lee import SE3, se3
import numpy as np
# from scipy.linalg import cho_factor, cho_solve
import scipy.sparse as sp
# from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky

#todo
#make general for se2 as well (needs general base se2 class or separate se2 class)
# in the Solver init make a check for SE2 SE3 in both gt and real dataset
#make sparse H and solvable with spsolve or cholmod (done)
#do some minor efficiency cleanups
#add LM version of solver
#implement base solver class that the GN or LM solver inherits (done for GN)

class Solver():
    def __init__(self, g2o_dataset_filepath, g2o_gt_filepath=""):
        self.edge_list = []
        self.pose_list = []

        self.edge_gt_list = []
        self.pose_gt_list = []

        with open(g2o_dataset_filepath, "r") as f:
            for line in f:
                self.g2o_data_parser(line, self.pose_list, self.edge_list)
        if g2o_gt_filepath:
            with open(g2o_gt_filepath, "r") as f:
                for line in f:
                    self.g2o_data_parser(line, self.pose_gt_list, self.edge_gt_list)
        
    def g2o_data_parser(self, line, pose_list, edge_list):
        tok = line.split()
        if line.startswith("EDGE_SE3:QUAT"):    
            if len(tok) != 31:
                raise ValueError(f"EDGE_SE3:QUAT needs 31 tokens, got {len(tok)}: {line.strip()[:60]}")
            i, j = int(tok[1]), int(tok[2])
            vals = np.array(tok[3:], dtype=float)

            trans = vals[0:3]
            quat = vals[3:7] / np.linalg.norm(vals[3:7])
            Z = SE3(tuple(quat), trans)

            I = np.zeros((6, 6))
            I[np.triu_indices(6)] = vals[7:]
            omega_g2o = I + np.triu(I, 1).T

            S = np.diag([1, 1, 1, 0.5, 0.5, 0.5])
            omega = S @ omega_g2o @ S

            edge_list.append(dict(i=i, j=j, relative_measurement=Z, omega=omega))

        if line.startswith("VERTEX_SE3:QUAT"):
            if len(tok) != 9:
                raise ValueError(f"VERTEX_SE3:QUAT needs 9 tokens, got {len(tok)}: {line.strip()[:60]}")
            id = int(tok[1])
            vals = np.array(tok[2:], dtype=float)
            trans = vals[0:3]
            quat = vals[3:7] / np.linalg.norm(vals[3:7])
            X = SE3(tuple(quat), trans)

            if id >= len(pose_list):
                pose_list.extend([None] * (id + 1 - len(pose_list))) #makes it longer by 1 + id 
            if pose_list[id] is not None:
                raise ValueError(f"duplicate vertex id {id}")
            pose_list[id] = X


class GNSolver(Solver): #Gauss-Newton Solver which will inherit from Solver base class
    def solve(self):
        d = 6 # hardcoded dimension, should be set in init method
        N = len(self.pose_list)
        loop_num = 20
        new_final_pose_list = self.pose_list.copy()

        prev_cost = None
        factor = None

        for loop in range(loop_num):
            #main loop
            # H = np.zeros((d * N, d * N))
            rows, cols, vals = [], [], []
            b = np.zeros(d * N)
            cost = 0.0
            
            for edge in self.edge_list:
                i, j = edge["i"], edge["j"]
                X_i, X_j = new_final_pose_list[i], new_final_pose_list[j] #returns SE3
                Z_ij = edge["relative_measurement"] #returns SE3
                omega = edge["omega"]
                
                T = X_i.inverse_pose() @ X_j
                E = Z_ij.inverse_pose() @ T
                e_ij = E.log_map() #this returns the se3 object, not the vector itself, that would be e_ij.vee
                e_ij_vector = e_ij.vee
            
                Jr, Jr_inv = e_ij.compute_left_jacobians(right=True)
            
                # A_ij = -Jr_inv @ T.inverse_pose().Ad()
                A_ij = -Jr_inv @ T.Ad_inv()
                B_ij = Jr_inv
            
                # H[i * d : i * d + d, i * d : i * d + d] += A_ij.T @ omega @ A_ij
                # H[i * d : i * d + d, j * d : j * d + d] += A_ij.T @ omega @ B_ij
                # H[j * d : j * d + d, i * d : i * d + d] += B_ij.T @ omega @ A_ij
                # H[j * d : j * d + d, j * d : j * d + d] += B_ij.T @ omega @ B_ij 
                
                # this makes it sparse fr
                self.add_block(rows, cols, vals, i, i, A_ij.T @ omega @ A_ij, d)
                self.add_block(rows, cols, vals, i, j, A_ij.T @ omega @ B_ij, d)
                self.add_block(rows, cols, vals, j, i, B_ij.T @ omega @ A_ij, d)
                self.add_block(rows, cols, vals, j, j, B_ij.T @ omega @ B_ij, d)
                
                b[i * d: i * d + d] += A_ij.T @ omega @ e_ij_vector
                b[j * d: j * d + d] += B_ij.T @ omega @ e_ij_vector

                cost += e_ij_vector @ omega @ e_ij_vector

            #this is a stop case
            if prev_cost is not None:
                rel = (prev_cost - cost) / prev_cost
                print(f"iter {loop}: cost {cost:.6e}  rel. decrease {rel:.2e}")
                if rel < -1e-9:
                    print("cost increased, stopping")
                    break
                if rel < 1e-6:
                    break
            prev_cost = cost
            
            # c, low = cho_factor(H[d:, d:]) #anchor for gauge freedom

            H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(d * N, d * N)).tocsc()
            delx = np.zeros(d * N)
            # # delx[d:] = cho_solve((c, low), -b[d:])
            # delx[d:] = spsolve(H[d:, d:], -b[d:]) #anchor for gauge freedom

            H_ff = H[d:, d:].tocsc() #anchor for gauge freedom
            if factor is None:
                factor = cholesky(H_ff)
            else:
                factor.cholesky_inplace(H_ff)

            delx[d:] = factor(-b[d:])

            for n in range(N):
                delta_x = delx[n * d : n * d + d]
                new_final_pose_list[n] = new_final_pose_list[n].boxplus(delta_x)

        return new_final_pose_list
        
    def add_block(self, rows, cols, vals, r, c, block, d):
        rr, cc = np.meshgrid(np.arange(r * d, r * d + d), np.arange(c * d, c * d + d), indexing="ij")
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(block.ravel())
            