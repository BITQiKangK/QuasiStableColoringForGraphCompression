import numpy as np
import torch
import scipy
import json
import os
import time
import logging
import itertools

def save_list_to_json(l:list, filename):
    with open(filename, 'w') as f:
        json.dump(l, f)


class ColorStatus:
    def __init__(self, v, n):
        """
        记录当前颜色划分下的图状态
        v 是图中结点的数量
        n 是当前各状态方阵的阶数
        neighbor 维护了各结点与各颜色划分的邻接关系，初始v*n的矩阵，此后每一次细化列数加一，与划分数量保持一致
        upper_base 维护了颜色与颜色之间最大连接数，大小为n*n
        lower_base 维护了颜色与颜色之间最小连接数，大小为n*n
        errors_base 维护了最大最小度之间的差值
        """
        self.v = v
        self.n = n
        self.neighbor = None
        self.upper_base = torch.zeros((n, n), dtype=torch.float32)
        self.lower_base = torch.full((n, n), fill_value=torch.inf, dtype=torch.float32)
        self.errors_base = torch.zeros((n, n), dtype=torch.float32)

    def resize(self, n):
        m = self.n
        self.n = n

        new_upper_base = torch.zeros((n, n), dtype=torch.float32)
        new_lower_base = torch.full((n, n), fill_value=torch.inf, dtype=torch.float32)
        new_errors_base = torch.zeros((n, n), dtype=torch.float32)

        new_upper_base[:m, :m] = self.upper_base
        new_lower_base[:m, :m] = self.lower_base
        new_errors_base[:m, :m] = self.errors_base

        del self.upper_base
        del self.lower_base
        del self.errors_base

        self.upper_base = new_upper_base
        self.lower_base = new_lower_base
        self.errors_base = new_errors_base


class QuasiStableColoring:
    def __init__(self, G, store_root, store=False, directed=True, store_step=100, log_step=10):
        self.G = G
        self.v = G.num_nodes
        self.p = list()
        self.p.append(torch.arange(G.num_nodes, dtype=torch.int32))
        self.BASE_MATRIX_SIZE = 128
        self.q_error = np.inf
        self.store = store
        self.store_root = store_root
        self.directed = directed
        if not os.path.exists(os.path.join(self.store_root)):
            os.makedirs(self.store_root)
        self.log_step = log_step
        self.store_step = store_step

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(self.store_root, 'logfile.log'), mode='w'), 
                                      logging.StreamHandler()])
        self.logger = logging.getLogger('my_logger')

    def split_color(self, color_status, witness_i, witness_j, threshold):
        split = color_status.neighbor[self.p[witness_i], witness_j]
        split_1 = split > threshold

        retain = self.p[witness_i][~split_1]
        eject = self.p[witness_i][split_1]

        assert len(retain) != 0
        assert len(eject) != 0

        self.p[witness_i] = retain
        self.p.append(eject)

    def partition_matrix(self):
        """返回初始结点与颜色对应关系稀疏矩阵"""
        p_matrix = torch.ones(self.v, 1)
        return p_matrix

    def init_status(self, color_status, weights):
        """初始化出度，入度颜色状态"""
        p_matrix = self.partition_matrix()
        color_status.neighbor = torch.mm(weights.to_sparse_coo(), p_matrix)

        m = len(self.p)
        upper_deg = color_status.upper_base[:m, :m]
        lower_deg = color_status.lower_base[:m, :m]
        errors = color_status.errors_base[:m, :m]

        for i, X in enumerate(self.p):
            upper_deg[i, :] = torch.max(color_status.neighbor[X, :], dim=0)[0]
            lower_deg[i, :] = torch.min(color_status.neighbor[X, :], dim=0)[0]

        errors[:, :] = upper_deg - lower_deg

    def update_status(self, color_status, weights, old, new):
        m = len(self.p)
        old_nodes = self.p[old]
        new_nodes = self.p[new]

        # expend neighbor by one column
        new_column = torch.zeros(color_status.neighbor.size(0), 1)
        color_status.neighbor = torch.cat((color_status.neighbor, new_column), dim=1)

        # update columns for old and new color
        old_degs = torch.tensor(weights[:, old_nodes].sum(axis=1))
        new_degs = torch.tensor(weights[:, new_nodes].sum(axis=1))
        color_status.neighbor[:, old] = old_degs.reshape(-1)
        color_status.neighbor[:, new] = new_degs.reshape(-1)
        del old_degs
        del new_degs

        upper_deg = color_status.upper_base[:m, :m]
        lower_deg = color_status.lower_base[:m, :m]
        errors = color_status.errors_base[:m, :m]

        upper_deg[old, :] = torch.max(color_status.neighbor[old_nodes, :], dim=0)[0]
        lower_deg[old, :] = torch.min(color_status.neighbor[old_nodes, :], dim=0)[0]

        upper_deg[new, :] = torch.max(color_status.neighbor[new_nodes, :], dim=0)[0]
        lower_deg[new, :] = torch.min(color_status.neighbor[new_nodes, :], dim=0)[0]

        for i, X in enumerate(self.p):
            upper_deg[i, old] = torch.max(color_status.neighbor[X, old])
            lower_deg[i, old] = torch.min(color_status.neighbor[X, old])

            upper_deg[i, new] = torch.max(color_status.neighbor[X, new])
            lower_deg[i, new] = torch.min(color_status.neighbor[X, new])

        errors[:, :] = upper_deg - lower_deg

    def pick_witness(self, color_status):
        m = len(self.p)
        t = torch.argmax(color_status.errors_base[:m, :m])
        witness = torch.tensor([t // m, t % m])

        q_error = color_status.errors_base[witness[0], witness[1]]

        split_deg = color_status.neighbor[self.p[witness[0]], witness[1]].mean()

        return witness[0], witness[1], split_deg, q_error

    def q_color_directed(self, n_colors=np.Inf, q_errors=0.0):
        time_cost_list = []
        q_error_in_list = []
        q_error_out_list = []
        q_error_list = []
        time1 = time.time()
        weights = self.G.adj_t.to_sparse_csc()
        weights_scipy = scipy.sparse.csc_matrix((weights.values(), weights.row_indices(), weights.ccol_indices()), shape=(self.v, self.v))
        weights_transpose = weights.t().to_sparse_csc()
        weights_transpose_scipy = weights_scipy.transpose().tocsc()

        color_status_out = ColorStatus(self.v, int(min(n_colors, self.BASE_MATRIX_SIZE)))
        color_status_in = ColorStatus(self.v, int(min(n_colors, self.BASE_MATRIX_SIZE)))
        self.init_status(color_status_out, weights)
        self.init_status(color_status_in, weights_transpose)
        del weights
        del weights_transpose
        time2 = time.time()
        self.logger.info(f'Start coloring, time cost: {time2-time1:.2f} seconds.')
        pre_time = time2

        while len(self.p) <= n_colors:
            if len(self.p) == color_status_in.n:
                color_status_in.resize(color_status_in.n * 2)
                color_status_out.resize(color_status_out.n * 2)

            witness_in_i, witness_in_j, split_deg_in, q_error_in = self.pick_witness(color_status_in)
            witness_out_i, witness_out_j, split_deg_out, q_error_out = self.pick_witness(color_status_out)

            if len(self.p) % self.log_step == 0:
                time3 = time.time()
                time_cost = round(time3 - pre_time, 2)
                q_error = max(q_error_in, q_error_out)
                self.logger.info(f"{len(self.p)} colors with {q_error} error, time_cost {time_cost} seconds.")
                time_cost_list.append(time_cost)
                q_error_in_list.append(q_error_in.item())
                q_error_out_list.append(q_error_out.item())
                q_error_list.append(q_error.item())
                pre_time = time.time()

            if len(self.p) % self.store_step == 0: 
                if self.store:
                    file_name = os.path.join(self.store_root, f"{len(self.p):04d}" + ".json")
                    converted_list = []
                    converted_list.append([t.tolist() for t in self.p])
                    adj_t = torch.zeros((len(self.p), len(self.p)), dtype=torch.float32)
                    neighbor = color_status_out.neighbor
                    for i, p in enumerate(self.p):
                        adj_t[i] = torch.sum(neighbor[p], dim=0)
                    
                    converted_list.append([t.tolist() for t in adj_t])
                    cost_time = sum(time_cost_list)
                    converted_list.append(cost_time)

                    save_list_to_json(converted_list, file_name)
                    pre_time = time.time()
                    
            if q_error_in <= q_errors and q_error_out <= q_errors:
                break

            if q_error_out >= q_error_in:
                self.split_color(color_status_out, witness_out_i, witness_out_j, split_deg_out)
                self.update_status(color_status_out, weights_scipy, witness_out_i, len(self.p) - 1)
                self.update_status(color_status_in, weights_transpose_scipy, witness_out_i, len(self.p) - 1)
            else:
                self.split_color(color_status_in, witness_in_i, witness_in_j, split_deg_in)
                self.update_status(color_status_out, weights_scipy, witness_in_i, len(self.p) - 1)
                self.update_status(color_status_in, weights_transpose_scipy, witness_in_i, len(self.p) - 1)
            

        _, _, _, q_errors_out = self.pick_witness(color_status_out)
        _, _, _, q_errors_in = self.pick_witness(color_status_in)
        self.q_error = max(q_errors_out, q_errors_in)
        self.logger.info(f"refined and got {len(self.p)} colors with {self.q_error} q-error")
        file_name = os.path.join(self.store_root, "time_cost_per_10_colors.json")
        save_list_to_json(time_cost_list, file_name)
        file_name = os.path.join(self.store_root, 'time_cost_over_all.json')
        save_list_to_json(list(itertools.accumulate(time_cost_list)), file_name)
        file_name = os.path.join(self.store_root, f"q_error_in" + ".json")
        save_list_to_json(q_error_in_list, file_name)
        file_name = os.path.join(self.store_root, f"q_error_out" + ".json")
        save_list_to_json(q_error_out_list, file_name)
        file_name = os.path.join(self.store_root, f"q_error" + ".json")
        save_list_to_json(q_error_list, file_name)

    def q_color_undirected(self, n_colors=np.Inf, q_errors=0.0):
        time_cost_list = []
        q_error_list = []
        time1 = time.time()
        weights = self.G.adj_t.to_sparse_csc()
        weights_scipy = scipy.sparse.csc_matrix((weights.values(), weights.row_indices(), weights.ccol_indices()), shape=(self.v, self.v))

        color_status = ColorStatus(self.v, int(min(n_colors, self.BASE_MATRIX_SIZE)))
        self.init_status(color_status, weights)
        del weights
        time2 = time.time()
        self.logger.info(f'Start coloring, time cost: {time2-time1:.2f} seconds.')

        pre_time = time2
        while len(self.p) <= n_colors:
            if len(self.p) == color_status.n:
                color_status.resize(color_status.n * 2)

            witness_i, witness_j, split_deg, q_error = self.pick_witness(color_status)

            if len(self.p) % self.log_step == 0:
                time3 = time.time()
                time_cost = round(time3 - pre_time, 2)
                self.logger.info(f"{len(self.p)} colors with {q_error} error, time_cost {time_cost} seconds.")
                time_cost_list.append(time_cost)
                q_error_list.append(q_error.item())
                pre_time = time.time()

            if len(self.p) % self.store_step == 0: 
                if self.store:
                    file_name = os.path.join(self.store_root, f"{len(self.p):04d}.json")
                    converted_list = []
                    converted_list.append([t.tolist() for t in self.p])
                    neighbor = color_status.neighbor
                    adj_t = torch.zeros(len(self.p), len(self.p), dtype=torch.float32)
                    for i, p in enumerate(self.p):
                        adj_t[i] = torch.sum(neighbor[p], dim=0)
                    converted_list.append([t.tolist() for t in adj_t])
                    cost_time = sum(time_cost_list)
                    converted_list.append(cost_time)
                    save_list_to_json(converted_list, file_name)
                    pre_time = time.time()
      
            if q_error <= q_errors:
                break

            self.split_color(color_status, witness_i, witness_j, split_deg)
            self.update_status(color_status, weights_scipy, witness_i, len(self.p) - 1)

        _, _, _, q_error = self.pick_witness(color_status)
        self.q_error = q_error
        self.logger.info(f"refined and got {len(self.p)} colors with {self.q_error} q-error")
        file_name = os.path.join(self.store_root, "time_cost_per_10_colors.json")
        save_list_to_json(time_cost_list, file_name)
        file_name = os.path.join(self.store_root, 'time_cost_over_all.json')
        save_list_to_json(list(itertools.accumulate(time_cost_list)), file_name)
        file_name = os.path.join(self.store_root, f"q_error" + ".json")
        save_list_to_json(q_error_list, file_name)

    def q_color(self, n_colors=np.Inf, q_errors=0.0):
        self.logger.info(f"Start logging, store root {self.store_root}, store colors: {self.store}, directed: {self.directed}")
        if self.directed:
            self.q_color_directed(n_colors, q_errors)
        else:
            self.q_color_undirected(n_colors, q_errors)