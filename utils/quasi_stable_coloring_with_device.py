import numpy as np
import torch
import scipy


class ColorStatus:
    def __init__(self, v, n, device):
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
        self.device = device
        self.neighbor = None
        self.upper_base = torch.zeros((n, n), dtype=torch.float32, device=device)
        self.lower_base = torch.full((n, n), fill_value=torch.inf, dtype=torch.float32, device=device)
        self.errors_base = torch.zeros((n, n), dtype=torch.float32, device=device)

    def resize(self, n):
        m = self.n
        self.n = n

        new_upper_base = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        new_lower_base = torch.full((n, n), fill_value=torch.inf, dtype=torch.float32, device=self.device)
        new_errors_base = torch.zeros((n, n), dtype=torch.float32, device=self.device)

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
    def __init__(self, G, device):
        self.G = G
        self.v = G.num_nodes
        self.p = list()
        self.p.append(torch.arange(G.num_nodes, dtype=torch.int32, device=device))
        self.BASE_MATRIX_SIZE = 128
        self.q_error = np.inf
        self.device = device

    def split_color(self, color_status, witness_i, witness_j, threshold):
        split = color_status.neighbor[self.p[witness_i].cpu(), witness_j]
        split = split > threshold
        split = split.to(self.device)

        retain = self.p[witness_i][~split]
        eject = self.p[witness_i][split]

        assert len(retain) != 0
        assert len(eject) != 0

        self.p[witness_i] = retain
        self.p.append(eject)

    def partition_matrix(self):
        """返回初始结点与颜色对应关系稀疏矩阵"""
        I = torch.arange(self.v, dtype=torch.int32, device=self.device)
        J = torch.zeros(self.v, dtype=torch.int32, device=self.device)
        V = torch.ones(self.v, dtype=torch.float32, device=self.device)
        p_sparse = torch.sparse_coo_tensor(torch.stack((I, J)), V, size=(self.v, len(self.p)), dtype=torch.float32)
        return p_sparse

    def init_status(self, color_status, weights):
        """初始化出度，入度颜色状态"""
        p_sparse = self.partition_matrix()
        color_status.neighbor = torch.mm(weights.to_sparse_coo(), p_sparse.to_dense()).to("cpu")

        m = len(self.p)
        upper_deg = color_status.upper_base[:m, :m]
        lower_deg = color_status.lower_base[:m, :m]
        errors = color_status.errors_base[:m, :m]

        for i, X in enumerate(self.p):
            t = X.cpu()
            upper_deg[i, :] = torch.max(color_status.neighbor[t, :], dim=0)[0].to(self.device)
            lower_deg[i, :] = torch.min(color_status.neighbor[t, :], dim=0)[0].to(self.device)

        errors[:, :] = upper_deg - lower_deg

    def update_status(self, color_status, weights, old, new):
        m = len(self.p)
        old_nodes = self.p[old].cpu()
        new_nodes = self.p[new].cpu()

        # expend neighbor by one column
        new_column = torch.zeros(color_status.neighbor.size(0), 1, device="cpu")
        color_status.neighbor = torch.cat((color_status.neighbor, new_column), dim=1)

        # update columns for old and new color
        old_degs = torch.tensor(weights[:, old_nodes].sum(axis=1), device="cpu")
        new_degs = torch.tensor(weights[:, new_nodes].sum(axis=1), device="cpu")
        color_status.neighbor[:, old] = old_degs.reshape(-1)
        color_status.neighbor[:, new] = new_degs.reshape(-1)
        del old_degs
        del new_degs

        upper_deg = color_status.upper_base[:m, :m]
        lower_deg = color_status.lower_base[:m, :m]
        errors = color_status.errors_base[:m, :m]

        upper_deg[old, :] = torch.max(color_status.neighbor[old_nodes, :], dim=0)[0].to(self.device)
        lower_deg[old, :] = torch.min(color_status.neighbor[old_nodes, :], dim=0)[0].to(self.device)

        upper_deg[new, :] = torch.max(color_status.neighbor[new_nodes, :], dim=0)[0].to(self.device)
        lower_deg[new, :] = torch.min(color_status.neighbor[new_nodes, :], dim=0)[0].to(self.device)

        for i, X in enumerate(self.p):
            X_cpu = X.cpu()
            upper_deg[i, old] = torch.max(color_status.neighbor[X_cpu, old]).to(self.device)
            lower_deg[i, old] = torch.min(color_status.neighbor[X_cpu, old]).to(self.device)

            upper_deg[i, new] = torch.max(color_status.neighbor[X_cpu, new]).to(self.device)
            lower_deg[i, new] = torch.min(color_status.neighbor[X_cpu, new]).to(self.device)
            del X_cpu

        errors[:, :] = upper_deg - lower_deg


    def pick_witness(self, color_status):
        m = len(self.p)
        t = torch.argmax(color_status.errors_base[:m, :m])
        witness = torch.tensor([t // m, t % m], device=self.device)

        q_error = color_status.errors_base[witness[0], witness[1]]

        t = color_status.neighbor[self.p[witness[0]].cpu(), witness[1]]
        split_deg = t.mean()

        return witness[0], witness[1], split_deg, q_error


    def q_color(self, n_colors=np.Inf, q_errors=0.0):
        weights = self.G.adj_t.to_sparse_csc()
        weights_gpu = weights.to(self.device)
        weights_scipy = scipy.sparse.csc_matrix((weights.values(), weights.row_indices(), weights.ccol_indices()), shape=(self.v, self.v))
        weights_gpu_transpose = weights_gpu.t().to_sparse_csc()
        weights_transpose_scipy = weights_scipy.transpose().tocsc()

        color_status_out = ColorStatus(self.v, int(min(n_colors, self.BASE_MATRIX_SIZE)), self.device)
        color_status_in = ColorStatus(self.v, int(min(n_colors, self.BASE_MATRIX_SIZE)), self.device)
        self.init_status(color_status_out, weights_gpu)
        self.init_status(color_status_in, weights_gpu_transpose)
        del weights
        del weights_gpu
        del weights_gpu_transpose

        while len(self.p) < n_colors:
            if len(self.p) == color_status_in.n:
                color_status_in.resize(color_status_in.n * 2)
                color_status_out.resize(color_status_out.n * 2)

            witness_in_i, witness_in_j, split_deg_in, q_error_in = self.pick_witness(color_status_in)
            witness_out_i, witness_out_j, split_deg_out, q_error_out = self.pick_witness(color_status_out)

            if len(self.p) % 10 == 0:
                print(f"{len(self.p)} colors with {max(q_error_in, q_error_out)} error, median: {np.median([len(i) for i in self.p])}")

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
        print(f"refined and got {len(self.p)} colors with {self.q_error} q-error")