import torch

def construct_adjacency_matrix(data):
    n_nodes = data.x.shape[0]
    n_edges = data.edge_index.shape[1]
    s = torch.sparse_coo_tensor(data.edge_index,
                                [1 for _ in range(n_edges)],
                                (n_nodes, n_nodes))
    return s.to_dense().numpy()


def normalize_enc_torch(L, eps=1e-8):
    """
    Row-wise L2-normalize a 2D tensor.

    L: (n, d) torch tensor
    returns: (n, d) tensor
    """
    norms = torch.norm(L, dim=1, keepdim=True)
    return L / (norms + eps)