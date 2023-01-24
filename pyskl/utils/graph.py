import numpy as np
import torch


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


part1 = [(1,20),(20,2),(2,3)]
part2 = [(2,20),(20,4),(4,5),(5,6),(6,7),(6,22),(22,21)]
part3 = [(7,6),(6,5),(5,4),(4,20),(20,8),(8,9),(9,10),(10,11),(10,24),(24,23)]
part4 = [(11,10),(10,9),(9,8),(8,20),(20,1),(1,0),(0,12),(12,13),(13,14),(14,15)]
part5 = [(14,13),(13,12),(12,0),(0,16),(16,17),(17,18),(18,19)]
part6 = [(18,17),(17,16),(16,0),(0,1)]

part7 = [(8,20),(20,2),(2,3),(20,4)]
part8 = [(8,9),(9,10),(10,11),(10,24),(24,23),(4,5),(5,6),(6,7),(6,22),(22,21)]
part9 = [(20,8),(20,4),(20,1),(1,0),(0,16),(0,12),]
part10 = [(16,17),(17,18),(18,19),(12,13),(13,14),(14,15)]

# part1 = [(0,1),(1,3),(0,2),(2,4)]
# part2 = [(0,5),(5,7),(7,9)]
# part3 = [(7,5),(5,0),(0,11),(11,13),(13,15)]
# part4 = [(13,11),(11,0),(0,12),(12,14),(14,16)]
# part5 = [(14,12),(12,0),(0,6),(6,8),(8,10)]
# part6 = [(8,6),(6,0)]
#
# part7 = [(0,0),(0,1),(0,2),(1,3),(2,4),(0,5),(0,6)]
# part8 = [(5,7),(7,9),(6,8),(8,10)]
# part9 = [(0,5),(0,6),(0,11),(0,12)]
# part10 = [(11,13),(13,15),(12,14),(14,16)]

class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 parts=[part1, part2, part3, part4, part5, part6, part7, part8, part9, part10],
                 max_hop=1):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.parts = parts

        assert layout in ['openpose', 'nturgb+d', 'coco']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)
            ]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        stack = []
        stack.append(edge2mat(self.self_link, self.num_node))
        for p in self.parts:
            In = normalize_digraph(edge2mat(p, self.num_node))
            stack.append(In)
            opp = [(y, x) for (x, y) in p]
            Out = normalize_digraph(edge2mat(opp, self.num_node))
            stack.append(Out)

        A = np.stack(stack)
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]
