import numpy as np
import networkx as nx
from scipy import sparse

def create_parent_child_matrix(tree, is_leaf=True):
    """
    Create a parent-child relationship matrix for a given tree.

    Parameters:
    tree (networkx.Graph): The tree as a NetworkX Graph.
    is_leaf (bool): If True, includes leaves in the matrix; otherwise, only internal nodes are included.

    Returns:
    scipy.sparse.csc_matrix: The parent-child relationship matrix.
    """
    n_node = len(tree.nodes())
    leaf_nodes = [node for node, degree in tree.degree() if degree == 1]
    n_leaf = len(leaf_nodes)
    root = max(list(tree.nodes()))

    path_leaves = [nx.shortest_path(tree, source=root, target=leaf) for leaf in leaf_nodes]

    n_edge = sum(len(path) for path in path_leaves)
    col_ind = np.zeros(n_edge)
    row_ind = np.zeros(n_edge)

    cnt = 0
    for path in path_leaves:
        leaf_index = leaf_nodes.index(path[-1])
        for node in path:
            node_index = list(tree.nodes()).index(node)
            col_ind[cnt] = leaf_index
            row_ind[cnt] = node_index
            cnt += 1

    parent_child_matrix = sparse.csc_matrix((np.ones(n_edge), (row_ind, col_ind)), shape=(n_node, n_leaf), dtype='float32')

    if not is_leaf:
        parent_child_matrix = parent_child_matrix[n_leaf:, :]

    return parent_child_matrix


