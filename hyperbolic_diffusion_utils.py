import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import norm
from diffusion_operator_util import *
from poincare_half_space_model import *
from util import *
from tree_Wasserstein import *

TOLERANCE = 1e-6

def compute_hyperbolic_diffusion(eigenvalues, left_eigenvectors, right_eigenvectors, num_steps):
    """
    Compute hyperbolic diffusion representation of data.
    Code modified from Lin, Y.-W. E., Coifman, R. R., Mishne, G., and Talmon, R. 
    Hyperbolic diffusion embedding and distance for hierarchical representation learning. 
    In International Conference on Machine Learning, pp. 21003–21025. PMLR, 2023.
    """
    time_steps = 1 / np.power(2, np.arange(num_steps))
    weights = 2 / np.power(2, np.arange(num_steps) / 2)

    diffusion_matrices = []
    embedding = []
    distance = []

    for i, time_step in enumerate(time_steps):
        scaled_eigenvalues = np.power(eigenvalues, time_step)
        diffusion_matrix = (left_eigenvectors @ np.diag(scaled_eigenvalues) @ right_eigenvectors)
        diffusion_matrix = np.sqrt(np.where(diffusion_matrix > TOLERANCE, diffusion_matrix, TOLERANCE))

        diffusion_matrices.append(2 * np.arcsinh(weights[i] * pairwise_distances(diffusion_matrix)))

        additional_column = (1 / (2 * weights[i])) * np.ones((left_eigenvectors.shape[0], 1))
        concatenated_matrix = np.concatenate((diffusion_matrix, additional_column), axis=1)
        embedding.append(concatenated_matrix)

    distance = np.sum(diffusion_matrices, axis=0)
    
    return embedding, distance

import numpy as np

def manifold_distance(embeddings, j, jp):
    """
    Calculate the distance d_Mc(j, jp) in the manifold Mc.

    Args:
    embeddings (list of numpy arrays): A list where each element is a numpy array of embeddings at level k.
    j (int): Index of the first point.
    jp (int): Index of the second point.
    Kc (int): The number of levels to sum over.

    Returns:
    float: The calculated distance.
    """
    distance = 0
    k_c = len(embeddings)
    for k in range(k_c):
        zj = embeddings[k][j]
        zjp = embeddings[k][jp]
        norm = np.linalg.norm(zj - zjp)
        distance += 2 * np.arcsinh(np.power(2, -k/2 + 1) * norm)
    return distance

def compute_geometric_mean(midpoints):
    """Compute the geometric mean of midpoints for each pair across levels."""
    level = len(midpoints)
    product = np.ones((midpoints[0].shape[0], midpoints[0].shape[1]))
    for k in range(level):
        product *= midpoints[k]
    hd_lca = np.power(product, 1.0 / (level))
    return hd_lca

def decode_tree(embeddings_list, distances):
    """Build a tree from a list of hyperbolic embeddings."""
    n = embeddings_list[0].shape[0]
    pairs = [(i, j) for i in range(n) for j in range(n)]

    # Compute midpoints for each pair across all levels of embeddings
    midpoints = []
    for k in range(len(embeddings_list)):
        midpoints_k =  np.array([midpoint_same_level(embeddings_list[k][i], embeddings_list[k][j])[1] for i, j in pairs])  
        midpoints_k = midpoints_k.reshape(n, n)
        midpoints.append(midpoints_k)
    
    pair_LCA = compute_geometric_mean(midpoints)
    parents = sl_np_mst(pair_LCA)
    # print('parents', sorted(parents))
    tree = nx.DiGraph()
    for i, j in enumerate(parents[:-1]):
        tree.add_edge(j, i)
    return tree

import numpy as np

def complete_tree(tree, leaves_embeddings_list):
    """Get embeddings of internal nodes from leaves' embeddings using HD LCA construction."""

    def _complete_tree(embeddings, node):
        children = list(tree.neighbors(node))
        if len(children) == 2:
            left_c, right_c = children
            left_leaf = is_leaf(tree, left_c)
            right_leaf = is_leaf(tree, right_c)
            if left_leaf and right_leaf:
                pass
            elif left_leaf and not right_leaf:
                embeddings = _complete_tree(embeddings, right_c)
            elif right_leaf and not left_leaf:
                embeddings = _complete_tree(embeddings, left_c)
            else:
                embeddings = _complete_tree(embeddings, right_c)
                embeddings = _complete_tree(embeddings, left_c)
            embeddings[node] = midpoint(embeddings[left_c], embeddings[right_c])
        return embeddings

    K = len(leaves_embeddings_list)  # Number of embeddings
    n = leaves_embeddings_list[0].shape[0]  # Dimension of each embedding

    # Initialize tree embeddings for each embedding in the list
    tree_embeddings_list = [np.zeros((2 * n - 1, n + 1)) for _ in range(K)]
    for i in range(K):
        tree_embeddings_list[i][:n, :] = leaves_embeddings_list[i]
    
    root = max(list(tree.nodes()))
    for i in range(K):
        tree_embeddings_list[i] = _complete_tree(tree_embeddings_list[i], root)

    return tree_embeddings_list

def is_leaf(tree, node):
    """Check if a node is a leaf in a tree."""
    return len(list(tree.neighbors(node))) == 0

def midpoint(x, y):
    """Find the midpoint of two points in hyperbolic space."""
    if x[-1] != y[-1]:
        # raise Warning("Points are not from the same level")
        if x[-1]>y[-1]:
            x[-1] = y[-1]
        else:
            y[-1] = x[-1]
    z, _ = midpoint_same_level(x,y)
    return z

def hyperbolic_diffusion_tree(tree, tree_embeddings_list):
    """
    Assign weights to the edges of a tree based on the distance between node embeddings and return the tree.

    Parameters:
    tree (networkx.Graph): The tree as a NetworkX Graph.
    embeddings (list or np.ndarray): The embeddings corresponding to each node.
                                    The index in the embeddings should correspond to the node.
    
    Returns:
    networkx.Graph: The tree with edge weights assigned.
    """
    n = len(tree.nodes())
    edge_weight_matrix = np.zeros((n, n))

    for edge in tree.edges():
        node1, node2 = edge
        distance = manifold_distance(tree_embeddings_list, node1, node2)
        tree[node1][node2]['weight'] = distance
        edge_weight_matrix[node1][node2] = distance
        edge_weight_matrix[node2][node1] = distance  
    return tree, edge_weight_matrix

def regularization_function(Z, j, j_prime, theta=1e-6):
    """
    Compute the regularization function for the given indices j and j_prime.

    Parameters:
    Z (np.ndarray): The matrix Z.
    j (int): The first column index.
    j_prime (int): The second column index.
    theta (float): The theta constant in the integral.

    Returns:
    float: The result of the regularization function.
    """
    def integrand(xi):
        return 1 / (np.sqrt(xi) + theta)

    distance = norm(Z[:, j] - Z[:, j_prime])
    result, _ = quad(integrand, 0, distance)
    return 0.5 * result

def regularization_matrix(Z, theta=1e-6):
    """
    Compute the regularization matrix for matrix Z.

    Parameters:
    Z (np.ndarray): The matrix Z.
    theta (float): The theta constant in the integral.

    Returns:
    np.ndarray: The regularization matrix.
    """
    def integrand(xi):
        return 1 / (np.sqrt(xi) + theta)

    m = Z.shape[1]
    regularization_mat = np.zeros((m, m))

    for j in range(m):
        for j_prime in range(m):
            if j != j_prime:
                distance = norm(Z[:, j] - Z[:, j_prime])
                result, _ = quad(integrand, 0, distance)
                regularization_mat[j, j_prime] = 0.5 * result

    return regularization_mat

def pairwise_tree_ot_distance(tree, embedding, a, b):
    """
    Calculate the tw distance between two probability distributions (a and b)
    over a tree structure. This function first computes a complete tree with the given embeddings,
    applies hyperbolic diffusion to determine edge weights, and then uses these weights along
    with a parent-child mask to calculate the distance between the distributions.

    Parameters:
    tree (networkx.Graph): A tree represented as a NetworkX Graph.
    embedding (np.ndarray): Embeddings associated with the nodes of the tree.
    a (np.ndarray): The first probability distribution over the nodes of the tree.
    b (np.ndarray): The second probability distribution over the nodes of the tree.

    Returns:
    float: The optimal transport distance between distributions a and b.
    """
    complete_tree__list = complete_tree(tree, embedding)
    _, edge_weight_matrix = hyperbolic_diffusion_tree(tree, complete_tree__list)
    parent_child_mask = create_parent_child_matrix(tree,n)
    distance = abs((edge_weight_matrix @ parent_child_mask).dot(a-b)).sum(0)
    return distance

def twd_hidden_feature(data_X, level_K, dis_mat=None, metric='cosine'):
    """
    Compute the Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy (twd_hidden) distances for a dataset. The function
    calculates pairwise distances using hyperbolic diffusion processes on a tree structure,
    which is derived from the data. It involves computing diffusion embeddings, constructing
    a tree, and then applying hyperbolic diffusion to determine the pairwise distances
    between data points.

    Parameters:
    data_X (np.ndarray): The input data matrix.
    level_K (int): The number of steps in the hyperbolic diffusion process.
    dis_mat (np.ndarray, optional): A precomputed distance matrix. If None, it will be computed.
    metric (str, optional): The distance metric to use if dis_mat is None. Defaults to 'cosine'.

    Returns:
    np.ndarray: A matrix of pairwise Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy
    """
    n = data_X.shape[0]
    twd_hidden = np.zeros((n, n))
    if dis_mat is None:
        dis_mat = pairwise_distances(data_X.T, metric=metric)
    eigenvalues, left_eigenvectors, right_eigenvectors = compute_diffusion_operator_normalized(data_X.T, dis_mat, full_spectrum=True)
    normalized_data = data_X / np.sum(data_X, axis=1)[:, None]
    embeddings, distances = compute_hyperbolic_diffusion(eigenvalues, left_eigenvectors, right_eigenvectors, level_K)
    tree = decode_tree(embeddings, distances)
    complete_tree_list = complete_tree(tree, embeddings)
    _, edge_weight_matrix = hyperbolic_diffusion_tree(tree, complete_tree_list)
    parent_child_mask = create_parent_child_matrix(tree, n)
    for i in range(n):
        for j in range(i):
            twd_hidden[i, j] = abs((edge_weight_matrix @ parent_child_mask) @ (normalized_data[i, :] - normalized_data[j, :])).sum()
    twd_hidden += twd_hidden.T
    return twd_hidden
