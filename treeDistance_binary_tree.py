import numpy as np

def generate_level_label(depth):
    """ Generate labels for each level in the binary tree. """
    n = 2 ** depth - 1
    return [int(np.log2(ii + 1)) if ii != 0 else 0 for ii in range(n)]

class Node:
    """ Class representing a node in a binary tree. """
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def path_to_node(root, path, k):
    """ Find path to a node in the binary tree. """
    if root is None:
        return False
    path.append(root.data)
    if root.data == k:
        return True
    if ((root.left and path_to_node(root.left, path, k)) or
        (root.right and path_to_node(root.right, path, k))):
        return True
    path.pop()
    return False

def distance_between_nodes(root, data1, data2):
    """ Compute the distance between two nodes in the binary tree. """
    path1, path2 = [], []
    if root:
        path_to_node(root, path1, data1)
        path_to_node(root, path2, data2)
        i = 0
        while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
            i += 1
        return (len(path1) + len(path2) - 2 * i)
    return 0

def build_binary_tree(depth):
    """ Build a binary tree of given depth. """
    n = 2 ** depth - 1
    nodes = [Node(ii) for ii in range(n)]
    for i in range(n):
        left_index, right_index = 2 * i + 1, 2 * i + 2
        if left_index < n:
            nodes[i].left = nodes[left_index]
        if right_index < n:
            nodes[i].right = nodes[right_index]
    return nodes[0]

def compute_distance_matrix(root, n):
    """ Compute the distance matrix for a binary tree. """
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            distance_matrix[i, j] = distance_between_nodes(root, i, j)
    return distance_matrix + distance_matrix.T
