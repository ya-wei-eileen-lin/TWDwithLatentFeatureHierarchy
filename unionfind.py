import numpy as np

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n

        self._next_id = n
        self._tree = [-1] * (2 * n - 1)
        self._id = list(range(n))

    def _find(self, i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self._find(self.parent[i])
            return self.parent[i]

    def find(self, i):
        if i < 0 or i >= self.n:
            raise ValueError("Out of bounds index.")
        return self._find(i)

    def union(self, i, j):
        root_i = self._find(i)
        root_j = self._find(j)
        if root_i == root_j:
            return False
        else:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self._build(root_j, root_i)
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self._build(root_i, root_j)
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                self._build(root_i, root_j)
            return True

    def merge(self, ij):
        """ Merge a sequence of pairs """
        for pair in ij:
            self.union(pair[0], pair[1])

    def _build(self, i, j):
        """ Track the tree changes when node j gets merged into node i """
        self._tree[self._id[i]] = self._next_id
        self._tree[self._id[j]] = self._next_id
        self._id[i] = self._next_id
        self._next_id += 1

    @property
    def sets(self):
        return 2 * self.n - self._next_id

    @property
    def parent_array(self):
        return [self.parent[i] for i in range(self.n)]

    @property
    def tree_array(self):
        return [self._tree[i] for i in range(2 * self.n - 1)]

# if __name__ == "__main__":
#     uf = UnionFind(5)
#     uf.merge(np.array([[0, 1], [2, 3], [0, 4], [3, 4]]))

#     print("parent_array:", uf.parent_array)
#     print("tree_array:", uf.tree_array)