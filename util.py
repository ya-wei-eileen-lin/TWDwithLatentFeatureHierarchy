from mst import mst
from unionfind import UnionFind

def sl_np_mst(similarities):
    n = similarities.shape[0]
    ij, _ = mst(similarities, n)
    uf = UnionFind(n)
    uf.merge(ij)
    return uf.tree_array