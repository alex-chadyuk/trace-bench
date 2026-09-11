"""CPDAG of a DAG (v-structures + Meek rules R1–R4) and its compelled edges.

Used to split orientation accuracy into compelled (identifiable, a method is
accountable for it) and reversible (a negative control at chance) — the
convention the lab's tail-gfn MEC validation follows. Pure Python, small graphs.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def cpdag_from_dag(edges):
    """edges: iterable of (a, b) meaning a -> b, acyclic. Returns
    (directed, undirected): directed = set of (a, b) compelled edges,
    undirected = set of frozenset({a, b}) reversible edges."""
    parents = defaultdict(set)
    adj = defaultdict(set)
    for a, b in edges:
        parents[b].add(a)
        adj[a].add(b)
        adj[b].add(a)
    nodes = set(adj)
    directed = set()
    # v-structures: a -> c <- b with a, b non-adjacent
    for c in nodes:
        for a, b in combinations(sorted(parents[c]), 2):
            if b not in adj[a]:
                directed.add((a, c))
                directed.add((b, c))
    undirected = {frozenset((a, b)) for a, b in edges if (a, b) not in directed}
    changed = True
    while changed:
        changed = False
        for e in sorted(undirected, key=lambda s: tuple(sorted(s))):
            a, b = sorted(e)
            for x, y in ((a, b), (b, a)):
                if _meek_orient(x, y, directed, undirected, adj):
                    directed.add((x, y))
                    undirected.discard(e)
                    changed = True
                    break
            if changed:
                break
    return directed, undirected


def _has_dir(directed, a, b):
    return (a, b) in directed


def _meek_orient(x, y, directed, undirected, adj):
    """True if the rules force x -> y for the undirected edge x - y."""
    nbrs = adj
    # R1: z -> x - y, z not adjacent to y  =>  x -> y
    for z in nbrs[x]:
        if _has_dir(directed, z, x) and z not in nbrs[y] and z != y:
            return True
    # R2: x -> z -> y and x - y  =>  x -> y
    for z in nbrs[x]:
        if _has_dir(directed, x, z) and _has_dir(directed, z, y):
            return True
    # R3: x - z1 -> y, x - z2 -> y, z1, z2 non-adjacent, x - y  =>  x -> y
    cands = [z for z in nbrs[x] if frozenset((x, z)) in undirected and _has_dir(directed, z, y)]
    for z1, z2 in combinations(cands, 2):
        if z2 not in nbrs[z1]:
            return True
    # R4: x - z1 -> z2 -> y with x - z2 (or x adjacent), z1 not adjacent to y  =>  x -> y
    for z1 in nbrs[x]:
        if frozenset((x, z1)) not in undirected:
            continue
        for z2 in nbrs[z1]:
            if _has_dir(directed, z1, z2) and _has_dir(directed, z2, y) and z2 in nbrs[x] and y not in nbrs[z1]:
                return True
    return False
