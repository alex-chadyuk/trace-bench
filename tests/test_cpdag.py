from tracebench.cpdag import cpdag_from_dag


def test_chain_is_reversible_and_collider_is_compelled():
    directed, undirected = cpdag_from_dag([("a", "b"), ("b", "c")])
    assert directed == set() and undirected == {frozenset("ab"), frozenset("bc")}
    directed, undirected = cpdag_from_dag([("a", "c"), ("b", "c")])
    assert directed == {("a", "c"), ("b", "c")} and undirected == set()


def test_meek_r1_orients_downstream_of_a_collider():
    # a -> c <- b, c - d  =>  c -> d (R1, since a and b are not adjacent to d)
    directed, undirected = cpdag_from_dag([("a", "c"), ("b", "c"), ("c", "d")])
    assert ("c", "d") in directed and undirected == set()
