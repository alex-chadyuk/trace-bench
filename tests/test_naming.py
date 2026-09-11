"""Synthetic names follow the public grammar and are a pure function of the seed
(PRD scenario 19's public half)."""
from tracebench.naming import NAME_PATTERNS, Namer, load_wordlist, matches_public_grammar
from tracebench.rng import instantiation_generator


def test_wordlist_is_clean():
    words = load_wordlist()
    assert len(words) >= 500
    assert all(w.isalpha() and w.islower() and 4 <= len(w) <= 12 for w in words)
    for real in ("user", "charging", "catalog", "fiber", "lotus", "login"):
        assert real not in words


def test_names_match_grammar_and_are_unique():
    nm = Namer(instantiation_generator(3))
    svcs = nm.services(50)
    eps = nm.endpoints(120)
    ext = nm.externals(2)
    pods = nm.pods(svcs[0], 4)
    assert len(set(svcs)) == 50 and all(matches_public_grammar("service", s) for s in svcs)
    assert len(set(eps)) == 120 and all(matches_public_grammar("endpoint", e) for e in eps)
    assert all(matches_public_grammar("external", e) for e in ext)
    assert all(matches_public_grammar("pod", p) for p in pods)
    assert matches_public_grammar("host", Namer.host(3))
    assert matches_public_grammar("internal_ip", Namer.internal_ip(300))
    assert matches_public_grammar("client_ip", Namer.client_ip(17))
    assert all(matches_public_grammar("client_op", c) for c in nm.client_ops(5))


def test_names_are_seed_deterministic():
    a = Namer(instantiation_generator(11))
    b = Namer(instantiation_generator(11))
    assert a.services(10) == b.services(10)
    assert a.endpoints(20) == b.endpoints(20)
    assert Namer(instantiation_generator(12)).services(10) != Namer(instantiation_generator(11)).services(10)


def test_names_outside_the_grammar_fail():
    assert not matches_public_grammar("service", "svc-user")
    assert not matches_public_grammar("endpoint", "/api/v3/account/login_base")
    assert not matches_public_grammar("host", "api.staging.example.com")
    assert set(NAME_PATTERNS) >= {"service", "endpoint", "pod", "host", "internal_ip", "client_ip"}
