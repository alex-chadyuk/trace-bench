"""Synthetic names that cannot collide with a real system's names.

Every service, endpoint, pod, host and address in a corpus is drawn from a
checked-in list of pseudo-words (consonant-vowel syllables, no natural-language
words) under a fixed grammar, so a published artifact can be scanned for the
grammar (PRD scenario 19) and, on the private side, against a denylist of real
names. Names are indexed by `Generator.integers`, never by iteration order of a
set, so they are a pure function of the seed.
"""
import re
from importlib import resources

SERVICE_PREFIX = "tb-"
HOST_SUFFIX = ".tb.internal"
INTERNAL_NET = "10.77"

# Public grammar: `verify` asserts every name in every artifact matches one of these.
SERVICE_RE = re.compile(r"^tb-[a-z]{4,12}$")
BFF_SERVICE = "tb-edge"
EXTERNAL_RE = re.compile(r"^ext-[a-z]{4,12}$")
ENDPOINT_RE = re.compile(r"^/v1/[a-z]{4,12}/[a-z]{4,12}$")
CLIENT_OP_RE = re.compile(r"^/page/[a-z]{4,12}(/[a-z]{4,12})?$")
POD_RE = re.compile(r"^(tb|ext)-[a-z]{4,12}-[0-9a-f]{5}-[0-9a-f]{5}$")
HOST_RE = re.compile(r"^node-\d+\.tb\.internal$")
INTERNAL_IP_RE = re.compile(r"^10\.77\.\d{1,3}\.\d{1,3}$")
# RFC 5737 documentation ranges for client addresses.
CLIENT_IP_RE = re.compile(r"^(192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}$")
NAME_PATTERNS = {
    "service": SERVICE_RE, "external": EXTERNAL_RE, "endpoint": ENDPOINT_RE,
    "client_op": CLIENT_OP_RE, "pod": POD_RE, "host": HOST_RE,
    "internal_ip": INTERNAL_IP_RE, "client_ip": CLIENT_IP_RE,
}


def load_wordlist():
    text = resources.files("tracebench").joinpath("wordlist.txt").read_text(encoding="utf-8")
    words = [w for w in text.split("\n") if w]
    if len(words) != len(set(words)):
        raise ValueError("wordlist.txt contains duplicates")
    return words


class Namer:
    """Seeded name factory. `rng` must be the INSTANTIATE stream; every draw is
    an `integers` call so the mapping seed -> names is stable."""

    def __init__(self, rng, wordlist=None):
        self.rng = rng
        self.words = list(wordlist or load_wordlist())
        self._perm = [self.words[i] for i in rng.permutation(len(self.words))]
        self._cursor = 0

    def _word(self):
        if self._cursor >= len(self._perm):
            raise ValueError(f"wordlist exhausted after {len(self._perm)} unique words")
        w = self._perm[self._cursor]
        self._cursor += 1
        return w

    def services(self, n):
        return [SERVICE_PREFIX + self._word() for _ in range(n)]

    def externals(self, n):
        return ["ext-" + self._word() for _ in range(n)]

    def endpoints(self, n):
        """`n` distinct `/v1/<noun>/<verb>` paths sharing one noun (the noun is
        unique per call, i.e. per service, so paths never repeat across
        services); verbs are drawn with replacement and de-duplicated."""
        noun = self._word()
        out = []
        while len(out) < n:
            verb = self.words[int(self.rng.integers(len(self.words)))]
            path = f"/v1/{noun}/{verb}"
            if path not in out:
                out.append(path)
        return out

    def client_ops(self, n):
        return [f"/page/{self._word()}" for _ in range(n)]

    def pods(self, service, n):
        return [f"{service}-{self.hex5()}-{self.hex5()}" for _ in range(n)]

    def hex5(self):
        return "".join("0123456789abcdef"[int(i)] for i in self.rng.integers(16, size=5))

    @staticmethod
    def host(index):
        return f"node-{index}{HOST_SUFFIX}"

    @staticmethod
    def internal_ip(index):
        return f"{INTERNAL_NET}.{(index // 254) % 256}.{index % 254 + 1}"

    @staticmethod
    def client_ip(index):
        nets = ("192.0.2", "198.51.100", "203.0.113")
        return f"{nets[index % 3]}.{(index // 3) % 254 + 1}"


def matches_public_grammar(kind, name):
    return bool(NAME_PATTERNS[kind].match(name))
