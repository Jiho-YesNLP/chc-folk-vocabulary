"""Lexical tooling for Stage-4 Step B: tokenization, per-ability over-representation
(the **lexical** fingerprint component), and cross-ability IDF (the distinctiveness
half of canonical-description salience).

All deterministic; nltk WordNet lemmatizer + a small built-in stopword set (no extra
downloads beyond wordnet/omw). Tokenization is content-word only: alphabetic, length
>= 3, not a stopword, lemmatized to noun then verb. Contractions and possessives are
reduced to their base first ("it's" -> "it" -> dropped as a stopword, "teacher's" ->
"teacher"), so function words do not leak in and a possessive is not counted as a
separate term from its base noun.

Also provides a **proper-noun filter** for the lexical component: corpus casing
evidence (``case_counts``) plus a WordNet membership test (``proper_nouns``). Personal
names and platform names picked up from the passages -- ``sharleen``, ``maddie``,
``reddit`` -- are corpus artifacts rather than folk vocabulary for an ability, but
``german``/``french``/``asl`` genuinely *are* the folk vocabulary for Gkn-KL and
Gkn-KF, so casing alone is not enough to tell them apart. Requiring both conditions
keeps the dictionary words and drops the names. No extra nltk downloads.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from nltk.corpus import wordnet as _wn
from nltk.stem import WordNetLemmatizer

_word_re = re.compile(r"[a-z][a-z'-]+")
_lem = WordNetLemmatizer()

# Contraction / possessive suffixes, longest-first so "n't" wins over "'t".
# Stripping these before the stopword check does two things: function-word contractions
# reduce to a stopword and drop out ("it's"->"it", "don't"->"do", "you're"->"you"), and
# possessives merge with their base noun ("teacher's"->"teacher") instead of being
# counted as a separate term. Residues shorter than 3 chars fall out on their own
# ("can't"->"ca", "won't"->"wo").
_CONTRACTIONS = ("n't", "'re", "'ve", "'ll", "'s", "'d", "'m", "'t")


def _decontract(w: str) -> str:
    for suf in _CONTRACTIONS:
        if w.endswith(suf) and len(w) > len(suf):
            return w[: -len(suf)]
    return w

# compact stopword list — descriptor passages, so keep it lean and obvious
STOPWORDS = set("""
a an the and or but if then than so because as of at by for with about against between into
through during before after above below to from up down in out on off over under again further
once here there all any both each few more most other some such no nor not only own same too very
can will just don should now i me my we our you your he him his she her it its they them their this
that these those am is are was were be been being have has had do does did doing would could should
get got getting really thing things stuff like one two also even still much many lot way ways able
kind sort make makes made making want wants try trying use using used know knew known see saw seen
think thought feel felt go goes going come comes came people person someone something anything
""".split())


def tokens(text: str) -> list[str]:
    """Content lemmas of a passage/span (lowercased, alpha, len>=3, de-stopworded)."""
    out: list[str] = []
    for w in _word_re.findall((text or "").lower()):
        w = _decontract(w.strip("'-")).strip("'-")
        if len(w) < 3 or w in STOPWORDS:
            continue
        lemma = _lem.lemmatize(_lem.lemmatize(w, "n"), "v")
        if len(lemma) >= 3 and lemma not in STOPWORDS:
            out.append(lemma)
    return out


# Sentence-final punctuation: a token right after one of these (or at the very start of
# the text) is capitalized by position, so its casing carries no proper-noun signal.
_SENT_END = set(".!?;:\n\r")
_OPENERS = set("\"'`([{*_>-‘“ \t")


def _lemma(word: str) -> str:
    """Lowercased word -> the same lemma ``tokens()`` would emit, or '' if not a content token."""
    w = _decontract(word.strip("'-")).strip("'-")
    if len(w) < 3 or w in STOPWORDS:
        return ""
    lemma = _lem.lemmatize(_lem.lemmatize(w, "n"), "v")
    return lemma if len(lemma) >= 3 and lemma not in STOPWORDS else ""


def case_counts(text: str, cap: Counter, low: Counter) -> None:
    """Accumulate per-lemma casing evidence from *raw* (un-lowercased) text.

    Counts only occurrences whose capitalization is *informative*, i.e. skips:
      - sentence-initial tokens (every sentence starts with a capital), and
      - ALL-CAPS tokens, which in this corpus are emphasis or acronyms ("I did TERRIBLE
        with math", "ASL"), not proper nouns.

    Mutates ``cap``/``low`` in place so one pass can serve the whole corpus.
    """
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]+", text or ""):
        word = m.group(0)
        # sentence-initial? walk back over opening punctuation/whitespace
        i = m.start() - 1
        while i >= 0 and text[i] in _OPENERS:
            i -= 1
        if i < 0 or text[i] in _SENT_END:
            continue
        if word.isupper():          # emphasis / acronym, not a name
            continue
        lemma = _lemma(word.lower())
        if not lemma:
            continue
        (cap if word[0].isupper() else low)[lemma] += 1


def proper_nouns(cap: Counter, low: Counter, min_obs: int = 4,
                 cap_ratio: float = 0.85) -> set[str]:
    """Lemmas that look like proper nouns and should be kept out of lexical markers.

    A lemma qualifies only when **both** hold:
      1. casing: at least ``min_obs`` informative occurrences, of which at least
         ``cap_ratio`` are capitalized; and
      2. WordNet: the lemma has no synset.

    Condition 2 is what protects the meaningful proper nouns -- ``german``, ``french``,
    ``english``, ``asl`` are all in WordNet and survive, while ``sharleen``, ``maddie``,
    ``reddit`` are not and are dropped. Returns the excluded set (log it; it is the
    audit trail for what the filter removed).
    """
    out: set[str] = set()
    for lemma, c in cap.items():
        total = c + low.get(lemma, 0)
        if total < min_obs or c / total < cap_ratio:
            continue
        if _wn.synsets(lemma):
            continue
        out.add(lemma)
    return out


def lexical_markers(
    ability_counts: Counter, background_counts: Counter,
    min_count: int = 3, top_k: int = 25, exclude: set[str] | None = None,
) -> list[dict]:
    """Terms over-represented in one ability vs. the pooled background, by frequency lift (PMI).

    score = log2( p(term | ability) / p(term | background) ); requires count >= min_count.
    ``exclude`` drops terms outright (see ``proper_nouns``) -- applied after the totals
    are computed, so removing a name does not reweight the remaining terms.
    Returns top_k dicts {term, count, ability_freq, lift} sorted by lift then count.
    """
    exclude = exclude or set()
    a_total = sum(ability_counts.values()) or 1
    bg_total = sum(background_counts.values()) or 1
    out = []
    for term, c in ability_counts.items():
        if c < min_count or term in exclude:
            continue
        p_a = c / a_total
        p_bg = background_counts.get(term, 0) / bg_total
        # Floor at one pseudo-count (not Laplace -- the denominator is unchanged, and this
        # only binds for terms absent from the background) so a term unique to this ability
        # still scores finitely.
        p_bg = max(p_bg, 1.0 / bg_total)
        lift = math.log2(p_a / p_bg)
        out.append({"term": term, "count": c, "ability_freq": round(p_a, 5), "lift": round(lift, 4)})
    out.sort(key=lambda d: (-d["lift"], -d["count"]))
    return out[:top_k]


def ability_idf(ability_token_sets: dict[str, set]) -> dict[str, float]:
    """IDF of each lemma across abilities: log((1+N)/(1+df)) + 1.

    df = number of abilities whose passages/spans contain the lemma. Used as the
    cross-ability *distinctiveness* weight for canonical-description salience.
    """
    n = len(ability_token_sets)
    df: Counter = Counter()
    for toks in ability_token_sets.values():
        for t in set(toks):
            df[t] += 1
    return {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
