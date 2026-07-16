#!/usr/bin/env python3
"""Claim-backing linter.

Verifies that every claim in a LaTeX manuscript is backed either by a
proof in the manuscript itself or by a precise external reference
(precise = a citation carrying a locator such as a theorem number,
section, or page: ``\\cite[Thm.~3.3]{key}``, ``\\cite[Sec.~V]{key}``,
``\\cite[p.~14]{key}`` — a bare ``\\cite{key}`` does not count for a
mathematical claim).

Pipeline (as requested):
  1. The body of the manuscript is stripped of preamble, comments,
     TikZ drawing code, algorithmic pseudocode, and embedded
     bibliographies, then split into paragraphs.
  2. Paragraphs are grouped into OVERLAPPING CHUNKS (default: 3
     paragraphs with stride 2), so every claim is analyzed together
     with its surrounding context at least once.
  3. Within each chunk, sentences carrying a claim signal (converges,
     is NP-hard, guarantees, is equivalent to, minimax, never
     increases, ...) are checked for backing IN CONTEXT — the sentence
     itself, its predecessor, and its successor inside the chunk:
       PROOF        anchored to a proposition/theorem/lemma/proof or an
                    equation chain in the manuscript,
       REF-PRECISE  citation with a locator,
       EVIDENCE     experimental claim anchored to a figure/table,
       REF-IMPRECISE (WARN) citation without a locator,
       UNBACKED     (FAIL) no anchor at all.
  4. Claims seen in several overlapping chunks are reported once.

Additionally, every theorem-like environment must be followed by a
proof environment (or an explicit pointer to one); otherwise FAIL.

Multi-file / companion-document mode: pass the main manuscript together
with a companion file (e.g. a supplementary/appendix document) and both
are loaded as one corpus before linting. A pointer phrase like "the proof
is deferred to the supplementary material" is then VERIFIED rather than
trusted at face value: it only clears the FAIL if the *other* loaded
file actually contains a ``\begin{proof}``. If no companion file is
given, such pointers fall back to the old trust-the-pointer behaviour
(each file is linted independently).

An optional ``--llm`` mode pipes each chunk, together with the
manuscript's proof inventory, to the ``claude`` CLI for semantic claim
extraction, and merges those verdicts with the heuristic ones. The
default mode is fully offline.

Usage:
    python3 linters/lint_claims.py FILE.tex [--chunk-size 3]
        [--stride 2] [--llm] [--show-backed]
    python3 linters/lint_claims.py ab_fedkmeans.tex supplementary.tex

Exit code: 1 if any FAIL finding, else 0.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Claim signals: a sentence containing one of these makes an assertion
# that needs backing.
# ---------------------------------------------------------------------------
CLAIM_SIGNALS = [
    r"\bis NP-hard\b", r"\bconverge[sd]?\b", r"\bconvergence\b.*\bguarantee",
    r"\bguarantee[sd]?\b", r"\bit is known\b", r"\bit can be shown\b",
    r"\b(?:was|has been|have been) (?:shown|proved|proven|established)\b",
    r"\bmonotonically (?:decreas|increas)", r"\bnever increases\b",
    r"\bis equivalent to\b", r"\bif and only if\b", r"\bminimax\b",
    r"\boptimal(?:ity)?\b(?![- ]transport)", r"\bupper[- ]bound", r"\blower[- ]bound",
    r"\bis (?:jointly )?convex\b", r"\bpositive (?:semi-)?definite\b",
    r"\bis invariant\b", r"\bpermutation-invariant\b.*\bby construction\b",
    r"\brecovers\b", r"\bcoincides? with\b", r"\bcorresponds to the limit\b",
    r"\bwe prove\b", r"\battains?\b", r"\bis minimized (?:exactly )?by\b",
    r"\bcomplexity\b", r"\bcosts? \$?\\?[a-zA-Z0-9^{}\\]+\$? per\b",
    r"\bimplies\b", r"\bholds (?:with|for|in|if|whenever)\b", r"\bis unbiased\b", r"\bexists? candidates\b",
    r"\bstrict(?:ly)? (?:descent|decreas)", r"\bunconditionally\b",
]
CLAIM_RE = re.compile("|".join(CLAIM_SIGNALS), re.IGNORECASE)

# Sentences that are announcements/definitions, not claims to verify.
EXEMPT_RE = re.compile(
    r"^\s*(?:We (?:propose|introduce|present|describe|denote|write|collect|"
    r"use|refer|call|study|consider|investigate|report|compare|evaluate)|"
    r"In this (?:paper|section|work)|Section~?\\ref|The rest of this paper|"
    r"Our (?:main )?contributions|Here, we used|We denote|Note that we always)",
    re.IGNORECASE)

# Backing anchors -----------------------------------------------------------
PROOF_ANCHOR_RE = re.compile(
    r"\\(?:eqref|ref)\{(?:prop|thm|lem|cor|sec|equ|eq|sapp)[-:_]"
    r"|Proposition[\s~]*\\ref|Theorem[\s~]*\\ref|Lemma[\s~]*\\ref"
    r"|Corollary[\s~]*\\ref"
    r"|\bAppendix\b|shown in Section|see the proof|by the .{0,40}argument"
    r"|\(shown in|\bProof\b|proved? (?:in|below|above)"
    r"|supplementary material",
    re.IGNORECASE)
# A prose pointer to the companion document, as distinct from an
# in-manuscript anchor: "the proof/bound/calculation/derivation is in the
# supplementary material". Only trusted at face value when no companion
# file was actually loaded; when one is loaded, it must be backed by a
# real proof/derivation found there (see cross_file_has_backing).
SUPP_POINTER_RE = re.compile(r"supplementary material", re.IGNORECASE)
EVIDENCE_ANCHOR_RE = re.compile(
    r"\\ref\{(?:fig|tab)[:_]|Figure[s]?[~ ]|Table[s]?[~ ]|Fig\.~?\\ref")
CITE_ANY_RE = re.compile(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])?\{[^}]+\}")
CITE_PRECISE_RE = re.compile(
    r"\\cite[a-zA-Z]*\[(?:[^\]]*(?:Thm|Theorem|Prop|Proposition|Lem|Lemma|Cor|"
    r"Corollary|Sec|Section|Ch|Chapter|Eq|eq\.|p\.|pp\.|page|Alg|Table|"
    r"Fig)[^\]]*)\]\{[^}]+\}")
MATH_DISPLAY_RE = re.compile(r"\\begin\{(?:equation|align)")

THEOREM_ENVS = ["theorem", "proposition", "lemma", "corollary"]

ABBREVS = ["e.g.", "i.e.", "cf.", "et al.", "vs.", "w.r.t.", "resp.",
           "Sec.", "Fig.", "Figs.", "Eq.", "Thm.", "Prop.", "No.", "nr.",
           "max.", "min.", "Ch.", "pp.", "p.", "Alg.", "approx."]


# ---------------------------------------------------------------------------
# Manuscript preparation
# ---------------------------------------------------------------------------

def strip_env(text, env):
    return re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
                  f" [{env} omitted] ", text, flags=re.DOTALL)


def prepare_body(text):
    m = re.search(r"\\begin\{document\}", text)
    if m:
        text = text[m.end():]
    text = strip_env(text, "abstract")   # abstracts summarize; claims
                                         # are checked where elaborated
    text = re.sub(r"(?<!\\)%.*", "", text)              # comments
    for env in ("filecontents", "tikzpicture", "algorithmic",
                "thebibliography", "IEEEbiographynophoto"):
        text = strip_env(text, env)
    return text


def split_paragraphs(text):
    """Paragraphs with their starting line numbers (in the body)."""
    paras, cur, start = [], [], 1
    for i, ln in enumerate(text.split("\n"), 1):
        if ln.strip():
            if not cur:
                start = i
            cur.append(ln.strip())
        elif cur:
            paras.append((start, " ".join(cur)))
            cur = []
    if cur:
        paras.append((start, " ".join(cur)))
    # drop pure-markup paragraphs
    return [(l, p) for (l, p) in paras
            if re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*|\W", "", p)]


def make_chunks(paras, size, stride):
    chunks = []
    i = 0
    while i < len(paras):
        chunk = paras[i:i + size]
        if chunk:
            chunks.append(chunk)
        if i + size >= len(paras):
            break
        i += stride
    return chunks


def split_sentences(paragraph):
    guarded = paragraph
    for a in ABBREVS:
        guarded = guarded.replace(a, a.replace(".", "․"))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\\$(])", guarded)
    return [p.replace("․", ".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def classify(sentence, context, companion_has_proofs=None):
    """Return (verdict, detail). `context` = sentence plus its chunk
    neighbours. `companion_has_proofs`: None if no companion document was
    loaded (a bare "supplementary material" pointer is then trusted at
    face value, as before); True/False if one was loaded, recording
    whether it actually contains proof/derivation content."""
    if CITE_PRECISE_RE.search(sentence):
        return "REF-PRECISE", "citation with locator"
    if SUPP_POINTER_RE.search(sentence) and companion_has_proofs is False:
        return ("UNBACKED", "points to the supplementary material, but no "
                "proof/derivation was found there — pass the companion "
                "file so this can be verified, or add one")
    if PROOF_ANCHOR_RE.search(sentence):
        detail = ("anchored in-manuscript" if not
                  SUPP_POINTER_RE.search(sentence) else
                  "verified against proof found in companion document")
        return "PROOF", detail
    if EVIDENCE_ANCHOR_RE.search(sentence):
        return "EVIDENCE", "anchored to figure/table"
    # look in surrounding context (previous / next sentence in chunk)
    if CITE_PRECISE_RE.search(context):
        return "REF-PRECISE", "locator citation in surrounding context"
    if SUPP_POINTER_RE.search(context) and companion_has_proofs is False:
        return ("UNBACKED", "surrounding context points to the "
                "supplementary material, but no proof/derivation was "
                "found there")
    if PROOF_ANCHOR_RE.search(context):
        return "PROOF", "anchor in surrounding context"
    if EVIDENCE_ANCHOR_RE.search(context):
        return "EVIDENCE", "figure/table in surrounding context"
    if MATH_DISPLAY_RE.search(context):
        return "PROOF", "derivation displayed in surrounding context"
    if re.search(r"\b(?:since|because|owing to)\b", sentence,
                 re.IGNORECASE):
        return "INLINE", "carries its own inline justification"
    if CITE_ANY_RE.search(sentence) or CITE_ANY_RE.search(context):
        return "REF-IMPRECISE", ("citation without locator (add "
                                 "[Thm./Sec./p.] to the \\cite)")
    return "UNBACKED", "no proof anchor, citation, or evidence found"


def in_theorem_or_proof(body, pos_line, spans):
    return any(a <= pos_line <= b for a, b in spans)


def theorem_env_start_lines(body):
    """Sorted line numbers where a theorem-like environment begins."""
    lines = []
    for env in THEOREM_ENVS:
        for m in re.finditer(r"\\begin\{" + env + r"\}", body):
            lines.append(body[:m.start()].count("\n") + 1)
    return sorted(lines)


def leads_into_theorem(pline, theorem_lines, paras, max_intervening_paras=2):
    """True if a theorem-like environment begins within
    `max_intervening_paras` paragraphs after this claim sentence's own
    paragraph — i.e. the claim is informal setup prose that the
    following formal statement backs (a common pattern: a
    plain-language walkthrough of a mechanism or bound, followed
    shortly by the \\begin{proposition}/\\begin{theorem} that formalizes
    it, possibly after a short transition paragraph or a blanket
    "proofs are in the supplementary material" aside). Measured in
    paragraphs rather than raw lines so it isn't sensitive to how long
    individual paragraphs happen to be."""
    next_theorem = next((t for t in theorem_lines if t >= pline), None)
    if next_theorem is None:
        return False
    para_starts = [p for p, _ in paras if pline <= p < next_theorem]
    return len(para_starts) <= max_intervening_paras


FORWARD_REF_RE = re.compile(
    r"\bwe (?:prove|show|establish|demonstrate)\b|"
    r"\bcontributions?\b|\bwe (?:instantiate|further present)\b",
    re.IGNORECASE)


def first_theorem_env_line(body):
    """Line number of the first theorem-like environment in the document,
    or None if there isn't one. A claim sentence before this point is
    exempted ONLY if it also carries explicit forward-reference /
    announcement phrasing ("we prove/show/establish that...", or sits in
    a Contributions paragraph) — the paper's own convention of stating a
    result early and proving it later in the body. A claim-signal
    sentence that happens to appear early but does NOT announce a
    later-proved result (e.g. an inline factual assertion in Related
    Work) is not covered by this and is still checked normally."""
    first = None
    for env in THEOREM_ENVS:
        m = re.search(r"\\begin\{" + env + r"\}", body)
        if m:
            line = body[:m.start()].count("\n") + 1
            if first is None or line < first:
                first = line
    return first


def env_line_spans(text, envs):
    spans = []
    for env in envs:
        for m in re.finditer(r"\\begin\{" + env + r"\}.*?\\end\{" + env
                             + r"\}", text, re.DOTALL):
            a = text[:m.start()].count("\n") + 1
            b = text[:m.end()].count("\n") + 1
            spans.append((a, b))
    return spans


# ---------------------------------------------------------------------------
# Theorem-env / proof pairing
# ---------------------------------------------------------------------------

BLANKET_POINTER_RE = re.compile(
    r"(?:complete |all )?proofs? of (?:every|all|each)\b.{0,120}?"
    r"(?:is|are) given in (?:the )?supplementary material"
    r"|(?:proofs?|derivations?) (?:for|of) (?:every|all|each|the "
    r"(?:remaining|other))\b.{0,120}?(?:is|are|can be found)\b.{0,80}?"
    r"supplementary material"
    r"|supplementary material.{0,120}?(?:gives|contains|includes) "
    r"(?:complete |full )?proofs? (?:for|of) (?:every|all|each)",
    re.IGNORECASE | re.DOTALL)


def find_blanket_pointer(body):
    """A pointer phrase stated once, applying to *all* theorem-like
    environments in the document (e.g. "Complete proofs of every
    theorem... are given in the supplementary material"), as opposed to
    a per-result pointer following one specific environment. Returns the
    character offset of the first such phrase, or None."""
    m = BLANKET_POINTER_RE.search(body)
    return m.start() if m else None


def check_theorem_proofs(body, findings, companion_has_proofs=None):
    """`companion_has_proofs`: None if no companion document was loaded
    (a pointer phrase is then trusted at face value, as before);
    True/False if one was loaded, recording whether it actually contains
    proof/derivation content — lets a "deferred to the supplementary
    material" pointer be verified instead of merely trusted."""
    proofs = [m.start() for m in re.finditer(r"\\begin\{proof\}", body)]
    blanket_pos = find_blanket_pointer(body)
    for env in THEOREM_ENVS:
        for m in re.finditer(r"\\begin\{" + env + r"\}", body):
            line = body[:m.start()].count("\n") + 1
            tail = body[m.end():m.end() + 4000]
            has_proof_after = any(p > m.start() for p in proofs)
            pointer = re.search(r"proof (?:is given|can be found|appears|"
                                r"is deferred)|see Appendix|proof omitted",
                                tail, re.IGNORECASE)
            supp_pointer = (SUPP_POINTER_RE.search(tail) or
                            (blanket_pos is not None and
                             blanket_pos < m.start()))
            if has_proof_after or pointer:
                continue
            if supp_pointer:
                if companion_has_proofs is False:
                    findings.append(
                        ("FAIL", line,
                         f"\\begin{{{env}}} points to the supplementary "
                         "material for its proof, but no companion file "
                         "with proof content was found there"))
                continue    # trusted at face value, or verified True
            findings.append(("FAIL", line,
                             f"\\begin{{{env}}} without a subsequent "
                             "proof environment or explicit pointer"))


# ---------------------------------------------------------------------------
# Optional LLM verification of chunks
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are checking a scientific manuscript chunk for unbacked \
claims. A claim is a mathematical or factual assertion (convergence, \
optimality, complexity, equivalence, bounds, prior-work facts). It is BACKED \
if the chunk (i) points to a proof/proposition/derivation in the same \
manuscript, (ii) cites a reference WITH a locator like [Thm. 2], [Sec. V], \
[p. 14], or (iii) is an experimental statement pointing to a figure/table. \
A bare citation without locator is IMPRECISE. Anything else is UNBACKED. \
Output ONLY JSON lines, one per claim: \
{"claim": "<short quote>", "verdict": "BACKED|IMPRECISE|UNBACKED", \
"why": "<one line>"}. Chunk follows:\n\n"""


def llm_check(chunk_text):
    exe = shutil.which("claude")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-p", LLM_PROMPT + chunk_text[:6000]],
            capture_output=True, text=True, timeout=120)
        results = []
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln.startswith("{"):
                try:
                    results.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        return results
    except Exception:                                   # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--chunk-size", type=int, default=3,
                    help="paragraphs per chunk (default 3)")
    ap.add_argument("--stride", type=int, default=2,
                    help="paragraph stride between chunks (default 2; "
                         "stride < chunk-size gives overlap)")
    ap.add_argument("--llm", action="store_true",
                    help="additionally verify chunks with the claude CLI")
    ap.add_argument("--show-backed", action="store_true",
                    help="also list claims that passed")
    args = ap.parse_args()

    # Load every file up front so a claim or theorem in one document (the
    # main paper) can be checked against proof content that actually
    # lives in another (e.g. supplementary.tex), instead of each file
    # being linted in isolation.
    docs = []
    for path in args.files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        body_offset = raw[:re.search(r"\\begin\{document\}", raw).end()
                          ].count("\n") if "\\begin{document}" in raw else 0
        body = prepare_body(raw)
        has_proofs = bool(re.search(r"\\begin\{proof\}", body))
        docs.append({"path": path, "body": body,
                     "body_offset": body_offset, "has_proofs": has_proofs})

    any_fail = False
    for doc in docs:
        path, body, body_offset = doc["path"], doc["body"], doc["body_offset"]
        others = [d for d in docs if d is not doc]
        # None when linting a single file alone (old, trust-the-pointer
        # behaviour); otherwise True iff some *other* loaded file has
        # actual proof content, so a "see supplementary material" pointer
        # gets verified rather than taken on faith.
        companion_has_proofs = (any(d["has_proofs"] for d in others)
                                if others else None)
        paras = split_paragraphs(body)
        chunks = make_chunks(paras, args.chunk_size, args.stride)
        skip_spans = env_line_spans(body, THEOREM_ENVS + ["proof"])
        preview_end_line = first_theorem_env_line(body)
        theorem_lines = theorem_env_start_lines(body)

        print(f"== {path}: {len(paras)} paragraphs, {len(chunks)} "
              f"overlapping chunks ==")
        findings, seen = [], set()
        check_theorem_proofs(body, findings, companion_has_proofs)

        for chunk in chunks:
            sentences = []
            for pline, ptext in chunk:
                for s in split_sentences(ptext):
                    sentences.append((pline, s))
            caption_lines = {pl for pl, pt in chunk
                             if "\\caption{" in pt}
            for idx, (pline, sent) in enumerate(sentences):
                if not CLAIM_RE.search(sent) or EXEMPT_RE.search(sent):
                    continue
                if pline in caption_lines:
                    continue   # captions annotate their own figure
                if in_theorem_or_proof(body, pline, skip_spans):
                    continue    # statements inside thm/proof envs are
                                # covered by the thm-proof pairing check
                if (preview_end_line is not None and pline < preview_end_line
                        and FORWARD_REF_RE.search(sent)):
                    continue    # preview claim before the first formal
                                # result (e.g. Introduction/Contributions
                                # forward-reference); proved later in body
                if leads_into_theorem(pline, theorem_lines, paras):
                    continue    # informal walkthrough immediately
                                # followed by the formal statement/proof
                                # that backs it
                key = hashlib.md5(
                    re.sub(r"\W", "", sent).lower().encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                ctx = " ".join(s for _, s in
                               sentences[max(0, idx - 1):idx + 2])
                verdict, detail = classify(sent, ctx, companion_has_proofs)
                quote = re.sub(r"\s+", " ", sent)[:110]
                if verdict == "UNBACKED":
                    findings.append(("FAIL", pline,
                                     f"unbacked claim: \"{quote}...\" "
                                     f"({detail})"))
                elif verdict == "REF-IMPRECISE":
                    findings.append(("WARN", pline,
                                     f"imprecise reference: \"{quote}...\" "
                                     f"({detail})"))
                elif args.show_backed:
                    findings.append(("OK", pline,
                                     f"[{verdict}] \"{quote}...\""))

            if args.llm:
                chunk_text = "\n\n".join(p for _, p in chunk)
                for r in (llm_check(chunk_text) or []):
                    if r.get("verdict") == "UNBACKED":
                        findings.append(
                            ("FAIL", chunk[0][0],
                             f"[llm] unbacked claim: "
                             f"\"{r.get('claim','')[:90]}\" — "
                             f"{r.get('why','')[:90]}"))
                    elif r.get("verdict") == "IMPRECISE":
                        findings.append(
                            ("WARN", chunk[0][0],
                             f"[llm] imprecise reference: "
                             f"\"{r.get('claim','')[:90]}\""))

        order = {"FAIL": 0, "WARN": 1, "OK": 2}
        findings.sort(key=lambda x: (order[x[0]], x[1]))
        counts = {}
        for level, line, msg in findings:
            counts[level] = counts.get(level, 0) + 1
            print(f"  [{level}] body line ~{line + body_offset}: {msg}")
        print("  summary:", ", ".join(f"{k}={v}"
              for k, v in sorted(counts.items())) or "clean")
        if counts.get("FAIL"):
            any_fail = True
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
