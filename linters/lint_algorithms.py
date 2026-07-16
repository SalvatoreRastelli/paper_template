#!/usr/bin/env python3
"""Algorithm formulation linter.

Checks every ``algorithm``/``algorithmic`` environment in a LaTeX file
for a clean, maximally self-contained formulation:

  A. Structure — the algorithm has a \\caption, a \\label, exactly one
     input specification (\\Require) and exactly one output
     specification (\\Ensure).
  B. Input completeness — every mathematical symbol used in a step must
     be (i) declared in \\Require, (ii) assigned by an earlier step
     (left-hand side of \\gets), or (iii) a loop variable of an
     enclosing \\For. Anything else is an undeclared symbol: the step is
     not self-contained.
  C. Output validity — every symbol in \\Ensure must be declared as
     input or produced by some step.
  D. Self-containedness — steps that lean on material outside the
     algorithm (\\eqref/\\ref to equations or sections) are reported, as
     are vague step formulations ("too small", "suitable", ...) that
     leave the operation underspecified.
  E. Input typing — every input symbol should come with a domain or
     type hint (\\in, >, \\geq, \\subseteq, "positive", ...).

Symbols are recognized as LaTeX macro tokens (e.g. \\regparam,
\\localdataset) and as standalone single letters in math mode (e.g. $P$,
$T$). Formatting and math-operator macros are ignored via a whitelist.

Usage:
    python3 linters/lint_algorithms.py FILE.tex [FILE.tex ...]

Exit code: 1 if any FAIL finding, else 0.
"""

import argparse
import re
import sys

# Macros that never denote an algorithm-specific symbol.
IGNORE = {
    # algorithmic / layout (algpseudocode's mixed-case spelling; the
    # `algorithmic` package's ALLCAPS spelling — \STATE, \FOR, \ENDFOR,
    # ... — is matched case-insensitively against this same set, see
    # ALGORITHMIC_LAYOUT below)
    "State", "Statex", "For", "EndFor", "If", "EndIf", "While", "EndWhile",
    "Require", "Ensure", "Return", "Call", "Comment",
    "textbf", "textit", "emph", "mbox", "text", "textrm", "textcolor",
    "hspace", "vspace", "hfill", "quad", "qquad", "parbox", "linewidth",
    "phantomsection", "label", "renewcommand", "alglinenumber",
    "addtocounter", "the", "numexpr", "relax", "footnotesize", "small",
    "algorithmicindent", "item", "caption", "begin", "end",
    # cross references / glossaries
    "eqref", "ref", "cite", "hyperref", "gls", "glspl", "Gls", "Glspl",
    # math operators & symbols
    "sum", "min", "max", "arg", "argmin", "argmax", "frac", "sqrt", "log", "exp",
    "in", "gets", "leftarrow", "forall", "exists", "mid", "colon",
    "ldots", "cdots", "dots", "times", "cdot", "setminus", "cup", "cap",
    "subseteq", "subset", "geq", "leq", "neq", "ll", "gg", "sim",
    "big", "Big", "bigg", "Bigg", "left", "right", "limits",
    "mathrm", "mathcal", "mathbb", "mathbf", "boldsymbol", "bm",
    "normgeneric", "norm", "defeq", "coloneqq", "prime",
    "widetilde", "widehat", "overline", "underline", "hat", "tilde", "bar",
    # counters / indices that are pure loop dummies by convention
    "tau", "ell",
    # set/formatting notation
    "firstnatural", "rm", "bf", "it", "cal",
}
# The `algorithmic` package (as opposed to `algpseudocode`) spells its
# control-flow macros in ALLCAPS: \STATE, \FOR, \ENDFOR, \IF, \ENDIF,
# \WHILE, \ENDWHILE, \RETURN. These never denote algorithm symbols either.
ALGORITHMIC_LAYOUT = {
    "State", "Statex", "For", "EndFor", "If", "EndIf", "While", "EndWhile",
    "Return", "Call", "Comment",
}
ALGO_LAYOUT_UPPER = {s.upper() for s in ALGORITHMIC_LAYOUT}
# Symbols that typically parameterize but are indices bound by loops in
# the surrounding text; still tracked, but only as INFO when undeclared.
SOFT = {"nodeidx", "clusteridx", "sampleidx", "iteridx"}
# Symbols derivable from a declared one (e.g., the neighbourhood is
# determined by the declared graph/edge set).
DERIVED = {
    "neighbourhood": {"graph", "edges"},
    "nodedegree": {"graph", "edges", "neighbourhood"},
    "neib": {"graph", "edges"},
}
# A step whose text contains one of these verbs *introduces* the symbols
# it mentions (natural-language assignment, e.g. "chooses initial
# centroids W^{(i)}_1", "gathers W^{(i')}_t from its neighbors").
INTRODUCER_RE = re.compile(
    r"\b(choose|chooses|store|stores|obtain|obtains|gather|gathers|"
    r"initialize|initializes|receive|receives|draw|draws|compute|"
    r"computes)\b", re.IGNORECASE)

VAGUE = [
    "too small", "too large", "suitable", "appropriate", "properly",
    "as needed", "if necessary", "sufficiently", "roughly", "somehow",
    "etc.",
]
DOMAIN_HINTS = re.compile(
    r"\\in\b|\\geq|\\leq|>|<|\\subseteq|positive|integer|natural|"
    r"max\.|nr\.|number", re.IGNORECASE)

MACRO_RE = re.compile(r"\\([A-Za-z]+)")
# standalone single capital letters inside math (rough heuristic);
# exclude letters glued to digits or hyphens (step labels like "Alg1-E2")
# and the roman numerals I/V used to label formula components.
SINGLE_RE = re.compile(r"(?<![A-Za-z\\{\-0-9])([A-HJ-UW-Z])(?![A-Za-z0-9}])")


def find_algorithms(text):
    """Yield (caption, label, body, start_line) per algorithm env."""
    for m in re.finditer(
            r"\\begin\{algorithm\}(.*?)\\end\{algorithm\}", text, re.DOTALL):
        body = m.group(1)
        line = text[:m.start()].count("\n") + 1
        cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body)
        lab = re.search(r"\\label\{([^}]*)\}", body)
        yield (cap.group(1) if cap else None,
               lab.group(1) if lab else None, body, line)


COMPOSITE_RE = re.compile(
    r"\\([A-Za-z]+)_\{\\(?:mathrm|text|rm)\{([A-Za-z]+)\}\}")


def symbols_in(text):
    """Macro-token symbols + standalone capital letters. A macro with a
    *textual* subscript (e.g. \\mW_{\\mathrm{in}}) is a distinct named
    variable and tracked as the composite 'name_sub'; numeric or index
    subscripts collapse to the base symbol."""
    syms = set()
    for m in COMPOSITE_RE.finditer(text):
        base, sub = m.group(1), m.group(2)
        if base not in IGNORE:
            syms.add(f"{base}_{sub}")
    plain = COMPOSITE_RE.sub(" ", text)
    for m in MACRO_RE.finditer(plain):
        name = m.group(1)
        if name not in IGNORE and name.upper() not in ALGO_LAYOUT_UPPER:
            syms.add(name)
    for m in SINGLE_RE.finditer(re.sub(r"\\[A-Za-z]+", " ", plain)):
        syms.add(m.group(1))
    return syms


def split_segments(body):
    """Return (require_text, ensure_text, steps) where steps is a list
    of (relative_line, text) for \\State/\\Statex/\\For lines.

    Recognizes two input/output conventions: the \\Require/\\Ensure
    macros (algpseudocode-style), and a plain
    \\STATE \\textbf{Input:}/\\textbf{Output:} line (algorithmic-style,
    since \\Require/\\Ensure aren't defined by the `algorithmic` package
    itself). Whichever convention starts an algorithm's segment governs
    it; the two are not mixed within one algorithm."""
    lines = body.split("\n")
    require, ensure, steps = [], [], []
    mode = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("\\Require"):
            mode = "req"
            require.append(stripped[len("\\Require"):])
            continue
        if stripped.startswith("\\Ensure"):
            mode = "ens"
            ensure.append(stripped[len("\\Ensure"):])
            continue
        state_io = re.match(
            r"\\STATE\s*\\textbf\{(Input|Output)s?:?\}\s*(.*)", stripped,
            re.IGNORECASE)
        if state_io:
            kind, rest = state_io.groups()
            if kind.lower() == "input":
                mode = "req"
                require.append(rest)
            else:
                mode = "ens"
                ensure.append(rest)
            continue
        if re.match(r"\\(State|Statex|For|If|While|Return)", stripped,
                    re.IGNORECASE):
            mode = "step"
            steps.append((i, stripped))
            continue
        if mode == "req":
            require.append(stripped)
        elif mode == "ens":
            ensure.append(stripped)
        elif mode == "step" and stripped and not stripped.startswith("\\end"):
            # continuation of a multi-line step: merge into previous
            line0, text0 = steps[-1]
            steps[-1] = (line0, text0 + " " + stripped)
    return " ".join(require), " ".join(ensure), steps


def lint_algorithm(caption, label, body, base_line, findings):
    name = label or (caption[:40] + "..." if caption else f"line {base_line}")

    def emit(level, msg, rel=0):
        findings.append((level, name, base_line + rel, msg))

    # A. structure
    if not caption:
        emit("FAIL", "algorithm has no \\caption")
    if not label:
        emit("WARN", "algorithm has no \\label")
    n_req = (len(re.findall(r"\\Require\b", body))
             + len(re.findall(r"\\STATE\s*\\textbf\{Inputs?:?\}", body,
                              re.IGNORECASE)))
    n_ens = (len(re.findall(r"\\Ensure\b", body))
             + len(re.findall(r"\\STATE\s*\\textbf\{Outputs?:?\}", body,
                              re.IGNORECASE)))
    if n_req != 1:
        emit("FAIL", "expected exactly one \\Require or "
                     f"\\STATE \\textbf{{Input:}} (input spec), found {n_req}")
    if n_ens != 1:
        emit("FAIL", "expected exactly one \\Ensure or "
                     f"\\STATE \\textbf{{Output:}} (output spec), found {n_ens}")
    if n_req == 0 or n_ens == 0:
        return

    require, ensure, steps = split_segments(body)
    declared = symbols_in(require)
    known = set(declared)
    # symbols derivable from declared ones
    for der, sources in DERIVED.items():
        if sources & known:
            known.add(der)

    # B. definition-before-use over the steps
    for rel, step in steps:
        # loop variables become known
        for lv in re.findall(r"\\For\{\$?\\?([A-Za-z]+)", step, re.IGNORECASE):
            known.add(lv)
        used = symbols_in(step)
        # left-hand sides of \gets become known AFTER this step,
        # but flag use-before-def within the same scan order.
        lhs = set()
        for gm in re.finditer(r"([^$]{0,80}?)\\gets", step):
            lhs |= symbols_in(gm.group(1))
        # natural-language introductions and \Return lines introduce
        # (rather than consume) the symbols they mention
        if INTRODUCER_RE.search(step) or "\\Return" in step:
            known |= used
            continue
        unknown = {s for s in used - known - lhs
                   if s not in SOFT and len(s) > 1}
        soft_unknown = {s for s in used - known - lhs if s in SOFT}
        for s in sorted(unknown):
            emit("FAIL", f"step uses undeclared symbol '\\{s}' — not in "
                         "\\Require, not assigned earlier; step is not "
                         "self-contained", rel)
        for s in sorted(soft_unknown):
            emit("INFO", f"index symbol '\\{s}' used without local "
                         "binding (bound outside the algorithm?)", rel)
        single_unknown = {s for s in used - known - lhs if len(s) == 1}
        anchored = bool(re.search(r"\\eqref\{", step))
        for s in sorted(single_unknown):
            emit("INFO" if anchored else "WARN",
                 f"step uses symbol '{s}' that is neither an input nor "
                 "assigned earlier"
                 + (" (anchored by an \\eqref in the same step)"
                    if anchored else ""), rel)
        known |= lhs

        # D. self-containedness of individual steps
        for xr in re.findall(r"\\eqref\{([^}]*)\}", step):
            emit("INFO", f"step refers to external equation ({xr}); "
                         "consider restating the expression inside the "
                         "algorithm", rel)
        low = step.lower()
        for phrase in VAGUE:
            if phrase in low:
                emit("WARN", f"vague step formulation: '...{phrase}...' — "
                             "specify a precise criterion", rel)

    # C. output validity
    out_syms = {s for s in symbols_in(ensure) if len(s) > 1 and s not in SOFT}
    for s in sorted(out_syms - known):
        emit("FAIL", f"output symbol '\\{s}' in \\Ensure is never "
                     "declared or produced by any step")

    # E. input typing
    for chunk in re.split(r"[;.]", require):
        syms = {s for s in symbols_in(chunk) if len(s) > 1 and s not in SOFT}
        if syms and not DOMAIN_HINTS.search(chunk):
            pretty = ", ".join("\\" + s for s in sorted(syms))
            emit("INFO", f"input(s) {pretty} declared without a "
                         "domain/type hint")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    any_fail = False
    for path in args.files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        algos = list(find_algorithms(text))
        print(f"== {path}: {len(algos)} algorithm environment(s) ==")
        findings = []
        for caption, label, body, line in algos:
            lint_algorithm(caption, label, body, line, findings)
        order = {"FAIL": 0, "WARN": 1, "INFO": 2}
        findings.sort(key=lambda x: (x[1], order[x[0]], x[2]))
        counts = {}
        for level, name, line, msg in findings:
            counts[level] = counts.get(level, 0) + 1
            print(f"  [{level}] {name} (line {line}): {msg}")
        print("  summary:", ", ".join(f"{k}={v}"
              for k, v in sorted(counts.items())) or "clean")
        if counts.get("FAIL"):
            any_fail = True
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
