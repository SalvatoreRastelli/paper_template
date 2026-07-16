#!/usr/bin/env python3
"""Bibliography accuracy linter.

Checks that each entry in a BibTeX bibliography accurately identifies an
existing work in the literature. Two layers:

OFFLINE (always on): structural/plausibility checks that catch entries
which could not identify any real work — missing required fields,
placeholder text, malformed authors/pages/DOIs/years, duplicate keys,
near-duplicate titles.

ONLINE (--online): resolves each entry against public registries and
compares titles. DOI -> Crossref, arXiv eprint -> arXiv API, otherwise a
Crossref bibliographic search on title+author. Reports entries that do
not resolve or whose best match differs from the claimed title. This is
the layer that catches fabricated or misattributed references.

Input: one or more .bib files, or .tex files containing a
``filecontents`` block with the bibliography (the convention in this
repository).

Usage:
    python3 linters/lint_bib_accuracy.py FILE [FILE ...] [--online]
        [--mailto you@example.org] [--min-similarity 0.75]

Exit code: 1 if any FAIL finding, else 0.
"""

import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

CURRENT_YEAR = 2026

REQUIRED_FIELDS = {
    "article":       ["author", "title", "journal", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "incollection":  ["author", "title", "booktitle", "year"],
    "book":          ["title", "publisher", "year"],   # author OR editor checked separately
    "phdthesis":     ["author", "title", "school", "year"],
    "mastersthesis": ["author", "title", "school", "year"],
    "techreport":    ["author", "title", "institution", "year"],
    "misc":          ["title", "year"],
}
RECOMMENDED_FIELDS = {
    "article": ["volume", "pages"],
    "inproceedings": ["pages"],
}
PLACEHOLDER_RE = re.compile(
    r"XXX+|TODO|FIXME|\?\?\?|placeholder|unknown author|anonymous|"
    r"^\s*authors?\s+et\s+al", re.IGNORECASE)
DOI_RE = re.compile(r"^(https?://(dx\.)?doi\.org/)?10\.\d{4,9}/\S+$")
PAGES_RE = re.compile(r"^\s*(e?[\divxlc]+)\s*(--\s*e?[\divxlc]+)?\s*$", re.IGNORECASE)
ARXIV_OLD_RE = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}$")
ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def extract_bib_text(path):
    """Return bibliography text from a .bib file or the filecontents
    block(s) of a .tex file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if path.endswith(".bib"):
        return text
    blocks = re.findall(
        r"\\begin\{filecontents\*?\}(?:\[[^\]]*\])?\{[^}]*\.bib\}(.*?)"
        r"\\end\{filecontents\*?\}", text, re.DOTALL)
    if blocks:
        return "\n".join(blocks)
    # fall back: maybe the tex embeds thebibliography instead
    if "\\begin{thebibliography}" in text:
        sys.stderr.write(
            f"note: {path} uses an inline thebibliography; this linter "
            "checks BibTeX entries only.\n")
    return ""


def parse_entries(bibtext):
    """Tolerant BibTeX parser: returns list of dicts with keys
    'type', 'key', 'fields' (dict), 'line' (1-based line of the @)."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{", bibtext):
        etype = m.group(1).lower()
        if etype in ("string", "comment", "preamble"):
            continue
        start = m.end()
        depth = 1
        i = start
        while i < len(bibtext) and depth > 0:
            if bibtext[i] == "{":
                depth += 1
            elif bibtext[i] == "}":
                depth -= 1
            i += 1
        body = bibtext[start:i - 1]
        line = bibtext[:m.start()].count("\n") + 1
        keym = re.match(r"\s*([^,\s]+)\s*,", body)
        if not keym:
            continue
        key = keym.group(1)
        fields = {}
        rest = body[keym.end():]
        for fm in re.finditer(r"(\w[\w-]*)\s*=\s*", rest):
            name = fm.group(1).lower()
            j = fm.end()
            if j >= len(rest):
                continue
            if rest[j] == "{":
                depth, k = 1, j + 1
                while k < len(rest) and depth > 0:
                    if rest[k] == "{":
                        depth += 1
                    elif rest[k] == "}":
                        depth -= 1
                    k += 1
                value = rest[j + 1:k - 1]
            elif rest[j] == '"':
                k = rest.find('"', j + 1)
                value = rest[j + 1:k] if k > 0 else ""
            else:
                k = j
                while k < len(rest) and rest[k] not in ",\n":
                    k += 1
                value = rest[j:k]
            fields[name] = re.sub(r"\s+", " ", value).strip()
        entries.append({"type": etype, "key": key, "fields": fields,
                        "line": line})
    return entries


def normalize_title(t):
    t = re.sub(r"<[^>]+>", " ", t)              # strip XML/MathML tags (Crossref titles)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)          # strip latex macros
    t = re.sub(r"[{}$\\]", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# Offline checks
# --------------------------------------------------------------------------

def check_entry_offline(e, findings):
    key, f, etype = e["key"], e["fields"], e["type"]

    def emit(level, msg):
        findings.append((level, key, e["line"], msg))

    required = REQUIRED_FIELDS.get(etype, ["title", "year"])
    for req in required:
        if req not in f or not f[req]:
            emit("FAIL", f"missing required field '{req}' for @{etype}")
    if etype == "book" and "author" not in f and "editor" not in f:
        emit("FAIL", "book entry has neither 'author' nor 'editor'")
    for rec in RECOMMENDED_FIELDS.get(etype, []):
        if rec not in f:
            emit("INFO", f"recommended field '{rec}' missing for @{etype}")

    for name, value in f.items():
        if PLACEHOLDER_RE.search(value):
            emit("FAIL", f"placeholder text in field '{name}': '{value[:60]}'")

    title = f.get("title", "")
    if title and len(normalize_title(title)) < 8:
        emit("WARN", f"suspiciously short title: '{title}'")

    author = f.get("author", "")
    if author:
        if re.search(r"\bet\s+al\b", author, re.IGNORECASE):
            emit("FAIL", "author field contains 'et al.' — list authors "
                         "explicitly or use 'and others'")
        parts = [a.strip() for a in re.split(r"\s+and\s+", author)]
        for a in parts:
            if a.lower() == "others":
                continue
            if a and len(a.split()) == 1 and "," not in a and "{" not in a:
                emit("WARN", f"single-token author name '{a}' — surname "
                             "only, or a typo?")

    year = f.get("year", "")
    if year:
        if not re.match(r"^\d{4}$", year):
            emit("FAIL", f"non-numeric year '{year}'")
        elif not (1900 <= int(year) <= CURRENT_YEAR + 1):
            emit("FAIL", f"implausible year '{year}'")

    eprint = f.get("eprint", "")
    if eprint and not (ARXIV_OLD_RE.match(eprint) or ARXIV_NEW_RE.match(eprint)):
        emit("WARN", f"eprint '{eprint}' does not look like an arXiv id")
    if eprint and ARXIV_NEW_RE.match(eprint) and year:
        yy = int(eprint[:2])
        arxiv_year = 2000 + yy
        if abs(arxiv_year - int(year)) > 1:
            emit("WARN", f"year {year} inconsistent with arXiv id "
                         f"{eprint} (posted {arxiv_year})")

    doi = f.get("doi", "")
    if doi and not DOI_RE.match(doi):
        emit("WARN", f"malformed DOI '{doi}'")

    pages = f.get("pages", "")
    if pages and not PAGES_RE.match(pages):
        if re.match(r"^\s*\d+\s*-\s*\d+\s*$", pages):
            emit("INFO", f"pages '{pages}' uses single hyphen; use '--'")
        else:
            emit("WARN", f"unusual pages field '{pages}'")


def check_global_offline(entries, findings):
    seen_keys = {}
    for e in entries:
        if e["key"] in seen_keys:
            findings.append(("FAIL", e["key"], e["line"],
                             f"duplicate key (first at line "
                             f"{seen_keys[e['key']]})"))
        else:
            seen_keys[e["key"]] = e["line"]
    titles = {}
    for e in entries:
        nt = normalize_title(e["fields"].get("title", ""))
        if not nt:
            continue
        if nt in titles:
            findings.append(("WARN", e["key"], e["line"],
                             f"title duplicates entry '{titles[nt]}' — "
                             "same work under two keys?"))
        else:
            titles[nt] = e["key"]


# --------------------------------------------------------------------------
# Online checks
# --------------------------------------------------------------------------

def http_get_json(url, mailto):
    req = urllib.request.Request(
        url, headers={"User-Agent": f"bib-lint/1.0 (mailto:{mailto})"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def crossref_by_doi(doi, mailto):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    data = http_get_json(url, mailto)
    item = data["message"]
    titles = item.get("title") or [""]
    return titles[0]


def crossref_search(title, author, year, mailto):
    q = urllib.parse.quote(title + " " + author.split(" and ")[0])
    url = (f"https://api.crossref.org/works?query.bibliographic={q}"
           f"&rows=5&select=title,author,issued")
    data = http_get_json(url, mailto)
    return [(it.get("title") or [""])[0]
            for it in data["message"].get("items", [])]


def arxiv_title(eprint, mailto):
    url = ("http://export.arxiv.org/api/query?id_list="
           + urllib.parse.quote(eprint) + "&max_results=1")
    req = urllib.request.Request(
        url, headers={"User-Agent": f"bib-lint/1.0 (mailto:{mailto})"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        tree = ET.parse(resp)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = tree.getroot().find("a:entry", ns)
    if entry is None:
        return None
    t = entry.find("a:title", ns)
    return t.text if t is not None else None


def similarity(a, b):
    return difflib.SequenceMatcher(
        None, normalize_title(a), normalize_title(b)).ratio()


def check_entry_online(e, findings, mailto, min_sim):
    key, f = e["key"], e["fields"]
    title = f.get("title", "")
    if not title:
        return

    def emit(level, msg):
        findings.append((level, key, e["line"], msg))

    try:
        if f.get("doi") and DOI_RE.match(f["doi"]):
            reg_title = crossref_by_doi(f["doi"], mailto)
            sim = similarity(title, reg_title)
            if sim < min_sim:
                emit("FAIL", f"DOI resolves to a different work "
                             f"(similarity {sim:.2f}): '{reg_title[:70]}'")
            else:
                emit("OK", f"DOI verified (similarity {sim:.2f})")
            return
        if f.get("eprint"):
            reg_title = arxiv_title(f["eprint"], mailto)
            if reg_title is None:
                emit("FAIL", f"arXiv id '{f['eprint']}' does not resolve")
                return
            sim = similarity(title, reg_title)
            if sim < min_sim:
                emit("FAIL", f"arXiv id points to a different work "
                             f"(similarity {sim:.2f}): '{reg_title[:70]}'")
            else:
                emit("OK", f"arXiv verified (similarity {sim:.2f})")
            return
        candidates = crossref_search(title, f.get("author", ""),
                                     f.get("year", ""), mailto)
        best = max(((similarity(title, c), c) for c in candidates),
                   default=(0.0, ""))
        if best[0] >= min_sim:
            emit("OK", f"found in Crossref (similarity {best[0]:.2f})")
        elif candidates:
            emit("WARN", f"no close Crossref match (best {best[0]:.2f}: "
                         f"'{best[1][:70]}') — verify manually "
                         "(may be arXiv-only, a book, or misattributed)")
        else:
            emit("WARN", "no Crossref results — verify manually")
    except Exception as exc:                       # noqa: BLE001
        emit("INFO", f"online check failed ({type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--online", action="store_true",
                    help="verify entries against Crossref/arXiv")
    ap.add_argument("--mailto", default="alex.jung@aalto.fi",
                    help="contact email for polite API use")
    ap.add_argument("--min-similarity", type=float, default=0.75)
    args = ap.parse_args()

    any_fail = False
    for path in args.files:
        bibtext = extract_bib_text(path)
        if not bibtext.strip():
            print(f"== {path}: no BibTeX content found ==")
            continue
        entries = parse_entries(bibtext)
        print(f"== {path}: {len(entries)} entries ==")
        findings = []
        check_global_offline(entries, findings)
        for e in entries:
            check_entry_offline(e, findings)
            if args.online:
                check_entry_online(e, findings, args.mailto,
                                   args.min_similarity)
                time.sleep(0.7)
        order = {"FAIL": 0, "WARN": 1, "INFO": 2, "OK": 3}
        findings.sort(key=lambda x: (order[x[0]], x[2]))
        counts = {}
        for level, key, line, msg in findings:
            counts[level] = counts.get(level, 0) + 1
            if level == "OK" and not args.online:
                continue
            print(f"  [{level}] {key} (bib line {line}): {msg}")
        print("  summary:", ", ".join(f"{k}={v}"
              for k, v in sorted(counts.items())) or "clean")
        if counts.get("FAIL"):
            any_fail = True
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
