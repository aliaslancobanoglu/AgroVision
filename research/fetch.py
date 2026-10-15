"""
AgroVision AI — weekly research digest.

Collects the last N days (default 7) of:
  - Ultralytics GitHub releases
  - arXiv papers for a fixed set of project-relevant queries

and writes research/digest_YYYY-MM-DD.md with title, link and a one-line
summary per item. Standard library only, so it runs outside the .venv too.

Usage
-----
    python research/fetch.py
    python research/fetch.py --days 14
    python research/fetch.py --out some/dir

Set GITHUB_TOKEN to lift GitHub's 60 requests/hour unauthenticated limit
(one request per run, so normally not needed).
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

GITHUB_RELEASES_URL = "https://api.github.com/repos/ultralytics/ultralytics/releases?per_page=100"
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Label -> arXiv search_query. Quoted strings are exact phrases; the TensorRT
# query is an AND of terms because the exact phrase almost never appears.
ARXIV_QUERIES = {
    "weed detection": 'all:"weed detection"',
    "precision agriculture spraying": 'all:"precision agriculture" AND all:spraying',
    "small object detection": 'all:"small object detection"',
    "edge inference TensorRT": "all:edge AND all:inference AND all:TensorRT",
}
ARXIV_MAX_RESULTS = 50
ARXIV_DELAY_S = 3  # arXiv API terms of use: one request every 3 seconds

ATOM = {"a": "http://www.w3.org/2005/Atom"}
USER_AGENT = "AgroVision-research-digest/1.0"
SUMMARY_MAX_CHARS = 200


def http_get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def one_line(text, max_chars=SUMMARY_MAX_CHARS):
    """Collapse whitespace, keep the first sentence, truncate."""
    text = re.sub(r"\s+", " ", text or "").strip()
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    if match:
        text = match.group(1)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def md_escape(text):
    return text.replace("[", r"\[").replace("]", r"\]")


def fetch_ultralytics_releases(since):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    items = []
    page = 1
    while True:
        releases = json.loads(http_get(f"{GITHUB_RELEASES_URL}&page={page}", headers))
        if not releases:
            break
        page_items, reached_end = _parse_releases(releases, since)
        items += page_items
        if reached_end:
            break
        page += 1
    return items


def _parse_releases(releases, since):
    """Returns (items within window, whether an older release was seen)."""
    items = []
    reached_end = False
    for rel in releases:
        if rel.get("draft") or not rel.get("published_at"):
            continue
        published = datetime.fromisoformat(rel["published_at"].replace("Z", "+00:00"))
        if published < since:
            reached_end = True
            continue
        items.append({
            "title": rel.get("name") or rel["tag_name"],
            "link": rel["html_url"],
            "date": published,
            "summary": release_summary(rel.get("body") or ""),
        })
    return items, reached_end


def release_summary(body):
    """First substantive line of a release body (skips headings, blank lines, images)."""
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "![", "<", "---")):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)                     # list bullet
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)       # [text](url) -> text
        line = re.sub(r"[*_`]", "", line)                          # emphasis / code
        line = re.sub(r"^[^\w(\"']+", "", line)                    # leading emoji
        if line:
            return one_line(line)
    return "(no release notes)"


def fetch_arxiv(search_query, since):
    """Pages through results (newest first) until one is older than `since`."""
    items = []
    start = 0
    while True:
        if start:
            time.sleep(ARXIV_DELAY_S)
        params = urllib.parse.urlencode({
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start,
            "max_results": ARXIV_MAX_RESULTS,
        })
        root = ET.fromstring(http_get(f"{ARXIV_API_URL}?{params}", timeout=60))
        entries = root.findall("a:entry", ATOM)
        page_items = _parse_arxiv_entries(entries, since)
        items += page_items
        if len(page_items) < len(entries) or len(entries) < ARXIV_MAX_RESULTS:
            return items
        start += ARXIV_MAX_RESULTS


def _parse_arxiv_entries(entries, since):
    items = []
    for entry in entries:
        published = datetime.fromisoformat(
            entry.findtext("a:published", "", ATOM).replace("Z", "+00:00"))
        if published < since:
            break  # sorted newest first
        items.append({
            "id": entry.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1].split("v")[0],
            "title": re.sub(r"\s+", " ", entry.findtext("a:title", "", ATOM)).strip(),
            "link": entry.findtext("a:id", "", ATOM).strip(),
            "date": published,
            "summary": one_line(entry.findtext("a:summary", "", ATOM)),
        })
    return items


def render_item(item):
    return (f"- **[{md_escape(item['title'])}]({item['link']})** "
            f"({item['date']:%Y-%m-%d}) — {item['summary']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7, help="look-back window (default 7)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                        help="output directory (default: research/)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)
    errors = []
    sections = []

    print("Fetching Ultralytics releases...")
    try:
        releases = fetch_ultralytics_releases(since)
    except Exception as exc:
        releases = []
        errors.append(f"Ultralytics releases: {exc}")
    sections.append(("Ultralytics releases", releases))

    seen = set()
    for i, (label, query) in enumerate(ARXIV_QUERIES.items()):
        if i:
            time.sleep(ARXIV_DELAY_S)
        print(f"Fetching arXiv: {label}...")
        try:
            papers = fetch_arxiv(query, since)
        except Exception as exc:
            papers = []
            errors.append(f"arXiv '{label}': {exc}")
        # A paper matching several queries is listed only under the first one.
        papers = [p for p in papers if p["id"] not in seen]
        seen.update(p["id"] for p in papers)
        sections.append((f"arXiv — {label}", papers))

    lines = [
        f"# Research digest — {now:%Y-%m-%d}",
        "",
        f"Window: {since:%Y-%m-%d} → {now:%Y-%m-%d} (last {args.days} days, UTC)",
        "",
    ]
    for title, items in sections:
        lines += [f"## {title} ({len(items)})", ""]
        lines += [render_item(it) for it in items] or ["_Nothing new._"]
        lines.append("")
    if errors:
        lines += ["## Fetch errors", ""] + [f"- {e}" for e in errors] + [""]

    args.out.mkdir(parents=True, exist_ok=True)
    # Non-default windows get a suffix so they don't overwrite the weekly digest.
    suffix = "" if args.days == 7 else f"_{args.days}d"
    out_path = args.out / f"digest_{now:%Y-%m-%d}{suffix}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    total = sum(len(items) for _, items in sections)
    print(f"Wrote {out_path} ({total} items, {len(errors)} errors)")
    return 1 if len(errors) == len(sections) else 0


if __name__ == "__main__":
    sys.exit(main())
