"""Resolve a cited arXiv id to a real paper, for Gate 3.

An adapter module rather than part of ``gates/``, because ``gates/`` never opens
a socket (D41). Shared by every host: the gate only ever sees the returned
function.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any
from collections.abc import Callable
from xml.etree import ElementTree

from ..schema import PaperRecord


#: arXiv's Atom API. The one endpoint anything in this project contacts.
_ARXIV_API = "http://export.arxiv.org/api/query?id_list={}&max_results=1"


#: arXiv asks for roughly one request every three seconds. Enforced between
#: requests rather than as a per-call sleep, so a cache hit costs nothing.
_ARXIV_MIN_INTERVAL_S = 3.0


_ATOM = "{http://www.w3.org/2005/Atom}"


#: Trailing version, stripped before the request (D26). Asking arXiv for a
#: specific version would make a v4 citation of a paper the run read as v2 look
#: like a different paper, which is the registry check's question, not this one's.
_ARXIV_VERSION = re.compile(r"v\d+$")


def _fetch_url(url: str) -> str:
    """One GET, stdlib only. Raises on anything that is not a 200 with a body.

    Separated so the resolver can be tested without a socket: every test above
    passes its own ``fetch``, and this is what the real one does.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "gates-validity-layer (citation resolution)"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def arxiv_lookup(
    *,
    cache_dir: str = ".cache/arxiv",
    fetch: Callable[[str], str] = _fetch_url,
) -> Callable[[str], PaperRecord | None]:
    """A resolver for ``Gate3Config.lookup``, backed by arXiv and a disk cache.

    Lives in the adapter by D41, because ``gates/`` never opens a socket and
    ``rig/`` is the model-free scenario loop. The gate receives only the returned
    function, so it cannot tell arXiv from the dict-backed fake the suite uses.

    Three behaviours the gate depends on:

    * ``None`` means arXiv has no such paper. That is a verdict.
    * **Raising** means the question could not be asked. Gate 3 turns that into
      an INFO row saying citations went unchecked, never a rejection: an outage
      is not a defect in the manuscript.
    * A resolved *and* an absent answer are both cached; a failure is not. A
      cached outage would keep citations unchecked after the network returned,
      and an uncached absence would cost one request per fabricated citation on
      every turn of the retry loop, which is the case the loop exists to retry.
    """
    path = Path(cache_dir) / "records.json"
    cache: dict[str, dict[str, Any] | None] = {}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # The cache is a convenience. A truncated write from an earlier run must
        # not take the resolver down, and through it the whole writing phase.
        cache = {}
    last_request: list[float | None] = [None]

    def lookup(identifier: str) -> PaperRecord | None:
        key = _ARXIV_VERSION.sub("", identifier.strip())
        if key in cache:
            row = cache[key]
            if row is None:
                return None
            record = _as_record(row)
            if record is not None:
                return record
            # A row this cannot read is a miss, not an error. Raising would
            # reach Gate 3 as "citations went unchecked", which would hide a
            # corrupt cache behind an outage message.

        if last_request[0] is not None:
            wait = _ARXIV_MIN_INTERVAL_S - (time.monotonic() - last_request[0])
            if wait > 0:
                time.sleep(wait)
        body = fetch(_ARXIV_API.format(key))
        last_request[0] = time.monotonic()

        entry = ElementTree.fromstring(body).find(f"{_ATOM}entry")
        cache[key] = None if entry is None else _parse_entry(entry, key, body)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            # An unwritable cache costs requests, not correctness.
            pass
        row = cache[key]
        return None if row is None else _as_record(row)

    return lookup


def _parse_entry(entry: Any, key: str, body: str) -> dict[str, Any]:
    """One Atom entry as a cache row. Missing fields stay empty, never guessed."""
    published = (entry.findtext(f"{_ATOM}published") or "").strip()
    return {
        "identifier": key,
        "title": " ".join((entry.findtext(f"{_ATOM}title") or "").split()),
        "authors": [
            " ".join((a.findtext(f"{_ATOM}name") or "").split())
            for a in entry.findall(f"{_ATOM}author")
        ],
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "locator": (entry.findtext(f"{_ATOM}id") or "").strip(),
        # What was retrieved, so "the same paper" is checkable later. Hashing the
        # response rather than the parsed fields catches a metadata correction.
        "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def _as_record(row: Any) -> PaperRecord | None:
    """A cache row as a record, or ``None`` if the row cannot be read as one.

    Returns rather than raises so a corrupt cache is a miss and gets refetched.
    A raise here would surface in Gate 3 as an unreachable resolver, which would
    report citations as unchecked when the network was fine.
    """
    if not isinstance(row, dict) or not isinstance(row.get("title"), str):
        return None
    year = row.get("year")
    return PaperRecord(
        identifier=str(row.get("identifier") or ""),
        title=row["title"],
        authors=tuple(row.get("authors") or ()),
        year=year if isinstance(year, int) else None,
        locator=str(row.get("locator") or ""),
        content_hash=str(row.get("content_hash") or ""),
    )
