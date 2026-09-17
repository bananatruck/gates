"""The one component that touches the network, tested without touching it.

D41 puts the resolver in the adapter rather than ``rig/``: adapters are where
host knowledge lives, and ``rig/`` is the model-free scenario loop and should
never open a socket. ``gates/`` never opens one either, which is why the resolver
arrives injected (B2).

The live test at the bottom is the only one that reaches arXiv, and it is skipped
unless ``GATES_LIVE_ARXIV`` is set, so the default suite stays hermetic and
offline.
"""

from __future__ import annotations

import json
import os

import pytest

from gates.adapters.agentlab import arxiv_lookup
from gates.schema import PaperRecord

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2410.21676v4</id>
    <title>Simplifying Graph Convolutional Networks</title>
    <published>2019-02-19T15:26:05Z</published>
    <author><name>Felix Wu</name></author>
    <author><name>Tianyi Zhang</name></author>
  </entry>
</feed>
"""

EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
    >0</opensearch:totalResults>
</feed>
"""


def fetcher(*responses, record=None):
    """A stand-in for the HTTP call. Records every URL it was given."""
    queue = list(responses)

    def fetch(url):
        if record is not None:
            record.append(url)
        if not queue:
            raise AssertionError(f"unexpected second fetch: {url}")
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return fetch


def test_an_identifier_resolves_to_a_record_with_its_provenance(tmp_path):
    lookup = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(ATOM))
    record = lookup("2410.21676")
    assert isinstance(record, PaperRecord)
    assert record.title == "Simplifying Graph Convolutional Networks"
    assert record.authors == ("Felix Wu", "Tianyi Zhang")
    assert record.year == 2019
    assert record.locator == "http://arxiv.org/abs/2410.21676v4"
    assert record.content_hash


def test_an_identifier_arxiv_does_not_know_resolves_to_none(tmp_path):
    """None is a verdict: no such paper. It is not the same as a failed fetch,
    which raises so the gate can report that citations went unchecked."""
    lookup = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(EMPTY))
    assert lookup("2501.00001") is None


def test_a_second_call_is_served_from_the_cache(tmp_path):
    """arXiv asks for about one request every three seconds, so a manuscript
    citing the same paper twice must not cost two requests."""
    urls: list[str] = []
    lookup = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(ATOM, record=urls))
    first = lookup("2410.21676")
    second = lookup("2410.21676")
    assert first == second
    assert len(urls) == 1


def test_the_cache_survives_a_new_resolver(tmp_path):
    """The cache is on disk under .cache/, so a later run does not refetch."""
    arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(ATOM))("2410.21676")
    reopened = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher())
    assert reopened("2410.21676").title.startswith("Simplifying")


def test_a_cached_absence_is_remembered_too(tmp_path):
    """Otherwise every fabricated citation costs a request on every attempt of
    the loop, which is exactly the case the loop retries."""
    arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(EMPTY))("2501.00001")
    reopened = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher())
    assert reopened("2501.00001") is None


def test_a_failed_fetch_raises_rather_than_reporting_no_such_paper(tmp_path):
    """The distinction the whole design rests on. Returning None here would make
    an outage look like a fabricated citation and reject an honest manuscript."""
    lookup = arxiv_lookup(
        cache_dir=str(tmp_path), fetch=fetcher(OSError("connection refused"))
    )
    with pytest.raises(OSError):
        lookup("2410.21676")


def test_a_failed_fetch_is_not_cached(tmp_path):
    """A cached outage would keep the citations unchecked long after the network
    came back."""
    lookup = arxiv_lookup(
        cache_dir=str(tmp_path), fetch=fetcher(OSError("down"), ATOM)
    )
    with pytest.raises(OSError):
        lookup("2410.21676")
    assert lookup("2410.21676").year == 2019


def test_the_version_is_stripped_before_the_request(tmp_path):
    """D26. Asking for a specific version would make a v4 citation of a paper the
    run read as v2 look like a different paper."""
    urls: list[str] = []
    lookup = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(ATOM, record=urls))
    lookup("2410.21676v4")
    assert "2410.21676" in urls[0] and "v4" not in urls[0]


def test_malformed_xml_raises_rather_than_resolving(tmp_path):
    """A truncated response is a failed fetch, not an absent paper."""
    lookup = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher("<feed><entry"))
    with pytest.raises(Exception):
        lookup("2410.21676")


def test_an_unreadable_cache_file_is_rebuilt_rather_than_raising(tmp_path):
    """The cache is a convenience. A truncated write from an earlier run must not
    take the resolver, and through it the whole writing phase, down with it."""
    (tmp_path / "records.json").write_text("{not json", encoding="utf-8")
    lookup = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(ATOM))
    assert lookup("2410.21676").year == 2019


def test_a_cache_row_missing_its_fields_is_refetched(tmp_path):
    """A row this resolver cannot turn into a PaperRecord is a cache miss, not an
    error. Raising here reaches the gate as "citations went unchecked", which
    would hide a corrupt cache behind an outage message."""
    (tmp_path / "records.json").write_text('{"2410.21676": {"nonsense": 1}}', encoding="utf-8")
    lookup = arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(ATOM))
    assert lookup("2410.21676").title.startswith("Simplifying")


def test_the_cache_file_is_readable_by_a_human(tmp_path):
    """It is evidence. An operator has to be able to see what was resolved."""
    arxiv_lookup(cache_dir=str(tmp_path), fetch=fetcher(ATOM))("2410.21676")
    cached = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert cached["2410.21676"]["title"].startswith("Simplifying")


@pytest.mark.skipif(
    not os.environ.get("GATES_LIVE_ARXIV"),
    reason="live arXiv call; set GATES_LIVE_ARXIV=1 to run",
)
def test_the_resolver_works_against_the_real_arxiv(tmp_path):
    """D41 allows exactly one live test. A resolver never run against the real
    API is a check with no evidence behind it, and the Atom fixtures above are
    only as good as my reading of the format."""
    lookup = arxiv_lookup(cache_dir=str(tmp_path))
    record = lookup("1902.07153")
    assert record is not None
    assert "Simplifying Graph Convolutional Networks" in record.title
    assert lookup("2501.99999") is None
