from __future__ import annotations

import codecs
import csv
import heapq
import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..profile import ArtifactRead, ProfileArchive

PAGE_SIZE_DEFAULT = 20
PAGE_SIZE_MAX = 50
ISSUE_EXAMPLE_LIMIT = 20
_UNSET = object()


class IssueSamples:
    """Keep a bounded set of issue examples while counting every issue."""

    def __init__(self, examples: Iterable[Any] = ()) -> None:
        self.count = 0
        self.examples: list[Any] = []
        self.extend(examples)

    def add(self, issue: Any) -> None:
        self.count += 1
        if len(self.examples) < ISSUE_EXAMPLE_LIMIT:
            self.examples.append(issue)

    append = add

    def extend(self, issues: Iterable[Any]) -> None:
        if isinstance(issues, IssueSamples):
            self.count += issues.count
            room = ISSUE_EXAMPLE_LIMIT - len(self.examples)
            if room > 0:
                self.examples.extend(issues.examples[:room])
            return
        for issue in issues:
            self.add(issue)

    def __bool__(self) -> bool:
        return self.count > 0

    def __iter__(self):
        return iter(self.examples)

    def __len__(self) -> int:
        return self.count


@dataclass(frozen=True)
class CSVScan:
    seen: int
    parsed: int
    rejected: int
    issues: IssueSamples
    missing_columns: bool = False
    fatal_error: bool = False


class DetailRecords:
    def __init__(self, include_records: bool, *, limit: int | None = None, after: tuple[Any, ...] | None = None, key: Any = None):
        self.records: list[dict[str, Any]] | None = [] if include_records else None
        self.count = 0
        self._limit = limit if include_records else None
        self._after = after
        self._eligible_count = 0
        self._last_key: tuple[Any, ...] | None = None
        self._key = key
        self._heap: list[tuple[_ReverseKey, dict[str, Any]]] = []

    def fork(self) -> "DetailRecords":
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        if self.records is not None:
            result.records = list(self.records)
        result._heap = list(self._heap)
        return result

    def append(
        self,
        record: dict[str, Any] | Callable[[], dict[str, Any]],
        *,
        sort_key: Any = _UNSET,
    ) -> None:
        self.count += 1
        if self.records is None:
            return
        if self._limit is None:
            self.records.append(record() if callable(record) else record)
            return
        if sort_key is _UNSET:
            record = record() if callable(record) else record
            sort_key = self._key(record)
        natural_key = (sort_key, self.count)
        if self._after is not None and natural_key <= self._after:
            return
        self._eligible_count += 1
        if len(self._heap) < self._limit:
            heapq.heappush(self._heap, (_ReverseKey(natural_key), record() if callable(record) else record))
        elif natural_key < self._heap[0][0].key:
            heapq.heapreplace(self._heap, (_ReverseKey(natural_key), record() if callable(record) else record))

    def count_only(self) -> None:
        self.count += 1

    def sort(self) -> None:
        if self.records is not None:
            if self._limit is None:
                self.records.sort(key=self._key)
            else:
                page = sorted(self._heap, key=lambda item: item[0].key)
                self.records = [row for _, row in page]
                self._last_key = page[-1][0].key if page else None
                self._heap.clear()

    @property
    def has_more(self) -> bool:
        return self._eligible_count > len(self.records or [])

    @property
    def last_key(self) -> tuple[Any, ...] | None:
        return self._last_key

    def __len__(self) -> int:
        return self.count

    def __bool__(self) -> bool:
        return self.count > 0


class _ReverseKey:
    def __init__(self, key: Any):
        self.key = key

    def __lt__(self, other: "_ReverseKey") -> bool:
        return self.key > other.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReverseKey) and self.key == other.key


def observation(
    name: str,
    value: Any,
    unit: str | None,
    scope: str,
    aggregation: str,
    sample_count: int,
    sources: Iterable[ArtifactRead],
    **extra: Any,
) -> dict[str, Any]:
    item = {
        "name": name,
        "value": value,
        "unit": unit,
        "scope": scope,
        "aggregation": aggregation,
        "sample_count": sample_count,
        "source_refs": [source.source() for source in sources],
    }
    item.update(extra)
    return item


def source_coverage(
    archive: ProfileArchive,
    families: list[tuple[str, list[str], bool]],
    records: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = records or {}
    result = []
    for family, patterns, required in families:
        sources = family_sources(archive, patterns)
        issues = IssueSamples()
        for source in sources:
            if source.status == "missing":
                issues.add({"path": source.path, "problem": "expected source was not indexed"})
            elif source.status == "empty":
                issues.add({"path": source.path, "problem": "file is empty"})
            elif source.status == "invalid_data":
                issues.add({"path": source.path, "problem": source.error or "file is unreadable or failed integrity validation"})
        counts = records.get(family, {})
        issues.extend(counts.get("issues", ()))
        result.append({
            "source_family": family,
            "required": required,
            "files_found": sum(source.status != "missing" for source in sources),
            "files_read": sum(source.status in {"present", "empty"} for source in sources),
            "files_empty": sum(source.status == "empty" for source in sources),
            "files_unreadable": sum(source.status == "invalid_data" for source in sources),
            "records_seen": counts.get("records_seen", 0),
            "records_parsed": counts.get("records_parsed", 0),
            "records_rejected": counts.get("records_rejected", 0),
            "issues": issues.examples,
            "issue_count": issues.count,
            "issue_examples_returned": len(issues.examples),
            "issue_examples_omitted": issues.count - len(issues.examples),
            "source_refs": [source.source() for source in sources],
        })
    return result


def make_result(
    archive: ProfileArchive,
    tuner: str,
    tool: str,
    families: list[tuple[str, list[str], bool]],
    observations: list[dict[str, Any]],
    limitations: list[str],
    *,
    records: dict[str, dict[str, Any]] | None = None,
    detail_records: DetailRecords | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": 2,
        "tool": tool,
        "tuner": tuner,
        "profile": dict(archive.identity),
        "source_coverage": source_coverage(archive, families, records),
        "observations": observations,
        "limitations": limitations,
    }
    if detail_records is not None:
        result["details"] = {
            "total_records": len(detail_records),
            "page_size_default": PAGE_SIZE_DEFAULT,
            "page_size_max": PAGE_SIZE_MAX,
        }
        if detail_records.records is not None:
            result["_detail_records"] = detail_records.records
            result["_detail_last_key"] = detail_records.last_key
            result["_detail_has_more"] = detail_records.has_more
    return result


def json_value(source: ArtifactRead) -> tuple[Any, int, str | None]:
    if source.status != "present" or source.data is None:
        return None, 0, source.error or source.status
    try:
        return json.loads(source.data), 1, None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 1, "invalid JSON document"


def json_array_count(source: ArtifactRead, field: str) -> tuple[int | None, int, str | None]:
    if source.status != "present" or source.data is None:
        return None, 0, source.error or source.status
    try:
        text = source.data.decode(json.detect_encoding(source.data))
        decoder = json.JSONDecoder()
        index = _skip_whitespace(text, 0)
        if index >= len(text) or text[index] != "{":
            json.loads(source.data)
            return None, 1, None
        index += 1
        count = None
        while True:
            index = _skip_whitespace(text, index)
            if index < len(text) and text[index] == "}":
                index += 1
                break
            name, index = decoder.raw_decode(text, index)
            if not isinstance(name, str):
                raise json.JSONDecodeError("object key must be a string", text, index)
            index = _skip_whitespace(text, index)
            if index >= len(text) or text[index] != ":":
                raise json.JSONDecodeError("expected ':' after object key", text, index)
            index = _skip_whitespace(text, index + 1)
            if name != field or index >= len(text) or text[index] != "[":
                _, index = decoder.raw_decode(text, index)
                if name == field:
                    count = None
            else:
                index += 1
                count = 0
                while True:
                    index = _skip_whitespace(text, index)
                    if index < len(text) and text[index] == "]":
                        index += 1
                        break
                    _, index = decoder.raw_decode(text, index)
                    count += 1
                    index = _skip_whitespace(text, index)
                    if index < len(text) and text[index] == ",":
                        index = _skip_whitespace(text, index + 1)
                        if index >= len(text) or text[index] == "]":
                            raise json.JSONDecodeError("expected an array value", text, index)
                    elif index < len(text) and text[index] == "]":
                        index += 1
                        break
                    else:
                        raise json.JSONDecodeError("expected ',' or ']' in array", text, index)
            index = _skip_whitespace(text, index)
            if index < len(text) and text[index] == ",":
                index = _skip_whitespace(text, index + 1)
                if index >= len(text) or text[index] == "}":
                    raise json.JSONDecodeError("expected an object key", text, index)
            elif index < len(text) and text[index] == "}":
                index += 1
                break
            else:
                raise json.JSONDecodeError("expected ',' or '}' in object", text, index)
        if _skip_whitespace(text, index) != len(text):
            raise json.JSONDecodeError("extra data", text, index)
        return count, 1, None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 1, "invalid JSON document"


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _validate_utf8(data: bytes) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")()
    view = memoryview(data)
    for start in range(0, len(view), 64 * 1024):
        decoder.decode(view[start:start + 64 * 1024], final=False)
    decoder.decode(b"", final=True)


def utf8_lines(data: bytes, *, newline: str | None = "\n") -> io.TextIOWrapper:
    _validate_utf8(data)
    return io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", newline=newline)


def jsonl_records(
    source: ArtifactRead,
    consume: Callable[[int, dict[str, Any]], None],
) -> tuple[int, int, IssueSamples]:
    if source.status != "present" or source.data is None:
        return 0, 0, IssueSamples([source.error or source.status])
    try:
        lines = utf8_lines(source.data)
    except UnicodeDecodeError:
        return 1, 0, IssueSamples(["file is not UTF-8"])
    issues = IssueSamples()
    seen = 0
    parsed = 0
    with lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues.add(f"line {line_number}: invalid JSON")
                continue
            if not isinstance(row, dict):
                issues.add(f"line {line_number}: JSONL record is not an object")
                continue
            parsed += 1
            consume(line_number, row)
    return seen, parsed, issues


def csv_records(
    source: ArtifactRead,
    required_columns: set[str],
    consume: Callable[[int, dict[str, str]], None] | None = None,
) -> CSVScan:
    if source.status != "present" or source.data is None:
        return CSVScan(0, 0, 1, IssueSamples([source.error or source.status]), fatal_error=True)

    try:
        # Decode as the CSV reader consumes the stream.  A separate UTF-8
        # validation pass would traverse every byte a second time before the
        # actual CSV parse.
        with io.TextIOWrapper(io.BytesIO(source.data), encoding="utf-8", errors="strict", newline="") as text:
            reader = csv.DictReader(text)
            if not reader.fieldnames:
                return CSVScan(0, 0, 0, IssueSamples(["CSV header is missing"]))
            required_columns_missing = required_columns - set(reader.fieldnames)
            seen = parsed = rejected = 0
            issues = IssueSamples()
            for row in reader:
                seen += 1
                if None in row or any(value is None for value in row.values()):
                    rejected += 1
                    issues.add(f"row {seen}: CSV columns do not match the header")
                else:
                    parsed += 1
                    if not required_columns_missing and consume is not None:
                        # Preserve the physical CSV record number even after
                        # earlier structurally invalid rows.
                        consume(seen, row)
        return CSVScan(seen, parsed, rejected, issues, missing_columns=bool(required_columns_missing))
    except (UnicodeDecodeError, csv.Error):
        return CSVScan(0, 0, 1, IssueSamples(["invalid CSV"]), fatal_error=True)


def add_records(
    stats: dict[str, dict[str, Any]],
    family: str,
    *,
    seen: int = 0,
    parsed: int = 0,
    rejected: int = 0,
    issues: Iterable[str] | IssueSamples = (),
    source: ArtifactRead | None = None,
) -> None:
    item = stats.setdefault(family, {
        "records_seen": 0, "records_parsed": 0, "records_rejected": 0, "issues": IssueSamples(),
    })
    item["records_seen"] += seen
    item["records_parsed"] += parsed
    item["records_rejected"] += rejected
    if isinstance(issues, IssueSamples):
        samples = IssueSamples()
        samples.extend({"path": source.path if source else None, "problem": issue} for issue in issues.examples)
        samples.count = issues.count
        item["issues"].extend(samples)
    else:
        for issue in issues:
            item["issues"].add({"path": source.path if source else None, "problem": issue})


def family_sources(archive: ProfileArchive, patterns: list[str]) -> list[ArtifactRead]:
    found = {}
    for pattern in patterns:
        for source in archive.sources(pattern):
            found[source.path] = source
    if found:
        return [found[path] for path in sorted(found)]
    return [archive.expected_source(patterns[0])]
