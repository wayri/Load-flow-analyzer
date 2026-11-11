"""Strict editable JSON/CSV dispatch inputs and portable schedule receipts.

case.json contains the complete dataclass representation. Existing table CSVs
override their matching JSON tables; deleting a CSV restores its JSON values.
All saved table files are consequently editable inputs, not ignored previews.
"""
import csv
from dataclasses import MISSING, asdict, fields, is_dataclass
import json
import math
from pathlib import Path
from typing import get_args

from .model import Battery, Case, FlexibleLoad, Generator, Interval, Site


MAX_JSON_BYTES = 32 * 2**20
MAX_CSV_BYTES = 64 * 2**20
_TABLES = {"sites": Site, "batteries": Battery, "generators": Generator,
           "flexible_loads": FlexibleLoad}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Require a JSON file no larger than 32 MiB: {path}")
    def reject(value):
        raise ValueError(f"JSON number must be finite: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_object, parse_constant=reject)


def _construct(cls, value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    allowed = {field.name for field in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    for field in fields(cls):
        if field.name in value and field.type is str and not isinstance(value[field.name], str):
            raise ValueError(f"{label}.{field.name} must be a string")
    try:
        return cls(**value)
    except TypeError as error:
        raise ValueError(f"Invalid {label}: {error}") from error


def case_from_dict(value):
    """Build a validated Case, rejecting unknown fields at every table level."""
    if not isinstance(value, dict):
        raise ValueError("Dispatch case must be a JSON object")
    data = dict(value)
    for name, cls in _TABLES.items():
        if name not in data and name != "sites":
            data[name] = []
        if not isinstance(data.get(name), list):
            raise ValueError(f"{name} must be a list")
        data[name] = [_construct(cls, row, name) for row in data[name]]
    periods = data.get("periods")
    if not isinstance(periods, dict) or any(not isinstance(k, str) or not isinstance(v, list)
                                            for k, v in periods.items()):
        raise ValueError("periods must map each site ID to its interval list")
    data["periods"] = {site: [_construct(Interval, row, "interval") for row in rows]
                       for site, rows in periods.items()}
    return _construct(Case, data, "case").validate()


def _csv_rows(path, expected):
    path = Path(path)
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError(f"Input CSV exceeds 64 MiB: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or set(reader.fieldnames) != set(expected)):
            raise ValueError(f"{path.name}: expected exactly these CSV columns: {','.join(expected)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}:{line}: incorrect column count")
            if any(len(value) > 4096 for value in row.values()):
                raise ValueError(f"{path.name}:{line}: CSV field exceeds 4096 characters")
            yield {key: value.strip() for key, value in row.items()}


def _csv_value(field, value):
    if field.type is str:
        return value
    if value == "":
        if field.default is not MISSING:
            return field.default
        raise ValueError(f"Missing required CSV value: {field.name}")
    try:
        if field.type is int or int in get_args(field.type):
            return int(value)
        if field.type is float or float in get_args(field.type):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("non-finite number")
            return result
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {field.name} CSV value: {value}") from error
    raise ValueError(f"Unsupported CSV field type: {field.name}")


def _table_dict(cls, row):
    return {field.name: _csv_value(field, row[field.name]) for field in fields(cls)}


def load_case(directory):
    directory = Path(directory)
    data = _read_json(directory / "case.json")
    if not isinstance(data, dict):
        raise ValueError("case.json must contain an object")
    # Validate field names before considering table overrides; misspelled JSON
    # configuration must not become silently accepted because a CSV exists.
    unknown = set(data) - {field.name for field in fields(Case)}
    if unknown:
        raise ValueError(f"Unknown case fields: {', '.join(sorted(unknown))}")
    for name, cls in _TABLES.items():
        if name in data:
            if not isinstance(data[name], list):
                raise ValueError(f"{name} must be a list")
            for row in data[name]:
                _construct(cls, row, name)
    if "periods" in data:
        if not isinstance(data["periods"], dict) or any(not isinstance(rows, list) for rows in data["periods"].values()):
            raise ValueError("periods must map site IDs to lists")
        for rows in data["periods"].values():
            for row in rows:
                _construct(Interval, row, "interval")
    for name, cls in _TABLES.items():
        path = directory / (name + ".csv")
        if path.exists():
            data[name] = [_table_dict(cls, row)
                          for row in _csv_rows(path, [field.name for field in fields(cls)])]
    path = directory / "periods.csv"
    if path.exists():
        periods = {}
        for row in _csv_rows(path, ["site", *[field.name for field in fields(Interval)]]):
            site = row.pop("site")
            periods.setdefault(site, []).append(_table_dict(Interval, row))
        data["periods"] = periods
    return case_from_dict(data)


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    # Solver results may contain NumPy scalar/array values. Only numerical
    # conversion is allowed; arbitrary objects are never stringified or executed.
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _write_json(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Preserve existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, default=_json_default, allow_nan=False, indent=2)
        stream.write("\n")


def _write_csv(path, names, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows({key: "" if row.get(key) is None else
                          int(row[key]) if isinstance(row[key], bool) else row[key] for key in names}
                         for row in rows)


def save_case(case, directory):
    case.validate()
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"Input export requires a new or empty directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "case.json", case)
    _write_csv(directory / "periods.csv", ["site", *[field.name for field in fields(Interval)]],
               (dict(site=site, **asdict(row)) for site, rows in case.periods.items() for row in rows))
    for name, cls in _TABLES.items():
        _write_csv(directory / (name + ".csv"), [field.name for field in fields(cls)],
                   (asdict(row) for row in getattr(case, name)))
    return directory


def save_dispatch(dispatch, path):
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path / "dispatch.json"
    _write_json(path, dispatch)
    return path


def load_dispatch(path):
    from .optimizer import Dispatch
    path = Path(path)
    if path.is_dir():
        path = path / "dispatch.json"
    return _construct(Dispatch, _read_json(path), "dispatch")
