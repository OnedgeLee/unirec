import base64
import datetime as dt
import hashlib
import json
import math
import uuid
import numpy as np
import torch
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple, ClassVar
from torch import Tensor
from packaging.version import Version
from .version import Versioned

__all__ = ["Fingerprint", "Fingerprinter", "fingerprint_field", "fingerprint_property"]

FPItem = Tuple[str, Any]
FPTransform = Callable[[Any], Any]
_FINGERPRINT_META_KEY = "fingerprint_rule"


@dataclass(frozen=True, slots=True)
class _FingerprintRule:
    name: str | None = None
    transform: FPTransform | None = None


def fingerprint_field(
    *,
    name: str | None = None,
    transform: FPTransform | None = None,
    **kw: Any,
):
    meta = dict(kw.pop("metadata", {}) or {})
    meta[_FINGERPRINT_META_KEY] = _FingerprintRule(name=name, transform=transform)
    return field(metadata=meta, **kw)


def fingerprint_property(
    _obj: Any | None = None,
    *,
    name: str | None = None,
    transform: FPTransform | None = None,
    doc: str | None = None,
):
    rule = _FingerprintRule(name=name, transform=transform)

    def _apply_to_function(fget):
        setattr(fget, "__fingerprint_rule__", rule)
        return property(fget, doc=doc)

    def _apply_to_property(prop: property):
        if not isinstance(prop, property) or prop.fget is None:
            raise TypeError("@fingerprint_property must decorate a property or fget")
        setattr(prop.fget, "__fingerprint_rule__", rule)
        return property(
            prop.fget, prop.fset, prop.fdel, doc if doc is not None else prop.__doc__
        )

    if _obj is None:

        def _decorator(obj):
            return (
                _apply_to_property(obj)
                if isinstance(obj, property)
                else _apply_to_function(obj)
            )

        return _decorator

    return (
        _apply_to_property(_obj)
        if isinstance(_obj, property)
        else _apply_to_function(_obj)
    )


@dataclass(frozen=True, slots=True)
class Fingerprint:
    json_str: str
    algo: str
    hash_len: int
    fqn: str
    cls_version: str | None = None

    def __post_init__(self) -> None:
        if self.hash_len < 8:
            raise ValueError("hash_len must be >= 8")
        if not self.fqn:
            raise ValueError("fqn must be non-empty")

    @cached_property
    def digest(self) -> str:
        h = hashlib.new(self.algo)
        h.update(self.json_str.encode("utf-8"))
        return h.hexdigest()[: self.hash_len]

    def key(self) -> str:
        return f"{self.algo}:{self.hash_len}:{self.digest}"

    def as_pairs(self) -> list[tuple[str, Any]]:
        raw = json.loads(self.json_str)
        out: list[tuple[str, Any]] = []
        for kv in raw:
            if isinstance(kv, (list, tuple)) and len(kv) == 2:
                k, v = kv
                out.append((str(k), v))
        return out

    @staticmethod
    def from_pairs(
        *,
        head_pairs: Iterable[FPItem],
        attr_pairs_sorted: Iterable[FPItem],
        algo: str,
        hash_len: int,
        fqn: str,
        cls_version: str | None,
    ) -> "Fingerprint":
        payload = list(head_pairs)
        payload.extend(attr_pairs_sorted)
        json_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return Fingerprint(
            json_str=json_str,
            algo=algo,
            hash_len=hash_len,
            fqn=fqn,
            cls_version=cls_version,
        )

    @staticmethod
    def from_json(
        json_str: str, *, require_version: str | None = None
    ) -> "Fingerprint":
        try:
            items: list[tuple[str, Any]] = json.loads(json_str)
        except Exception as e:
            raise ValueError("invalid fingerprint JSON") from e

        head = dict(items[:5])
        fqn = head.get("__fqn__")
        algo = head.get("__hash_algo__")
        hash_len = head.get("__hash_len__")
        fp_ver = head.get("__fingerprint_ver__")
        cls_ver = head.get("__cls_version__", None)

        if not isinstance(fqn, str) or not fqn:
            raise ValueError("missing __fqn__")
        if not isinstance(algo, str) or not algo:
            raise ValueError("missing __hash_algo__")
        if not isinstance(hash_len, int) or hash_len < 8:
            raise ValueError("missing/invalid __hash_len__")
        if not isinstance(fp_ver, str) or not fp_ver:
            raise ValueError("missing __fingerprint_ver__")
        if require_version is not None and fp_ver != require_version:
            raise ValueError(
                f"incompatible fingerprint version: {fp_ver} != {require_version}"
            )

        return Fingerprint(
            json_str=json_str,
            algo=algo,
            hash_len=hash_len,
            fqn=fqn,
            cls_version=cls_ver if isinstance(cls_ver, str) else None,
        )


@dataclass(slots=True)
class Fingerprinter(Versioned):
    VERSION: ClassVar[Version] = Version("0.0.0")
    FLOAT_DECIMALS: ClassVar[int] = 6
    _HEAD_KEYS: ClassVar[tuple[str, ...]] = (
        "__fqn__",
        "__hash_algo__",
        "__hash_len__",
        "__fingerprint_ver__",
        "__cls_version__",
    )

    hash_algo: str = "blake2s"
    hash_len: int = 16

    def make(
        self,
        obj: Any,
        *,
        attrs: Sequence[str] | None = None,
        extra: Iterable[FPItem] = (),
        transforms: Mapping[str, FPTransform] | None = None,
        strict_attrs: bool = False,
        auto_discover: bool = True,
    ) -> Fingerprint:
        self._guard_len(self.hash_len)

        if attrs is None and auto_discover:
            base_pairs = list(self._discover(obj))
        else:
            base_pairs = []

        attr_items = self._collect_attrs(
            obj,
            attrs=attrs,
            base_pairs=base_pairs,
            extra=extra,
            transforms=transforms or {},
            strict_attrs=strict_attrs,
        )

        fqn = self._fqn(obj)
        fp_ver = str(type(self).version())
        cls_ver: str | None = (
            str(type(obj).version()) if isinstance(obj, Versioned) else None
        )

        head: list[FPItem] = [
            ("__fqn__", fqn),
            ("__hash_algo__", self.hash_algo),
            ("__hash_len__", self.hash_len),
            ("__fingerprint_ver__", fp_ver),
        ]
        if cls_ver is not None:
            head.append(("__cls_version__", cls_ver))

        return Fingerprint.from_pairs(
            head_pairs=head,
            attr_pairs_sorted=attr_items,
            algo=self.hash_algo,
            hash_len=self.hash_len,
            fqn=fqn,
            cls_version=cls_ver,
        )

    def key(
        self,
        obj: Any,
        *,
        attrs: Sequence[str] | None = None,
        extra: Iterable[FPItem] = (),
        transforms: Mapping[str, FPTransform] | None = None,
        strict_attrs: bool = False,
        auto_discover: bool = True,
    ) -> str:
        return self.make(
            obj,
            attrs=attrs,
            extra=extra,
            transforms=transforms,
            strict_attrs=strict_attrs,
            auto_discover=auto_discover,
        ).key()

    def _discover(self, obj: Any) -> Iterable[FPItem]:
        if is_dataclass(obj):
            for f in fields(obj):
                rule: _FingerprintRule | None = (
                    f.metadata.get(_FINGERPRINT_META_KEY) if f.metadata else None
                )
                if not rule:
                    continue
                name = rule.name or f.name
                if name in self._HEAD_KEYS:
                    raise ValueError(
                        f"fingerprint name '{name}' collides with reserved header key"
                    )
                val = getattr(obj, f.name, None)
                if rule.transform:
                    try:
                        val = rule.transform(val)
                    except Exception:
                        val = None
                yield (name, val)

        cls = type(obj)
        seen: set[str] = set()
        for base in cls.__mro__:
            for n, v in base.__dict__.items():
                if n in seen:
                    continue
                seen.add(n)
                if isinstance(v, property):
                    rule = (
                        getattr(v.fget, "__fingerprint_rule__", None)
                        if v.fget
                        else None
                    )
                    if not rule:
                        continue
                    name = rule.name or n
                    if name in self._HEAD_KEYS:
                        raise ValueError(
                            f"fingerprint name '{name}' collides with reserved header key"
                        )
                    try:
                        val = getattr(obj, n)
                    except Exception:
                        val = None
                    if rule.transform:
                        try:
                            val = rule.transform(val)
                        except Exception:
                            val = None
                    yield (name, val)

    def _collect_attrs(
        self,
        obj: Any,
        *,
        attrs: Sequence[str] | None,
        base_pairs: Iterable[FPItem],
        extra: Iterable[FPItem],
        transforms: Mapping[str, FPTransform],
        strict_attrs: bool,
    ) -> tuple[FPItem, ...]:
        pairs: list[FPItem] = list(base_pairs)

        if attrs:
            seen: set[str] = set(k for k, _ in pairs)
            for name in attrs:
                if name in seen:
                    raise ValueError(f"duplicate attribute name: {name}")
                if name in self._HEAD_KEYS:
                    raise ValueError(f"attribute '{name}' collides with header key")
                seen.add(name)

                have, val = True, None
                try:
                    val = getattr(obj, name)
                except Exception:
                    have = False
                if strict_attrs and not have:
                    raise AttributeError(f"missing attribute: {name}")
                if not have:
                    val = None

                pairs.append((name, val))

        if extra:
            for k, v in extra:
                if k in self._HEAD_KEYS:
                    raise ValueError(f"extra key '{k}' collides with header key")
                pairs.append((k, v))

        if transforms:
            tmp: list[FPItem] = []
            for k, v in pairs:
                if k in transforms:
                    try:
                        v = transforms[k](v)
                    except Exception:
                        v = None
                tmp.append((k, v))
            pairs = tmp

        items_sorted = tuple(
            sorted(((k, self._canon(v)) for k, v in pairs), key=lambda kv: kv[0])
        )
        return items_sorted

    def _guard_len(self, n: int) -> None:
        if n < 8:
            raise ValueError("hash_len must be >= 8")

    def _fqn(self, obj_or_cls: Any) -> str:
        cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
        mod = getattr(cls, "__module__", "") or ""
        qn = getattr(cls, "__qualname__", getattr(cls, "__name__", "<?>"))
        return f"{mod}.{qn}" if mod else qn

    def _canon_float(self, x: float) -> float | str:
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == 0.0:
            return 0.0
        return float(f"{x:.{self.FLOAT_DECIMALS}f}")

    def _canon(self, v: Any) -> Any:
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return self._canon_float(float(v))
        if isinstance(v, str):
            return v
        if isinstance(v, (dt.datetime, dt.date)):
            if isinstance(v, dt.datetime):
                if v.tzinfo is None:
                    v = v.replace(tzinfo=dt.timezone.utc)
                else:
                    v = v.astimezone(dt.timezone.utc)
            return ("dt", v.isoformat())
        if isinstance(v, uuid.UUID):
            return ("uuid", v.hex)
        if isinstance(v, (bytes, bytearray, memoryview)):
            return ("b64", base64.b64encode(bytes(v)).decode("ascii"))
        if isinstance(v, Mapping):
            return tuple(sorted((str(k), self._canon(v[k])) for k in v.keys()))
        if isinstance(v, (list, tuple)):
            return tuple(self._canon(x) for x in v)
        if isinstance(v, (set, frozenset)):
            return tuple(sorted((self._canon(x) for x in v), key=repr))
        if isinstance(v, np.ndarray):
            try:
                shp = tuple(int(d) for d in v.shape)
                dtype = str(v.dtype)
                with np.errstate(all="ignore"):
                    vf = v.astype(np.float64, copy=False)
                    mean = float(np.nanmean(vf))
                    var = float(np.nanvar(vf))
                return (
                    "nd",
                    shp,
                    dtype,
                    self._canon_float(mean),
                    self._canon_float(var),
                )
            except Exception:
                return ("nd", tuple(int(d) for d in getattr(v, "shape", ())), "err")
        if isinstance(v, Tensor):
            try:
                t = v.detach().cpu()
                shp = tuple(int(d) for d in t.shape)
                dtype = str(t.dtype)
                a = t.to(torch.float32).numpy()
                with np.errstate(all="ignore"):
                    mean = float(np.nanmean(a))
                    var = float(np.nanvar(a))

                return (
                    "t",
                    shp,
                    dtype,
                    self._canon_float(mean),
                    self._canon_float(var),
                )
            except Exception:
                return ("t", tuple(int(d) for d in getattr(v, "shape", ())), "err")
        s = str(v)
        return s if len(s) <= 128 else s[:125] + "..."
