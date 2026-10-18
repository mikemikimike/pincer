"""Storage browser: list, read, write, upload, move and delete files under one root.

The root is a local directory or a cloud prefix (`s3://`, `az://`, `gs://`),
reached through cloudpathlib with the same `storage_*` credentials as remote
skill roots (see `resolve_storage_root` for how it is chosen). Every key is a
POSIX path relative to that root; `confine` refuses anything that would leave
it, and `is_protected` keeps the skills directory (when it lives under the
root) safe from delete and move.

Everything here blocks — the API router runs it in a worker thread.
"""

from __future__ import annotations

import base64
import binascii
import bisect
import hashlib
import hmac
import io
import json
import logging
import mimetypes
import os
import secrets
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

from pincer.exceptions import PincerError
from pincer.storage import clients
from pincer.tools.path_sandbox import SandboxDenied, confine_path

if TYPE_CHECKING:
    from cloudpathlib import CloudPath

    from pincer.config import Settings

logger = logging.getLogger(__name__)

type StoragePath = Path | CloudPath
type Disposition = Literal["inline", "attachment"]

#: Most entries of one folder the browser can page through. Listing a folder
#: reads every name in it (cloudpathlib listings cannot start mid-way), so this
#: bounds the memory a huge folder can take; entries past it are not reachable.
MAX_LIST_ENTRIES = 100_000
#: Page size of a listing: the default, and the most a client may ask for.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
#: Default for `storage_upload_file_max_size`.
DEFAULT_UPLOAD_MAX_BYTES = 20 * 1024 * 1024
#: Largest file the text editor reads or saves.
MAX_TEXT_BYTES = 2 * 1024 * 1024
#: Object stores have no directories: an empty folder is this zero-byte marker.
FOLDER_MARKER = ".keep"
_UPLOAD_TMP_PREFIX = ".pincer-upload-"
_STAT_WORKERS = 16
_COPY_CHUNK = 1024 * 1024

mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("text/markdown", ".markdown")


# ── Errors ───────────────────────────────────────────────────────────────────


class StorageError(PincerError):
    """A storage request that cannot be served; `status` is its HTTP status."""

    status = 500


class InvalidKey(StorageError):
    status = 400


class InvalidSignature(StorageError):
    status = 403


class ProtectedKey(StorageError):
    status = 403


class KeyNotFound(StorageError):
    status = 404


class KeyConflict(StorageError):
    status = 409


class LinkExpired(StorageError):
    status = 410


class TooLarge(StorageError):
    status = 413


class NotText(StorageError):
    status = 415


class StorageUnavailable(StorageError):
    status = 503


# ── Keys ─────────────────────────────────────────────────────────────────────


def normalize_key(key: str, *, allow_root: bool = True) -> str:
    """`key` as a clean root-relative POSIX path ("" is the root).

    Absolute, home-relative and scheme-qualified keys, `..` segments and NUL
    bytes are refused outright rather than resolved: a key is a name under
    the root, never a path computation.
    """
    raw = key.replace("\\", "/")
    if raw.startswith(("/", "~")) or clients.ANY_SCHEME_RE.match(raw) or "\x00" in raw:
        raise InvalidKey(f"Invalid key: {key!r}")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise InvalidKey(f"Invalid key: {key!r}")
    normalized = "/".join(parts)
    if not normalized and not allow_root:
        raise InvalidKey("A key below the storage root is required")
    return normalized


def _join(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _etag(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _iso(mtime: float | None) -> str | None:
    return datetime.fromtimestamp(mtime, UTC).isoformat() if mtime else None


def _hidden(name: str) -> bool:
    return name == FOLDER_MARKER or name.startswith(_UPLOAD_TMP_PREFIX)


def content_type_for(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


type _SortKey = tuple[int, str, str]


def _sort_key(name: str, is_dir: bool) -> _SortKey:
    """Folders before files, then case-insensitive by name (the exact name breaks ties)."""
    return (0 if is_dir else 1, name.lower(), name)


def _encode_cursor(key: _SortKey) -> str:
    return _b64(json.dumps([key[0], key[2]], separators=(",", ":")).encode())


def _decode_cursor(cursor: str) -> _SortKey:
    try:
        kind, name = json.loads(_unb64(cursor))
        if kind not in (0, 1) or not isinstance(name, str):
            raise ValueError(cursor)
    except (ValueError, TypeError, binascii.Error) as e:
        raise InvalidKey("Invalid cursor") from e
    return (kind, name.lower(), name)


class _LimitedReader(io.RawIOBase):
    """Reads `source`, raising TooLarge once more than `limit` bytes came through."""

    def __init__(self, source: IO[bytes], limit: int) -> None:
        self._source, self._left, self._limit = source, limit, limit

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self._source.read(size if size is not None and size >= 0 else self._left + 1)
        self._left -= len(data)
        if self._left < 0:
            raise TooLarge(f"Uploads are limited to {self._limit} bytes")
        return data


# ── Root ─────────────────────────────────────────────────────────────────────


def _open_location(location: str | Path, settings: Settings) -> StoragePath:
    if not clients.is_remote_uri(location):
        root = Path(location).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    uri = str(location)
    scheme = clients.uri_scheme(uri)
    hint = clients.STORAGE_HINTS.get(scheme, "")
    try:
        from cloudpathlib.enums import FileCacheMode
    except ImportError as e:
        raise StorageUnavailable(
            f"Cloud storage ({uri}) needs cloudpathlib. Install it with: pip install 'pincer-agent[remote-skills]'"
        ) from e
    try:
        # Cached copies are only a transfer buffer here: each one is dropped as
        # soon as its file is closed, so the bucket is always what is served.
        client = clients.build_client(
            scheme, settings, settings.data_dir / "cache" / "storage", file_cache_mode=FileCacheMode.close_file
        )
        return client.CloudPath(uri)  # type: ignore[no-any-return]
    except Exception as e:
        raise StorageUnavailable(f"Cannot open storage root {uri}: {type(e).__name__}: {e}. {hint}") from e


def _cloud_uri(value: str) -> str:
    scheme, rest = value.split("://", 1)
    return f"{scheme.lower()}://{rest.strip('/')}"


def _relative_prefix(root: StoragePath, location: str | Path) -> str | None:
    """`location` relative to `root` ("" when they are the same), or None when outside it."""
    if isinstance(root, Path):
        if clients.is_remote_uri(location):
            return None
        target = Path(location).expanduser().resolve()
        if not target.is_relative_to(root):
            return None
        rel = target.relative_to(root).as_posix()
        return "" if rel == "." else rel
    if not clients.is_remote_uri(location):
        return None
    base_uri, target_uri = _cloud_uri(str(root)), _cloud_uri(str(location))
    if target_uri == base_uri:
        return ""
    return target_uri[len(base_uri) + 1 :] if target_uri.startswith(base_uri + "/") else None


def resolve_storage_root(settings: Settings) -> tuple[StoragePath, list[str]]:
    """The root the storage browser serves, and the prefixes under it that are protected.

    - `storage_root` set: that location. The skills dir is protected when it lies under it.
    - Unset, with a cloud `skills_dir`: the root of that bucket / container, with
      the skills prefix protected.
    - Unset, with a local `skills_dir`: `<data_dir>/storage`. The skills dir is
      not under it, so nothing is protected.
    """
    skills_dir = settings.skills_dir
    if settings.storage_root:
        root = _open_location(settings.storage_root, settings)
    elif clients.is_remote_uri(skills_dir):
        scheme, rest = str(skills_dir).split("://", 1)
        root = _open_location(f"{scheme.lower()}://{rest.split('/', 1)[0]}/", settings)
    else:
        root = _open_location(settings.data_dir / "storage", settings)
    protected = _relative_prefix(root, skills_dir)
    return root, [] if protected is None else [protected]


def _signing_key(settings: Settings) -> bytes:
    """The key for local presigned links, derived from the JWT secret so a
    link stays valid across restarts and replicas that share that secret."""
    from pincer.security.credentials import load_or_create_jwt_secret

    secret = load_or_create_jwt_secret(settings.jwt_secret.get_secret_value(), settings.data_dir)
    return hmac.new(secret.encode(), b"pincer-storage-presign", hashlib.sha256).digest()


# ── Browser ──────────────────────────────────────────────────────────────────


@dataclass
class Download:
    """A file to send: a local `path`, or an open cloud `stream`."""

    name: str
    size: int | None
    content_type: str
    path: Path | None = None
    stream: IO[bytes] | None = None


class StorageBrowser:
    def __init__(
        self,
        root: StoragePath,
        protected: list[str],
        presign_ttl: int,
        signing_key: bytes,
        upload_max_bytes: int = DEFAULT_UPLOAD_MAX_BYTES,
    ) -> None:
        self.root = root
        self.protected = protected
        self.presign_ttl = presign_ttl
        self.upload_max_bytes = upload_max_bytes
        self._signing_key = signing_key
        self._presigner: Any = None

    @classmethod
    def from_settings(cls, settings: Settings) -> StorageBrowser:
        root, protected = resolve_storage_root(settings)
        return cls(
            root,
            protected,
            settings.storage_presign_ttl,
            _signing_key(settings),
            upload_max_bytes=settings.storage_upload_file_max_size,
        )

    @property
    def is_cloud(self) -> bool:
        return not isinstance(self.root, Path)

    @property
    def scheme(self) -> str:
        return clients.uri_scheme(str(self.root)) if self.is_cloud else "file"

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.scheme,
            "root": str(self.root),
            "protected": self.protected,
            "presign_ttl": self.presign_ttl,
            "max_text_bytes": MAX_TEXT_BYTES,
            "max_upload_bytes": self.upload_max_bytes,
        }

    # ── Paths ──

    def confine(self, key: str, *, allow_root: bool = True) -> StoragePath:
        """The path for `key`, guaranteed to be inside the root."""
        normalized = normalize_key(key, allow_root=allow_root)
        try:
            if isinstance(self.root, Path):
                return confine_path(self.root, normalized or ".", label="the storage root")
            return clients.confine_cloud_path(self.root, normalized, label="the storage root")
        except SandboxDenied as e:
            raise InvalidKey(str(e)) from e

    def _entry_path(self, key: str) -> StoragePath:
        """Like `confine`, but a local symlink is the link itself, not its target:
        deleting or moving a link must not touch what it points at."""
        path = self.confine(key, allow_root=False)
        if isinstance(self.root, Path):
            raw = self.root / normalize_key(key)
            if raw.is_symlink():
                return raw
        return path

    def is_protected(self, key: str) -> bool:
        """True for a protected prefix, anything inside one, and every ancestor
        of one (removing a parent folder would take the skills dir with it)."""
        normalized = normalize_key(key)
        for prefix in self.protected:
            if (
                not prefix
                or not normalized
                or normalized == prefix
                or normalized.startswith(prefix + "/")
                or prefix.startswith(normalized + "/")
            ):
                return True
        return False

    def _require_unprotected(self, key: str) -> None:
        if self.is_protected(key):
            raise ProtectedKey(f"'{key}' is protected: it is, contains or is inside the skills directory")

    def _file(self, key: str) -> StoragePath:
        path = self.confine(key, allow_root=False)
        if not path.is_file():
            raise KeyNotFound(f"No such file: {key}")
        return path

    # ── Listing ──

    def _children(self, folder: StoragePath) -> list[tuple[StoragePath, bool]]:
        """Up to MAX_LIST_ENTRIES + 1 (child, is_dir) pairs, visible entries only."""
        if isinstance(folder, Path):
            source: Any = ((child, child.is_dir()) for child in folder.iterdir())
        else:
            # `iterdir` drops the is-dir flag the listing already has, and asking
            # again costs a request per child.
            source = folder.client._list_dir(folder, recursive=False)
        children: list[tuple[StoragePath, bool]] = []
        for child, is_dir in source:
            if child == folder or _hidden(child.name):
                continue
            children.append((child, is_dir))
            if len(children) > MAX_LIST_ENTRIES:
                break
        return children

    @staticmethod
    def _stat(path: StoragePath) -> tuple[int | None, float | None]:
        try:
            st = path.stat()
        except Exception:
            return None, None
        return st.st_size, st.st_mtime

    def list_dir(
        self, prefix: str = "", *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> dict[str, Any]:
        """One page of a folder: folders first, then files, each by name.

        `cursor` is the `next_cursor` of the page before (None for the first).
        It names the last entry served rather than a position, so entries
        added or removed in between neither repeat nor skip the rest.
        """
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise InvalidKey(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        after = _decode_cursor(cursor) if cursor else None
        normalized = normalize_key(prefix)
        folder = self.confine(normalized)
        if not folder.is_dir():
            # A fresh bucket root has no objects and so "is" no directory yet.
            if normalized:
                raise KeyNotFound(f"No such folder: {prefix}")
            return {"prefix": "", "folders": [], "files": [], "next_cursor": None, "truncated": False}

        children = self._children(folder)
        truncated = len(children) > MAX_LIST_ENTRIES
        ordered = sorted(
            ((_sort_key(c.name, is_dir), c) for c, is_dir in children[:MAX_LIST_ENTRIES]), key=lambda e: e[0]
        )
        start = bisect.bisect_right([key for key, _ in ordered], after) if after else 0
        page = ordered[start : start + limit]
        has_more = start + limit < len(ordered)

        dirs = [c for key, c in page if key[0] == 0]
        files = [c for key, c in page if key[0] == 1]
        if isinstance(folder, Path) or len(files) < 2:
            stats = [self._stat(f) for f in files]
        else:
            # One HEAD request per object: run them side by side.
            with ThreadPoolExecutor(max_workers=_STAT_WORKERS) as pool:
                stats = list(pool.map(self._stat, files))

        def entry(name: str) -> tuple[str, bool]:
            key = _join(normalized, name)
            return key, self.is_protected(key)

        return {
            "prefix": normalized,
            "folders": [
                {"key": key, "name": d.name, "protected": protected} for d in dirs for key, protected in [entry(d.name)]
            ],
            "files": [
                {
                    "key": key,
                    "name": f.name,
                    "size": size,
                    "last_modified": _iso(mtime),
                    "content_type": content_type_for(f.name),
                    "protected": protected,
                }
                for f, (size, mtime) in zip(files, stats, strict=True)
                for key, protected in [entry(f.name)]
            ],
            "next_cursor": _encode_cursor(page[-1][0]) if has_more else None,
            "truncated": truncated,
        }

    def _file_entry(self, key: str, path: StoragePath) -> dict[str, Any]:
        size, mtime = self._stat(path)
        return {
            "key": key,
            "name": path.name,
            "size": size,
            "last_modified": _iso(mtime),
            "content_type": content_type_for(path.name),
            "protected": self.is_protected(key),
        }

    # ── Reading ──

    def open_read(self, key: str) -> Download:
        """The file at `key`, ready to send. A cloud file is downloaded to a
        transfer buffer and must be closed by the caller."""
        path = self._file(key)
        size, _ = self._stat(path)
        content_type = content_type_for(path.name)
        if isinstance(path, Path):
            return Download(path.name, size, content_type, path=path)
        stream = path.open("rb", force_overwrite_from_cloud=True)
        return Download(path.name, size, content_type, stream=stream)

    def _read_bytes(self, path: StoragePath) -> bytes:
        if isinstance(path, Path):
            return path.read_bytes()
        with path.open("rb", force_overwrite_from_cloud=True) as f:
            return f.read()

    def read_text(self, key: str) -> dict[str, Any]:
        """A UTF-8 text file with its etag, for the editor."""
        path = self._file(key)
        size, _ = self._stat(path)
        if size is not None and size > MAX_TEXT_BYTES:
            raise TooLarge(f"{key} is larger than {MAX_TEXT_BYTES} bytes; download it instead")
        data = self._read_bytes(path)
        if len(data) > MAX_TEXT_BYTES:
            raise TooLarge(f"{key} is larger than {MAX_TEXT_BYTES} bytes; download it instead")
        if b"\x00" in data:
            raise NotText(f"{key} is not a text file")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise NotText(f"{key} is not UTF-8 text") from e
        return {"key": normalize_key(key), "content": content, "etag": _etag(data), "size": len(data)}

    # ── Writing ──

    def _write_stream(self, path: StoragePath, source: IO[bytes]) -> None:
        if isinstance(path, Path):
            # Written next to the target and renamed over it: a reader never sees half a file.
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{_UPLOAD_TMP_PREFIX}{secrets.token_hex(6)}")
            try:
                with tmp.open("wb") as out:
                    shutil.copyfileobj(source, out, _COPY_CHUNK)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
            return
        with path.open("wb", force_overwrite_to_cloud=True) as out:
            shutil.copyfileobj(source, out, _COPY_CHUNK)

    def write_text(self, key: str, content: str, expected_etag: str | None = None) -> dict[str, Any]:
        """Save `content` to `key`. With `expected_etag`, refuse (409) unless the
        file is still the version that was read — someone else's save in between
        must not be overwritten silently. Without it the write is unconditional."""
        data = content.encode("utf-8")
        if len(data) > MAX_TEXT_BYTES:
            raise TooLarge(f"Text files are limited to {MAX_TEXT_BYTES} bytes")
        path = self.confine(key, allow_root=False)
        if path.is_dir():
            raise KeyConflict(f"{key} is a folder")
        if expected_etag is not None:
            if not path.is_file():
                raise KeyConflict(f"{key} was deleted on the server")
            if not hmac.compare_digest(_etag(self._read_bytes(path)), expected_etag):
                raise KeyConflict(f"{key} changed on the server since it was opened")
        self._write_stream(path, io.BytesIO(data))
        return {"key": normalize_key(key), "etag": _etag(data), "size": len(data)}

    def upload(self, prefix: str, filename: str, source: IO[bytes], *, overwrite: bool = False) -> dict[str, Any]:
        """Store one uploaded file under `prefix`. `filename` may carry
        subfolders ("reports/q1.pdf", from a folder pick); they are kept."""
        relative = normalize_key(filename or "")
        if not relative:
            raise InvalidKey("The uploaded file has no name")
        key = _join(normalize_key(prefix), relative)
        path = self.confine(key, allow_root=False)
        if path.is_dir():
            raise KeyConflict(f"{key} is a folder")
        if not overwrite and path.exists():
            raise KeyConflict(f"{key} already exists")
        body = self._within_upload_limit(source)
        try:
            self._write_stream(path, body)
        finally:
            if body is not source:
                body.close()
        return self._file_entry(key, path)

    def _within_upload_limit(self, source: IO[bytes]) -> IO[bytes]:
        """`source`, refused with TooLarge if it is over `upload_max_bytes`.

        Checked before anything is written: a cloud write uploads whatever is in
        its buffer when it is closed, so failing half-way would store half a
        file. A source that cannot tell its size is spooled first, up to the limit.
        """
        limit = self.upload_max_bytes
        too_large = TooLarge(f"Uploads are limited to {limit} bytes")
        if source.seekable():
            start = source.tell()
            size = source.seek(0, os.SEEK_END) - start
            source.seek(start)
            if size > limit:
                raise too_large
            return source
        spool = tempfile.SpooledTemporaryFile(max_size=_COPY_CHUNK * 8)  # noqa: SIM115 — the caller closes it
        shutil.copyfileobj(_LimitedReader(source, limit), spool, _COPY_CHUNK)
        spool.seek(0)
        return spool  # type: ignore[return-value]

    def mkdir(self, prefix: str) -> dict[str, Any]:
        key = normalize_key(prefix, allow_root=False)
        path = self.confine(key, allow_root=False)
        if path.exists():
            raise KeyConflict(f"{key} already exists")
        if isinstance(path, Path):
            path.mkdir(parents=True)
        else:
            with (path / FOLDER_MARKER).open("wb", force_overwrite_to_cloud=True):
                pass
        return {"key": key, "name": path.name, "protected": self.is_protected(key)}

    def _remove(self, path: StoragePath) -> None:
        if isinstance(path, Path):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        elif path.is_dir():
            path.rmtree()
        else:
            path.unlink()

    def delete(self, key: str, *, folder: bool) -> dict[str, Any]:
        """Delete the file `key`, or with `folder` the folder `key` and everything in it."""
        normalized = normalize_key(key, allow_root=False)
        self._require_unprotected(normalized)
        path = self._entry_path(normalized)
        is_link = isinstance(path, Path) and path.is_symlink()
        if folder:
            if not path.is_dir() or is_link:
                raise KeyNotFound(f"No such folder: {key}")
        elif not (path.is_file() or is_link):
            raise KeyNotFound(f"No such file: {key}")
        self._remove(path)
        return {"key": normalized, "kind": "folder" if folder else "file"}

    def move(self, src: str, dst: str, *, overwrite: bool = False) -> dict[str, Any]:
        """Move or rename a file or folder. Cloud stores copy then delete."""
        src_key = normalize_key(src, allow_root=False)
        dst_key = normalize_key(dst, allow_root=False)
        if src_key == dst_key:
            raise InvalidKey("Source and destination are the same")
        if dst_key.startswith(src_key + "/"):
            raise InvalidKey("A folder cannot be moved into itself")
        self._require_unprotected(src_key)
        self._require_unprotected(dst_key)

        source = self._entry_path(src_key)
        source_is_link = isinstance(source, Path) and source.is_symlink()
        if not (source.exists() or source_is_link):
            raise KeyNotFound(f"No such file or folder: {src}")
        target = self._entry_path(dst_key)
        if target.exists():
            if not overwrite:
                raise KeyConflict(f"{dst_key} already exists")
            if target.is_dir() != source.is_dir():
                raise KeyConflict(f"{dst_key} exists and is a {'folder' if target.is_dir() else 'file'}")
            self._remove(target)

        if isinstance(source, Path):
            assert isinstance(target, Path)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        elif source.is_dir():
            assert not isinstance(target, Path)
            for dirpath, _dirs, files in source.walk():
                rel = dirpath.parts[len(source.parts) :]
                for name in files:
                    (dirpath / name).rename(target.joinpath(*rel, name))
            if source.exists():
                # Leftovers no file listing shows, such as the "dir/" placeholder
                # objects some tools create.
                source.rmtree()
        else:
            assert not isinstance(target, Path)
            source.rename(target)
        return {"from": src_key, "to": dst_key}

    # ── Links ──

    def presign(self, key: str, disposition: Disposition = "attachment") -> dict[str, Any]:
        """A link that serves `key` without a bearer token, valid for `presign_ttl` seconds.

        Cloud roots sign with the provider (S3 honours `disposition`); local
        roots get a relative `/api/storage/signed/<token>` URL, HMAC-signed.
        """
        path = self._file(key)
        normalized = normalize_key(key)
        expires = int(time.time()) + self.presign_ttl
        if isinstance(path, Path):
            url = f"/api/storage/signed/{self.sign(normalized, expires, disposition)}"
        else:
            url = self._presign_cloud(path, disposition)
        return {"url": url, "expires_at": datetime.fromtimestamp(expires, UTC).isoformat()}

    def _presign_cloud(self, path: CloudPath, disposition: Disposition) -> str:
        from cloudpathlib import S3Client

        client = path.client
        if isinstance(client, S3Client):
            # cloudpathlib's presign has no response headers; S3 takes them as
            # parameters, so a download link downloads instead of navigating.
            from urllib.parse import quote

            url: str = self._s3_presigner(client).generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": path.bucket,  # type: ignore[attr-defined]
                    "Key": path.key,  # type: ignore[attr-defined]
                    "ResponseContentDisposition": f"{disposition}; filename*=UTF-8''{quote(path.name)}",
                    "ResponseContentType": content_type_for(path.name),
                },
                ExpiresIn=self.presign_ttl,
            )
            return url
        return path.as_url(presign=True, expire_seconds=self.presign_ttl)

    def _s3_presigner(self, client: Any) -> Any:
        """A boto3 client like `client`'s, but signing with SigV4.

        boto3 may presign with SigV2 against a custom endpoint, and S3-compatible
        servers (SeaweedFS, MinIO) reject a SigV2 link that carries response-*
        parameters. AWS itself has deprecated SigV2.
        """
        if self._presigner is None:
            from botocore.config import Config

            base = client.client.meta
            config = Config(signature_version="s3v4")
            self._presigner = client.sess.client(
                "s3",
                endpoint_url=base.endpoint_url,
                region_name=base.region_name,
                config=base.config.merge(config) if base.config is not None else config,
            )
        return self._presigner

    def sign(self, key: str, expires: int, disposition: Disposition) -> str:
        payload = _b64(json.dumps({"k": key, "e": expires, "d": disposition}, separators=(",", ":")).encode())
        signature = _b64(hmac.new(self._signing_key, payload.encode(), hashlib.sha256).digest())
        return f"{payload}.{signature}"

    def verify(self, token: str) -> tuple[str, Disposition]:
        """The key and disposition a token signed by `sign` grants."""
        payload, _, signature = token.partition(".")
        expected = _b64(hmac.new(self._signing_key, payload.encode(), hashlib.sha256).digest())
        if not signature or not hmac.compare_digest(signature, expected):
            raise InvalidSignature("Invalid link")
        try:
            claims = json.loads(_unb64(payload))
            key, expires, disposition = str(claims["k"]), int(claims["e"]), claims["d"]
        except (ValueError, KeyError, TypeError, binascii.Error) as e:
            raise InvalidSignature("Invalid link") from e
        if expires < time.time():
            raise LinkExpired("This link has expired")
        if disposition not in ("inline", "attachment"):
            raise InvalidSignature("Invalid link")
        return key, disposition


_browser: tuple[Any, StorageBrowser] | None = None


def get_storage_browser(settings: Settings) -> StorageBrowser:
    """The browser for `settings`, built once per settings object (clients are
    not free to build: an S3 client starts a boto3 session)."""
    global _browser
    if _browser is None or _browser[0] is not settings:
        _browser = (settings, StorageBrowser.from_settings(settings))
    return _browser[1]


def reset_storage_browser() -> None:
    global _browser
    _browser = None
