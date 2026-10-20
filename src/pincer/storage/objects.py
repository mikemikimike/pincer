"""Reading and writing objects of a cloud storage root through the provider SDK.

cloudpathlib maps every key to one file in its local cache, shared by every
request: a save that overlaps a download of the same key can upload the old
content, and two writes can see that file unlinked half-way. Here a read
streams from the provider and a write uploads from the caller's own buffer,
so requests share nothing.

A write can also be conditional, which cloudpathlib has no API for:
`if_version` (the version `read` returned) or `if_absent` makes the provider
itself refuse the write when the object changed or exists, so no other
writer, on any replica, can get between the check and the write. A refused
write raises KeyConflict.

Listing, stat, move and delete still go through cloudpathlib: none of them
touch its cache.
"""

from __future__ import annotations

import contextlib
import io
import logging
import shutil
from typing import IO, TYPE_CHECKING, Any

from pincer.storage.errors import KeyConflict, KeyNotFound
from pincer.storage.keys import etag
from pincer.storage.limits import COPY_CHUNK

if TYPE_CHECKING:
    from cloudpathlib import CloudPath

logger = logging.getLogger(__name__)


def objects_for(root: CloudPath) -> CloudObjects:
    """The object IO for `root`'s client."""
    from cloudpathlib import AzureBlobClient, GSClient, S3Client

    client = root.client
    if isinstance(client, S3Client):
        return S3Objects(client)
    if isinstance(client, AzureBlobClient):
        return AzureObjects(client)
    if isinstance(client, GSClient):
        return GSObjects(client)
    return CloudObjects()


class CloudObjects:
    """Object IO through cloudpathlib itself, for a client with no SDK of its
    own below it (cloudpathlib's local test clients). The conditions are
    checked before the write, not by the store, so only the browser's
    in-process lock keeps the check and the write together."""

    def read(self, path: CloudPath) -> tuple[bytes, object]:
        """The object's content, and a version to pass back as `if_version`."""
        try:
            with path.open("rb", force_overwrite_from_cloud=True) as f:
                data = f.read()
        except FileNotFoundError as e:
            raise KeyNotFound(f"No such file: {path}") from e
        return data, etag(data)

    def open(self, path: CloudPath) -> IO[bytes]:
        """The object's content as a stream, for the caller to close."""
        return path.open("rb", force_overwrite_from_cloud=True)  # type: ignore[no-any-return]

    def write(
        self,
        path: CloudPath,
        source: IO[bytes],
        *,
        content_type: str,
        if_version: object = None,
        if_absent: bool = False,
    ) -> None:
        if if_absent and path.exists():
            raise KeyConflict(f"{path} already exists")
        if if_version is not None:
            try:
                current = self.read(path)[1]
            except KeyNotFound:
                current = None
            if current != if_version:
                raise KeyConflict(f"{path} changed")
        with path.open("wb", force_overwrite_to_cloud=True) as out:
            shutil.copyfileobj(source, out, COPY_CHUNK)


class S3Objects(CloudObjects):
    """S3 through boto3: `IfMatch` on the ETag, `IfNoneMatch: *` for a new object."""

    def __init__(self, client: Any) -> None:
        self._s3 = client.client
        self._unconditional = False

    def _get(self, path: CloudPath) -> Any:
        from botocore.exceptions import ClientError

        try:
            return self._s3.get_object(Bucket=path.bucket, Key=path.key)  # type: ignore[attr-defined]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise KeyNotFound(f"No such file: {path}") from e
            raise

    def read(self, path: CloudPath) -> tuple[bytes, object]:
        obj = self._get(path)
        with contextlib.closing(obj["Body"]) as body:
            return body.read(), obj["ETag"]

    def open(self, path: CloudPath) -> IO[bytes]:
        return self._get(path)["Body"]  # type: ignore[no-any-return]

    def write(
        self,
        path: CloudPath,
        source: IO[bytes],
        *,
        content_type: str,
        if_version: object = None,
        if_absent: bool = False,
    ) -> None:
        from botocore.exceptions import ClientError, ParamValidationError

        condition: dict[str, Any] = {}
        if if_absent:
            condition["IfNoneMatch"] = "*"
        elif if_version is not None:
            condition["IfMatch"] = if_version
        start = source.tell()

        def put(**extra: Any) -> None:
            source.seek(start)
            self._s3.put_object(
                Bucket=path.bucket,  # type: ignore[attr-defined]
                Key=path.key,  # type: ignore[attr-defined]
                Body=source,
                ContentType=content_type,
                **extra,
            )

        if not condition or self._unconditional:
            put()
            return
        try:
            put(**condition)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise KeyConflict(f"{path} changed or already exists") from e
            if code != "NotImplemented":
                raise
            self._fall_back(path, e)
            put()
        except ParamValidationError as e:
            # A botocore from before conditional writes.
            self._fall_back(path, e)
            put()

    def _fall_back(self, path: CloudPath, error: Exception) -> None:
        """Some S3-compatible servers have no conditional writes: write without
        them from now on, which leaves only the browser's in-process lock."""
        self._unconditional = True
        logger.warning(
            "Storage: %s does not take conditional writes (%s) — saves are only serialised within this process",
            path.bucket,  # type: ignore[attr-defined]
            error,
        )


class _Reader(io.RawIOBase):
    """An Azure blob download as a file object."""

    def __init__(self, downloader: Any) -> None:
        self._downloader = downloader

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> bytes:
        data: bytes = self._downloader.read(-1 if size is None else size)
        return data


class AzureObjects(CloudObjects):
    """Azure Blob through azure-storage-blob: `If-Match` on the ETag, no overwrite for a new blob."""

    def __init__(self, client: Any) -> None:
        self._service = client.service_client

    def _download(self, path: CloudPath) -> Any:
        from azure.core.exceptions import ResourceNotFoundError

        blob = self._service.get_blob_client(container=path.container, blob=path.blob)  # type: ignore[attr-defined]
        try:
            return blob.download_blob()
        except ResourceNotFoundError as e:
            raise KeyNotFound(f"No such file: {path}") from e

    def read(self, path: CloudPath) -> tuple[bytes, object]:
        downloader = self._download(path)
        return downloader.readall(), downloader.properties.etag

    def open(self, path: CloudPath) -> IO[bytes]:
        return _Reader(self._download(path))  # type: ignore[return-value]

    def write(
        self,
        path: CloudPath,
        source: IO[bytes],
        *,
        content_type: str,
        if_version: object = None,
        if_absent: bool = False,
    ) -> None:
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceExistsError, ResourceModifiedError
        from azure.storage.blob import ContentSettings

        condition: dict[str, Any] = {}
        if if_version is not None and not if_absent:
            condition = {"etag": if_version, "match_condition": MatchConditions.IfNotModified}
        blob = self._service.get_blob_client(container=path.container, blob=path.blob)  # type: ignore[attr-defined]
        try:
            blob.upload_blob(
                source,
                overwrite=not if_absent,
                content_settings=ContentSettings(content_type=content_type),
                **condition,
            )
        except (ResourceExistsError, ResourceModifiedError) as e:
            raise KeyConflict(f"{path} changed or already exists") from e


class GSObjects(CloudObjects):
    """Google Cloud Storage through google-cloud-storage: `ifGenerationMatch`, 0 for a new object."""

    def __init__(self, client: Any) -> None:
        self._gcs = client.client

    def _blob(self, path: CloudPath) -> Any:
        return self._gcs.bucket(path.bucket).blob(path.blob)  # type: ignore[attr-defined]

    def read(self, path: CloudPath) -> tuple[bytes, object]:
        from google.api_core.exceptions import NotFound

        blob = self._blob(path)
        try:
            data: bytes = blob.download_as_bytes()
        except NotFound as e:
            raise KeyNotFound(f"No such file: {path}") from e
        # Set from the download's response headers: the generation of what was read.
        return data, blob.generation

    def open(self, path: CloudPath) -> IO[bytes]:
        return self._blob(path).open("rb")  # type: ignore[no-any-return]

    def write(
        self,
        path: CloudPath,
        source: IO[bytes],
        *,
        content_type: str,
        if_version: object = None,
        if_absent: bool = False,
    ) -> None:
        from google.api_core.exceptions import PreconditionFailed

        condition: dict[str, Any] = {}
        if if_absent:
            condition["if_generation_match"] = 0
        elif if_version is not None:
            condition["if_generation_match"] = if_version
        try:
            self._blob(path).upload_from_file(source, content_type=content_type, **condition)
        except PreconditionFailed as e:
            raise KeyConflict(f"{path} changed or already exists") from e
