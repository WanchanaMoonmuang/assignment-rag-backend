from __future__ import annotations

import asyncio

from app.settings import Settings


async def upload_object(settings: Settings, object_name: str, data: bytes, content_type: str) -> None:
    if not settings.gcs_bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME is not configured")

    from google.cloud import storage

    def upload() -> None:
        client = storage.Client(project=settings.gcs_project_id or None)
        blob = client.bucket(settings.gcs_bucket_name).blob(object_name)
        blob.upload_from_string(data, content_type=content_type)

    await asyncio.to_thread(upload)


async def delete_object(settings: Settings, object_name: str) -> None:
    if not settings.gcs_bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME is not configured")

    from google.cloud import storage

    def delete() -> None:
        client = storage.Client(project=settings.gcs_project_id or None)
        client.bucket(settings.gcs_bucket_name).blob(object_name).delete()

    await asyncio.to_thread(delete)
