import io
import json
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Literal

import aioboto3
import boto3
import pandas as pd
import polars as pl

from gfwldata.config.settings import Settings

logger = logging.getLogger(__name__)


class BaseSerializer:
    """Serializes data for s3 put functionality"""

    def serialize_data(
        self, obj: Any, filetype: Literal["json", "csv"]
    ) -> tuple[bytes, str] | None:
        """Serialize data based on filetype using the dispatch table."""
        serializers: dict[str, Callable] = {
            "json": self._serialize_json,
            "csv": self._serialize_csv,
        }

        if filetype not in serializers:
            raise ValueError(f"Unsupported filetype: {filetype}")

        return serializers[filetype](obj)

    @staticmethod
    def _serialize_json(obj: dict | list) -> tuple[bytes, str]:
        """Serialize a dict or list to JSON bytes with application/json content type."""
        if not isinstance(obj, (dict, list)):
            raise ValueError("obj needs to be a dict or list")

        body = json.dumps(obj)
        return body.encode("utf-8"), "application/json"

    @staticmethod
    def _serialize_csv(obj: pd.DataFrame | pl.DataFrame) -> tuple[bytes, str]:
        """Serialize a DataFrame to CSV bytes with text/csv content type."""
        if isinstance(obj, pd.DataFrame):
            buffer = io.StringIO()
            obj.to_csv(buffer, index=False)
            body = buffer.getvalue()
        elif isinstance(obj, pl.DataFrame):
            body = obj.write_csv()
        else:
            raise ValueError("obj needs to be a DataFrame")

        return body.encode("utf-8"), "text/csv"


class S3Client(BaseSerializer):
    """Synchronous AWS S3 client"""

    def __init__(self, config: Settings, bucket_name: str):
        """Initialize the S3 client with configuration and bucket name."""
        self.config = config
        self.bucket_name = bucket_name
        self.client = boto3.client(
            "s3",
            aws_access_key_id=self.config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.config.AWS_SECRET_ACCESS_KEY,
            region_name=self.config.AWS_REGION,
        )

    def get_object(self, key: str) -> str | None:
        """Retrieve an object from the S3 bucket by key"""
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            body_str = response["Body"].read().decode("utf-8")

            logger.info("Retrieved object with key: %s", key)
            return body_str

        except Exception:
            logger.exception("Failed to retrieve object with key: %s", key)
            return None

    def put_object(
        self, key: str, obj: Any, filetype: Literal["json", "csv"]
    ) -> dict[str, Any] | None:
        """Upload an object to S3 with the specified filetype."""
        try:
            body, content_type = self.serialize_data(obj, filetype)

            response = self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
            logger.info("Successfully uploaded object with key: %s", key)
            return response

        except Exception:
            logger.exception("Failed to upload object with key: %s", key)
            return None

    def list_objects(self, prefix: str) -> list[str]:
        """List file keys inside the given prefix (folder/) in the S3 bucket."""
        keys = []
        continuation_token = None

        try:
            while True:
                kwargs = {"Bucket": self.bucket_name, "Prefix": prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token

                response = self.client.list_objects_v2(**kwargs)
                keys.extend([obj["Key"] for obj in response.get("Contents", [])])

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

            logger.info("Listed %d objects in prefix: %s", len(keys), prefix)

        except Exception:
            logger.exception("Failed to list objects in prefix: %s", prefix)
            return []

        return keys


@contextmanager
def get_s3_client(config: Settings, bucket_name: str):
    """Synchronous s3 client context manager."""
    yield S3Client(config, bucket_name)


class AsyncS3Client(BaseSerializer):
    """Asynchronous AWS S3 client for uploading objects."""

    def __init__(self, bucket_name: str, s3_client: aioboto3.Session):
        self.bucket_name = bucket_name
        self.client = s3_client

    async def put_object(
        self, key: str, obj: Any, filetype: Literal["json", "csv"]
    ) -> dict[str, Any] | None:
        """Asynchronously upload an object to S3 with the specified filetype."""
        try:
            body, content_type = self.serialize_data(obj, filetype)
            response = await self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
            logger.info("Successfully uploaded object with key: %s", key)
            return response

        except Exception:
            logger.exception("Failed to upload object with key: %s", key)
            return None


@asynccontextmanager
async def get_async_s3_session(config: Settings, bucket_name: str):
    session = aioboto3.Session(
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        region_name=config.AWS_REGION,
    )
    async with session.client("s3") as s3_session:
        yield AsyncS3Client(bucket_name, s3_session)
