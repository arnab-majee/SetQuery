import uuid
from pathlib import Path

from fastapi import UploadFile

from core.config import settings


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def validate_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    return suffix


async def save_upload(file: UploadFile) -> Path:
    if not file.filename:
        raise ValueError("Uploaded file has no filename.")

    suffix = validate_extension(file.filename)

    upload_dir = settings.upload_path
    upload_dir.mkdir(parents=True, exist_ok=True)

    destination = upload_dir / f"{uuid.uuid4().hex}{suffix}"

    total_bytes = 0
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_bytes:
                    destination.unlink(missing_ok=True)
                    raise ValueError(
                        f"Upload exceeds the "
                        f"{settings.max_upload_size_mb} MB limit."
                    )

                output.write(chunk)

    finally:
        await file.close()

    return destination


def delete_file(path: Path) -> None:
    path.unlink(missing_ok=True)