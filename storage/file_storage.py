from pathlib import Path 
from uuid import uuid4

from fastapi import  UploadFile
from core.config import settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif","tif","tiff"}

async def save_upload_file(upload_file: UploadFile,)->Path:
    filename= upload_file.filename or "" 
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File extension '{extension}' is not allowed.")
    upload_dir= Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_path= (upload_dir / f"{uuid4().hex}{extension}")

    max_bytes = (settings.max_upload_size_mb * 1024 * 1024)  # Convert MB to bytes
    total = 0

    with output_path.open("wb") as buffer:
        while True:
            chunk = await upload_file.read(1024 * 1024)  # Read in 1 MB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                output_path.unlink(missing_ok=True)  # Delete the partially written file

                raise ValueError(f"File size exceeds the maximum limit of {settings.max_upload_size_mb} MB.")
            buffer.write(chunk)

    return output_path