"""
Small, reusable helper functions for handling uploaded files.

Kept separate from app.py so app.py stays focused on routes, and separate
from detection/ because this is generic "upload handling", not computer
vision. Phase 3 (YOLO) will import saved image paths from here.
"""

import os
import uuid

from utils.config import ALLOWED_IMAGE_EXTENSIONS, UPLOAD_FOLDER


def is_allowed_image(filename: str) -> bool:
    """Check the file extension against the allowed image types."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def save_uploaded_image(file_storage, road_name: str) -> str:
    """
    Save an uploaded image to the uploads/ folder with a unique,
    predictable filename like "north_3f2a9c.jpg".

    Args:
        file_storage: the Flask FileStorage object from request.files
        road_name: "north" | "east" | "south" | "west"

    Returns:
        The relative filename that was saved (not the full path), so it
        can be stored/referenced easily and served via /uploads/<filename>.
    """
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    unique_id = uuid.uuid4().hex[:6]
    filename = f"{road_name.lower()}_{unique_id}.{ext}"

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(save_path)

    return filename
