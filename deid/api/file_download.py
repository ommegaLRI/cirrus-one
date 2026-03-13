"""
deid.api.file_download
----------------------

Download signed URLs from Supabase to local temp files.
"""

from pathlib import Path
import tempfile
import requests
from urllib.parse import urlparse


def download_to_temp(url: str) -> Path:
    r = requests.get(url, stream=True)
    r.raise_for_status()

    # Extract the original filename from the URL path
    parsed = urlparse(url)
    filename = Path(parsed.path).name

    suffix = Path(filename).suffix

    if not suffix:
        suffix = ".tmp"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)

    tmp.close()

    return Path(tmp.name)