"""
Download CFPB consumer complaint data.

Strategy
--------
The CFPB API returns at most 25 unique records (pagination is broken server-side).
The bulk CSV is the reliable path.  We stream-decompress the 200 MB ZIP and stop
as soon as we have enough decompressed bytes for *n_rows* — so we typically only
download 5–20 MB instead of the full archive.
"""
import io
import struct
import zlib
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CFPB_BULK_URL = (
    "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
)
CFPB_API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
)

_BYTES_PER_ROW_EST = 700   # generous estimate for CFPB rows (narratives can be long)


# ── ZIP streaming helper ──────────────────────────────────────────────────────

def _find_csv_data_offset(header: bytes) -> tuple[int, int]:
    """
    Find the first ZIP local file header in *header* and return
    (data_start_offset, compression_method).
    """
    sig = b"PK\x03\x04"
    idx = header.find(sig)
    if idx == -1:
        raise ValueError("ZIP local file header not found in buffer.")
    compression = struct.unpack_from("<H", header, idx + 8)[0]
    fname_len = struct.unpack_from("<H", header, idx + 26)[0]
    extra_len = struct.unpack_from("<H", header, idx + 28)[0]
    data_start = idx + 30 + fname_len + extra_len
    return data_start, compression


def stream_bulk_nrows(
    n_rows: int = 50_000,
    output_path: str = "data/complaints_sample.csv",
) -> str:
    """
    Download just enough of the CFPB bulk ZIP to produce *n_rows* CSV rows.

    We stream-decompress and stop once we've accumulated ~n_rows × 700 bytes of
    raw CSV text, then parse with pandas.  Typically downloads < 20 MB.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Sample already at {output_path} — skipping download.")
        return str(output_path)

    target_bytes = int(n_rows * _BYTES_PER_ROW_EST * 1.5)   # 50 % buffer for safety
    print(f"Streaming CFPB bulk ZIP (target ≈ {target_bytes // 1_000_000} MB decompressed)…")

    header_buf = bytearray()
    header_parsed = False
    data_start = 0
    compression = -1
    decomp: Optional[zlib.Decompress] = None
    decomp_buf = io.BytesIO()
    downloaded = 0

    with requests.get(CFPB_BULK_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tqdm(unit="B", unit_scale=True, desc="Downloading") as bar:
            for raw_chunk in resp.iter_content(chunk_size=131_072):
                downloaded += len(raw_chunk)
                bar.update(len(raw_chunk))

                if not header_parsed:
                    header_buf.extend(raw_chunk)
                    if len(header_buf) < 512:
                        continue
                    try:
                        data_start, compression = _find_csv_data_offset(bytes(header_buf))
                    except ValueError:
                        continue

                    header_parsed = True
                    if compression == 8:
                        decomp = zlib.decompressobj(-15)
                    elif compression == 0:
                        decomp = None
                    else:
                        raise ValueError(f"Unsupported ZIP compression method: {compression}")

                    payload = bytes(header_buf)[data_start:]
                else:
                    payload = raw_chunk

                if decomp is not None:
                    try:
                        decompressed = decomp.decompress(payload)
                    except zlib.error as e:
                        print(f"\nDecompression error: {e}")
                        break
                else:
                    decompressed = payload

                decomp_buf.write(decompressed)

                if decomp_buf.tell() >= target_bytes:
                    print(
                        f"\n  Downloaded {downloaded / 1_000_000:.1f} MB, "
                        f"decompressed {decomp_buf.tell() / 1_000_000:.1f} MB"
                    )
                    break

    decomp_buf.seek(0)
    print("Parsing CSV (may take a moment)…")
    df = pd.read_csv(
        decomp_buf,
        nrows=n_rows,
        low_memory=False,
        on_bad_lines="skip",
    )
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows → {output_path}")
    return str(output_path)


def download_bulk(
    output_path: str = "data/complaints.csv",
    nrows: Optional[int] = None,
) -> str:
    """Download and extract the complete CFPB bulk CSV zip (~200 MB compressed)."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Data already at {output_path} — skipping download.")
        return str(output_path)

    print(f"Downloading full dataset:\n  {CFPB_BULK_URL}")  # noqa: E501
    resp = requests.get(CFPB_BULK_URL, stream=True, timeout=600)
    resp.raise_for_status()

    import zipfile

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar:
        for chunk in resp.iter_content(chunk_size=131_072):
            buf.write(chunk)
            bar.update(len(chunk))

    buf.seek(0)
    print("Extracting archive…")
    with zipfile.ZipFile(buf) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, nrows=nrows, low_memory=False)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows → {output_path}")
    return str(output_path)


def fetch_api_sample(
    n_records: int = 25,
    output_path: str = "data/complaints_api.csv",
) -> str:
    """
    Fetch up to 25 narrated complaints from the CFPB public API.
    NOTE: The CFPB search API does not support pagination; use stream_bulk_nrows
    for larger samples.
    """
    output_path = Path(output_path)
    if output_path.exists():
        return str(output_path)

    resp = requests.get(
        CFPB_API_URL, params={"has_narrative": "true"}, timeout=30
    )
    resp.raise_for_status()
    d = resp.json()
    hits = d.get("hits", {}).get("hits", [])
    records = [h["_source"] for h in hits if h.get("_source", {}).get("complaint_what_happened")]

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows → {output_path}")
    return str(output_path)


def load_data(path: str, sample: Optional[int] = None) -> pd.DataFrame:
    """Load a complaints CSV, optionally down-sampling to *sample* rows."""
    df = pd.read_csv(path, low_memory=False)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df
