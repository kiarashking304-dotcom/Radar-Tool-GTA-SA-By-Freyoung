"""
GTA SA Radar Tool - Streamlit Edition (v2)
==========================================
Rewrite of the tkinter tool (v6.0 by Tensho Records), improved.

Tabs
----
1. Merge Tiles  : radar00..radar143 (PNG *or TXD*) -> one big map.png (12x12 grid)
2. Split Map    : big map -> 144 tiles as PNG, TXD or both (ZIP), with progress bar
3. PNG -> TXD   : radar*.png -> RenderWare TXD (GTA SA / D3D9), optional zero-padded names
4. TXD -> PNG   : read TXD files (A8R8G8B8, X8R8G8B8, 24/16-bit, PAL8, DXT1/3/5) -> PNG
5. Color Replace: fast NumPy recolor, fast low-res live preview, full-res on demand,
                  dominant-color swatches, two blend modes
6. Help

Run with:  streamlit run app.py
Needs   :  streamlit, pillow, numpy   (see requirements.txt)
"""

from __future__ import annotations

import inspect
import io
import os
import re
import struct
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ================================
#   Constants
# ================================

APP_TITLE = "GTA SA Radar Tool"
APP_ICON = "🛰️"

DEFAULT_TILE_SIZE = 256
GRID_SIZE = 12
TOTAL_TILES = GRID_SIZE * GRID_SIZE
MAX_TILE_SIZE = 1024                  # per-tile cap
MAX_CANVAS_PIXELS = 160_000_000       # ~640 MB RGBA: hard cap for merged/split maps
MAX_PREVIEW_DIM = 1600                # preview downscale limit (downloads stay full-res)
LIVE_PREVIEW_DIM = 800                # color-replace live preview size

# --- RenderWare / TXD constants (same bytes as the original tool) ---
RW_VERSION = 0x1803FFFF
SECTION_STRUCT = 0x01
SECTION_EXTENSION = 0x03
SECTION_TEX_NATIVE = 0x15
SECTION_TEX_DICT = 0x16

PLATFORM_D3D9 = 9
RASTER_FORMAT = 0x0500       # rwRASTERFORMAT8888
FILTER_FLAGS = 0x1106
DEPTH = 32
MIPMAP_COUNT = 1
TEX_TYPE = 4
D3D9_FLAGS = 0x01            # bit0 = hasAlpha

D3DFMT_R8G8B8 = 20
D3DFMT_A8R8G8B8 = 21
D3DFMT_X8R8G8B8 = 22
D3DFMT_R5G6B5 = 23
D3DFMT_A1R5G5B5 = 25
FOURCC_DXT1 = 0x31545844
FOURCC_DXT3 = 0x33545844
FOURCC_DXT5 = 0x35545844
FORMAT_LABELS = {
    D3DFMT_R8G8B8: "R8G8B8", D3DFMT_A8R8G8B8: "A8R8G8B8", D3DFMT_X8R8G8B8: "X8R8G8B8",
    D3DFMT_R5G6B5: "R5G6B5", D3DFMT_A1R5G5B5: "A1R5G5B5",
    FOURCC_DXT1: "DXT1", FOURCC_DXT3: "DXT3", FOURCC_DXT5: "DXT5",
}

# radar00.png / radar7.txd ... (matched against the file's base name)
TILE_NAME_RE = re.compile(r"^radar(\d+)\.(png|txd)$", re.IGNORECASE)

MODE_SHIFT = "Shift (keeps shading)"
MODE_BLEND = "Blend to color (flat)"


# ================================
#   Small helpers
# ================================

def _to_rgba(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGBA" else image.convert("RGBA")


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _zip_bytes(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buf.getvalue()


def _thumb(image: Image.Image, max_dim: int = MAX_PREVIEW_DIM) -> Image.Image:
    if max(image.size) <= max_dim:
        return image
    thumb = image.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return thumb


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def _rgb_label(hex_str: str) -> str:
    return "rgb(%d, %d, %d)" % _hex_to_rgb(hex_str)


def _human_size(n: int) -> str:
    return f"{n / 1024:,.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:,.1f} MB"


def _stretch_kw(func) -> dict:
    """Full-width kwarg for a Streamlit widget, across Streamlit versions."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params and "stretch" in str(params["width"].annotation):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


# ================================
#   Tile helpers
# ================================

def tile_index(filename: str) -> Optional[int]:
    """'radar013.png' -> 13. None if not a radar tile or out of 0..143."""
    match = TILE_NAME_RE.match(os.path.basename(filename))
    if not match:
        return None
    index = int(match.group(1))
    return index if 0 <= index < TOTAL_TILES else None


def tile_position(index: int, tile_w: int, tile_h: int, flip_vertical: bool) -> tuple[int, int]:
    """Top-left pixel of tile `index` on the merged map (original logic)."""
    col = index % GRID_SIZE
    row = index // GRID_SIZE
    if flip_vertical:
        row = GRID_SIZE - 1 - row
    return col * tile_w, row * tile_h


def canvas_problem(tile_w: int, tile_h: int) -> Optional[str]:
    """Error text if a 12x12 canvas of this tile size is too big, else None."""
    w, h = tile_w * GRID_SIZE, tile_h * GRID_SIZE
    if w * h > MAX_CANVAS_PIXELS:
        return (f"A {w}×{h} map (~{w * h * 4 / 1024 ** 2:,.0f} MB in memory) is over the "
                f"safety limit. Use a smaller tile size.")
    return None


def draw_grid(image: Image.Image, flip_vertical: bool) -> Image.Image:
    """Overlay the 12x12 grid and tile numbers on a preview image."""
    img = image.convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    cw, ch = w / GRID_SIZE, h / GRID_SIZE
    for k in range(GRID_SIZE + 1):
        draw.line([(k * cw, 0), (k * cw, h)], fill=(255, 255, 0, 170), width=1)
        draw.line([(0, k * ch), (w, k * ch)], fill=(255, 255, 0, 170), width=1)
    if cw >= 28:
        for i in range(TOTAL_TILES):
            row = i // GRID_SIZE
            if flip_vertical:
                row = GRID_SIZE - 1 - row
            x, y = (i % GRID_SIZE) * cw + 3, row * ch + 2
            draw.text((x + 1, y + 1), str(i), fill=(0, 0, 0, 255))
            draw.text((x, y), str(i), fill=(255, 255, 0, 255))
    return img


def tile_status_grid(placed: set[int], flip_vertical: bool) -> str:
    """12x12 emoji grid laid out like the merged map (green = present)."""
    rows = []
    for r in range(GRID_SIZE):
        src = GRID_SIZE - 1 - r if flip_vertical else r
        rows.append("".join("🟩" if src * GRID_SIZE + c in placed else "⬜"
                            for c in range(GRID_SIZE)))
    return "\n".join(rows)


# ================================
#   TXD writer
# ================================

def _rw_section(section_type: int, data: bytes) -> bytes:
    return struct.pack("<III", section_type, len(data), RW_VERSION) + data


def safe_texture_name(name: str) -> str:
    """ASCII-only, 31 chars max, no extension - what GTA SA texture names allow."""
    name = re.sub(r"\.(png|txd)$", "", os.path.basename(name), flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    return name[:31] or "texture"


def make_txd(texture_name: str, image: Image.Image) -> bytes:
    """Minimal RenderWare TXD (GTA SA, D3D9, A8R8G8B8, no mipmaps)."""
    img = image.convert("RGBA")
    r, g, b, a = img.split()
    pixel_data = Image.merge("RGBA", (b, g, r, a)).tobytes()   # RGBA -> BGRA
    w, h = img.size

    name_bytes = texture_name.encode("ascii")[:31].ljust(32, b"\x00")
    native_struct = (
        struct.pack("<II", PLATFORM_D3D9, FILTER_FLAGS)
        + name_bytes + b"\x00" * 32
        + struct.pack("<IIHHBBBBI", RASTER_FORMAT, D3DFMT_A8R8G8B8, w, h,
                      DEPTH, MIPMAP_COUNT, TEX_TYPE, D3D9_FLAGS, len(pixel_data))
        + pixel_data
    )
    tex_native = _rw_section(
        SECTION_TEX_NATIVE,
        _rw_section(SECTION_STRUCT, native_struct) + _rw_section(SECTION_EXTENSION, b""),
    )
    dict_body = (
        _rw_section(SECTION_STRUCT, struct.pack("<HH", 1, 2))
        + tex_native
        + _rw_section(SECTION_EXTENSION, b"")
    )
    return _rw_section(SECTION_TEX_DICT, dict_body)


def txd_texture_name(filename: str, zero_pad: bool) -> str:
    base = re.sub(r"\.png$", "", os.path.basename(filename), flags=re.IGNORECASE)
    if zero_pad:
        m = re.fullmatch(r"radar(\d+)", base, re.IGNORECASE)
        if m:
            base = f"radar{int(m.group(1)):02d}"
    return safe_texture_name(base)


# ================================
#   TXD reader
# ================================

@dataclass
class TxdTexture:
    name: str
    width: int
    height: int
    fmt: str
    mip_levels: int
    image: Image.Image


def _dds_wrap(fourcc: int, w: int, h: int, payload: bytes) -> bytes:
    """Wrap raw DXT data in a DDS header so Pillow can decode it."""
    header = struct.pack("<4s7I", b"DDS ", 124, 0x81007, h, w, len(payload), 0, 0)
    header += b"\x00" * 44                                       # reserved[11]
    header += struct.pack("<II I 5I", 32, 4, fourcc, 0, 0, 0, 0, 0)  # pixel format
    header += struct.pack("<5I", 0x1000, 0, 0, 0, 0)             # caps
    return header + payload


def _decode_mip0(d3d_format: int, w: int, h: int, payload: bytes,
                 palette: Optional[bytes]) -> Image.Image:
    if palette is not None:
        idx = np.frombuffer(payload, np.uint8, w * h).reshape(h, w)
        pal = np.frombuffer(palette, np.uint8).reshape(256, 4)[:, [2, 1, 0, 3]]
        return Image.fromarray(np.ascontiguousarray(pal[idx]))
    if d3d_format in (D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8):
        arr = np.frombuffer(payload, np.uint8, w * h * 4).reshape(h, w, 4)[..., [2, 1, 0, 3]].copy()
        if d3d_format == D3DFMT_X8R8G8B8:
            arr[..., 3] = 255
        return Image.fromarray(arr)
    if d3d_format == D3DFMT_R8G8B8:
        arr = np.frombuffer(payload, np.uint8, w * h * 3).reshape(h, w, 3)[..., ::-1]
        return Image.fromarray(np.ascontiguousarray(arr)).convert("RGBA")
    if d3d_format in (D3DFMT_R5G6B5, D3DFMT_A1R5G5B5):
        v = np.frombuffer(payload, "<u2", w * h).reshape(h, w).astype(np.uint32)
        if d3d_format == D3DFMT_R5G6B5:
            r, g, b, a = (v >> 11) & 31, (v >> 5) & 63, v & 31, np.full_like(v, 255)
            r, g, b = r * 255 // 31, g * 255 // 63, b * 255 // 31
        else:
            a = (v >> 15) * 255
            r, g, b = ((v >> 10) & 31) * 255 // 31, ((v >> 5) & 31) * 255 // 31, (v & 31) * 255 // 31
        return Image.fromarray(np.dstack([r, g, b, a]).astype(np.uint8))
    if d3d_format in (FOURCC_DXT1, FOURCC_DXT3, FOURCC_DXT5):
        block = 8 if d3d_format == FOURCC_DXT1 else 16
        need = ((w + 3) // 4) * ((h + 3) // 4) * block
        if len(payload) < need:
            raise ValueError("compressed data is truncated")
        dds = _dds_wrap(d3d_format, w, h, payload[:need])
        return Image.open(io.BytesIO(dds)).convert("RGBA")
    raise ValueError(f"unsupported pixel format {d3d_format:#x}")


def _parse_native(data: bytes, start: int) -> TxdTexture:
    sec_type, _, _ = struct.unpack_from("<III", data, start)
    if sec_type != SECTION_STRUCT:
        raise ValueError("texture has no struct section")
    p = start + 12
    if p + 92 > len(data):
        raise ValueError("file is truncated")
    platform = struct.unpack_from("<I", data, p)[0]
    if platform != PLATFORM_D3D9:
        raise ValueError(f"platform {platform} is not D3D9 (only GTA SA PC textures are supported)")
    name = data[p + 8:p + 40].split(b"\x00")[0].decode("ascii", "replace")
    raster, d3d_format, w, h, _depth, mips, _rtype, _flags = struct.unpack_from("<IIHHBBBB", data, p + 72)
    q = p + 88
    palette = None
    if raster & 0x2000:                       # PAL8
        palette = data[q:q + 1024]
        q += 1024
    elif raster & 0x4000:                     # PAL4
        raise ValueError("4-bit palette textures are not supported")
    if q + 4 > len(data):
        raise ValueError("file is truncated")
    size = struct.unpack_from("<I", data, q)[0]
    payload = data[q + 4:q + 4 + size]
    if len(payload) < size:
        raise ValueError("pixel data is truncated")
    image = _decode_mip0(d3d_format, w, h, payload, palette)
    fmt = "PAL8" if palette is not None else FORMAT_LABELS.get(d3d_format, hex(d3d_format))
    return TxdTexture(name, w, h, fmt, max(1, mips), image)


def parse_txd(data: bytes) -> list[TxdTexture]:
    """Decode every texture (first mip level) in a GTA SA TXD."""
    if len(data) < 28 or struct.unpack_from("<I", data, 0)[0] != SECTION_TEX_DICT:
        raise ValueError("not a TXD file")
    end = min(len(data), 12 + struct.unpack_from("<I", data, 4)[0])
    pos = 12
    st_type, st_size, _ = struct.unpack_from("<III", data, pos)
    if st_type != SECTION_STRUCT:
        raise ValueError("damaged TXD header")
    count = struct.unpack_from("<H", data, pos + 12)[0]
    pos += 12 + st_size
    textures: list[TxdTexture] = []
    while pos + 12 <= end and len(textures) < count:
        sec_type, size, _ = struct.unpack_from("<III", data, pos)
        if sec_type == SECTION_TEX_NATIVE:
            textures.append(_parse_native(data, pos + 12))
        pos += 12 + size
    if not textures:
        raise ValueError("no textures found")
    return textures


# ================================
#   Feature 1 - Merge
# ================================

@dataclass
class MergeReport:
    placed: set = field(default_factory=set)
    ignored: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    resized: int = 0


def _load_tile(upload) -> Image.Image:
    raw = upload.getvalue()
    if upload.name.lower().endswith(".txd"):
        return _to_rgba(parse_txd(raw)[0].image)
    return _to_rgba(Image.open(io.BytesIO(raw)))


def merge_uploads(uploads: list, tile_w: int, tile_h: int, flip_vertical: bool,
                  background: tuple = (0, 0, 0, 0)) -> tuple[Image.Image, MergeReport]:
    """Stitch uploaded tiles one at a time (keeps memory low)."""
    canvas = Image.new("RGBA", (tile_w * GRID_SIZE, tile_h * GRID_SIZE), background)
    report = MergeReport()
    for upload in uploads:
        index = tile_index(upload.name)
        if index is None:
            report.ignored.append(upload.name)
            continue
        try:
            img = _load_tile(upload)
        except Exception as exc:
            report.ignored.append(f"{upload.name} ({exc})")
            continue
        if index in report.placed:
            report.duplicates.append(upload.name)
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            report.resized += 1
        canvas.paste(img, tile_position(index, tile_w, tile_h, flip_vertical))
        report.placed.add(index)
    return canvas, report


# ================================
#   Feature 2 - Split
# ================================

def split_to_zip(map_image: Image.Image, tile_w: int, tile_h: int, flip_vertical: bool,
                 make_png: bool, make_txd_files: bool,
                 progress: Optional[Callable[[int, int], None]] = None) -> bytes:
    """Cut the map into 144 tiles and zip them as PNG and/or TXD."""
    map_image = _to_rgba(map_image)
    target = (tile_w * GRID_SIZE, tile_h * GRID_SIZE)
    if map_image.size != target:
        map_image = map_image.resize(target, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(TOTAL_TILES):
            x, y = tile_position(i, tile_w, tile_h, flip_vertical)
            tile = map_image.crop((x, y, x + tile_w, y + tile_h))
            name = f"radar{i:02d}"
            if make_png:
                zf.writestr(f"{name}.png", _png_bytes(tile))
            if make_txd_files:
                zf.writestr(f"{name}.txd", make_txd(name, tile))
            if progress:
                progress(i + 1, TOTAL_TILES)
    return buf.getvalue()


# ================================
#   Feature 3 / 4 - PNG <-> TXD
# ================================

def convert_pngs_to_txd(uploads: list, zero_pad: bool,
                        progress: Optional[Callable[[int, int], None]] = None):
    """Returns (zip entries, table rows, error messages)."""
    results: list[tuple[str, bytes]] = []
    rows: list[dict] = []
    errors: list[str] = []
    used: set[str] = set()
    for n, upload in enumerate(uploads, 1):
        try:
            name = txd_texture_name(upload.name, zero_pad)
            if name.lower() in used:
                raise ValueError(f"duplicate texture name '{name}' - skipped")
            img = Image.open(io.BytesIO(upload.getvalue()))
            data = make_txd(name, img)
            used.add(name.lower())
            results.append((name + ".txd", data))
            rows.append({"Source": upload.name, "Texture": name,
                         "Size": f"{img.size[0]}×{img.size[1]}", "TXD": _human_size(len(data))})
        except Exception as exc:
            errors.append(f"{upload.name}: {exc}")
        if progress:
            progress(n, len(uploads))
    return results, rows, errors


def extract_txds(uploads: list):
    """Returns (zip entries, table rows, previews, errors)."""
    files: list[tuple[str, bytes]] = []
    rows: list[dict] = []
    previews: list[tuple[str, Image.Image]] = []
    errors: list[str] = []
    used: set[str] = set()
    for upload in uploads:
        stem = os.path.splitext(os.path.basename(upload.name))[0]
        try:
            for tex in parse_txd(upload.getvalue()):
                fname = f"{tex.name}.png"
                if fname.lower() in used:
                    fname = f"{stem}_{tex.name}.png"
                used.add(fname.lower())
                files.append((fname, _png_bytes(tex.image)))
                rows.append({"TXD": upload.name, "Texture": tex.name,
                             "Size": f"{tex.width}×{tex.height}", "Format": tex.fmt,
                             "Mips": tex.mip_levels})
                if len(previews) < 8:
                    previews.append((tex.name, _thumb(tex.image, 256)))
        except Exception as exc:
            errors.append(f"{upload.name}: {exc}")
    return files, rows, previews, errors


# ================================
#   Feature 5 - Color Replace
# ================================

def recolor_image(image: Image.Image, target_hex: str, replace_hex: str,
                  tolerance: int, mode: str) -> Image.Image:
    """Soft color replace, vectorized with NumPy and processed in row chunks.

    mask = clamp(1 - distance(pixel, target) / tolerance, 0, 1)
    SHIFT : out = pixel + (replace - target) * mask     (original behaviour)
    BLEND : out = pixel + (replace - pixel)  * mask     (pulls toward a flat color)
    """
    arr = np.asarray(_to_rgba(image), dtype=np.uint8)
    out = arr.copy()
    t = np.array(_hex_to_rgb(target_hex), np.float32)
    r = np.array(_hex_to_rgb(replace_hex), np.float32)
    tol = float(max(1e-6, tolerance))
    for y0 in range(0, arr.shape[0], 256):
        rgb = arr[y0:y0 + 256, :, :3].astype(np.float32)
        dist = np.sqrt(((rgb - t) ** 2).sum(axis=-1))
        mask = np.clip(1.0 - dist / tol, 0.0, 1.0)[..., None]
        res = rgb + ((r - t) if mode == MODE_SHIFT else (r - rgb)) * mask
        out[y0:y0 + 256, :, :3] = np.clip(res + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


@st.cache_data(show_spinner=False, max_entries=4)
def recolor_png(image_bytes: bytes, target_hex: str, replace_hex: str,
                tolerance: int, mode: str, max_dim: int = 0) -> bytes:
    image = _to_rgba(Image.open(io.BytesIO(image_bytes)))
    if max_dim:
        image = _thumb(image, max_dim)
    return _png_bytes(recolor_image(image, target_hex, replace_hex, tolerance, mode))


@st.cache_data(show_spinner=False, max_entries=2)
def preview_png(image_bytes: bytes, max_dim: int) -> bytes:
    return _png_bytes(_thumb(_to_rgba(Image.open(io.BytesIO(image_bytes))), max_dim))


@st.cache_data(show_spinner=False, max_entries=2)
def dominant_colors(image_bytes: bytes, n: int = 8) -> list[str]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((256, 256))
    q = img.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = sorted(q.getcolors() or [], reverse=True)
    return ["#%02x%02x%02x" % tuple(pal[i * 3:i * 3 + 3]) for _, i in counts]


def _set_target(hex_value: str) -> None:
    st.session_state["target_hex"] = hex_value


# ================================
#   UI
# ================================

HELP_TEXT = """
### 🛰️ About GTA SA radar files
The radar is built from **144 square tiles** in a **12×12 grid**, named
`radar00` … `radar143`, left-to-right, top-to-bottom. In the game each tile is
its own `radarNN.txd` holding one texture `radarNN`.

### Typical workflow
1. **TXD → PNG**: extract the game's original `radarNN.txd` files.
2. **Merge Tiles**: drop the PNGs *or the TXDs directly* to get one big map.
3. Edit the map in any image editor (or use **Color Replace**).
4. **Split Map** with output **TXD** (or PNG + TXD) and put the files back in the game.

### Notes
- **Flip Vertical** reverses the row order. Use the same setting when splitting
  that you used when merging.
- Tile size defaults to 256 (game standard). Tiles of another size are resized.
- Texture names are limited to 31 ASCII characters. *Zero-pad* turns `radar5` into `radar07`-style names.
- TXD reading supports A8R8G8B8 / X8R8G8B8 / R8G8B8 / 565 / 1555 / PAL8 / DXT1 / DXT3 / DXT5
  (first mip level). Writing always produces uncompressed A8R8G8B8.
- **Privacy:** files are processed in memory by the machine running this app. If you run it
  locally nothing leaves your PC; if it is hosted, uploads go to that server.

*Streamlit edition - based on RadarTool v6.0 by Tensho Records.*
"""


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    BTN, DL, IMG = _stretch_kw(st.button), _stretch_kw(st.download_button), _stretch_kw(st.image)

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Merge · Split · Convert · Extract · Recolor - GTA San Andreas radar maps")

    tab_merge, tab_split, tab_txd, tab_extract, tab_color, tab_help = st.tabs(
        ["🧩 Merge Tiles", "✂️ Split Map", "📦 PNG → TXD", "📤 TXD → PNG",
         "🎨 Color Replace", "❓ Help"])

    # ---------- Merge ----------
    with tab_merge:
        st.markdown("Upload radar tiles (`radar00` … `radar143`, **PNG or TXD**) and stitch them "
                    "into one **map.png**.")
        uploads = st.file_uploader("Radar tiles", type=["png", "txd"],
                                   accept_multiple_files=True, key="merge_uploads")
        c1, c2, c3 = st.columns(3)
        tile_w = c1.number_input("Tile width (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="merge_w")
        tile_h = c2.number_input("Tile height (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="merge_h")
        fill = c3.selectbox("Missing tiles", ["Transparent", "Black", "White"], key="merge_fill")
        o1, o2 = st.columns(2)
        flip = o1.checkbox("🔃 Flip Vertical", key="merge_flip",
                           help="Mirror the row order - use if the map is upside-down in game.")
        show_grid = o2.checkbox("🔢 Show tile grid on preview", key="merge_grid")

        problem = canvas_problem(int(tile_w), int(tile_h))
        if problem:
            st.error(problem)

        if st.button("🧩 Merge Tiles", type="primary", disabled=not uploads or bool(problem), **BTN):
            try:
                bg = {"Transparent": (0, 0, 0, 0), "Black": (0, 0, 0, 255),
                      "White": (255, 255, 255, 255)}[fill]
                with st.spinner("Merging tiles…"):
                    canvas, rep = merge_uploads(uploads, int(tile_w), int(tile_h), flip, bg)
                if not rep.placed:
                    st.error("No usable radar tiles found. Names must look like `radar00.png` … `radar143.png`.")
                else:
                    warnings = []
                    if rep.ignored:
                        warnings.append("Ignored: " + ", ".join(rep.ignored))
                    if rep.duplicates:
                        warnings.append("Duplicate tile numbers (last one wins): " + ", ".join(rep.duplicates))
                    if rep.resized:
                        warnings.append(f"{rep.resized} tile(s) were resized to {tile_w}×{tile_h}.")
                    st.session_state.merge_result = {
                        "png": _png_bytes(canvas), "thumb": _png_bytes(_thumb(canvas)),
                        "size": canvas.size, "placed": rep.placed, "flip": flip, "warnings": warnings}
            except Exception as exc:
                st.error(f"Merge failed: {exc}")
        elif not uploads:
            st.info("Upload at least one radar tile to get started.")

        res = st.session_state.get("merge_result")
        if res:
            st.success(f"map.png built ({res['size'][0]}×{res['size'][1]}) from "
                       f"{len(res['placed'])}/{TOTAL_TILES} tiles.")
            for w in res["warnings"]:
                st.warning(w)
            preview = Image.open(io.BytesIO(res["thumb"]))
            st.image(draw_grid(preview, res["flip"]) if show_grid else preview, **IMG)
            if len(res["placed"]) < TOTAL_TILES:
                missing = [str(i) for i in range(TOTAL_TILES) if i not in res["placed"]]
                with st.expander(f"{len(missing)} missing tiles"):
                    st.code(tile_status_grid(res["placed"], res["flip"]), language=None)
                    st.caption("Missing: " + ", ".join(missing))
            st.download_button("⬇️ Download map.png", res["png"], "map.png", "image/png",
                               key="merge_download", **DL)

    # ---------- Split ----------
    with tab_split:
        st.markdown("Slice one large map into exactly **144 tiles** (`radar00` … `radar143`), as a ZIP.")
        map_upload = st.file_uploader("Map image", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                      key="split_upload")
        c1, c2, c3 = st.columns(3)
        s_w = c1.number_input("Tile width (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="split_w")
        s_h = c2.number_input("Tile height (px)", 1, MAX_TILE_SIZE, DEFAULT_TILE_SIZE, key="split_h")
        out_fmt = c3.radio("Output", ["PNG", "TXD", "PNG + TXD"], horizontal=True, key="split_fmt")
        s_flip = st.checkbox("🔃 Flip Vertical", key="split_flip",
                             help="Exact inverse of 'Flip Vertical' when merging.")

        map_img = None
        s_problem = canvas_problem(int(s_w), int(s_h))
        if s_problem:
            st.error(s_problem)
        elif map_upload:
            try:
                map_img = Image.open(io.BytesIO(map_upload.getvalue()))
                mw, mh = map_img.size
                st.caption(f"Loaded **{map_upload.name}** - {mw}×{mh} px.")
                if mw * mh > MAX_CANVAS_PIXELS:
                    st.error("This image is too large to process safely.")
                    map_img = None
                else:
                    expected = (int(s_w) * GRID_SIZE, int(s_h) * GRID_SIZE)
                    if map_img.size != expected:
                        st.info(f"The map will be resized to {expected[0]}×{expected[1]} (12 × tile size).")
                    if abs(mw / mh - 1) > 0.01:
                        st.warning("The map is not square - it will be stretched to a square.")
            except Exception as exc:
                st.error(f"Could not read image: {exc}")
                map_img = None

        if st.button("✂️ Split Map", type="primary", disabled=map_img is None, **BTN):
            bar = st.progress(0.0, text="Starting…")
            try:
                data = split_to_zip(
                    map_img, int(s_w), int(s_h), s_flip,
                    make_png=out_fmt in ("PNG", "PNG + TXD"),
                    make_txd_files=out_fmt in ("TXD", "PNG + TXD"),
                    progress=lambda i, n: bar.progress(i / n, text=f"Tile {i}/{n}"))
                st.session_state.split_result = {"zip": data, "info": f"{s_w}×{s_h} · {out_fmt}"}
            except Exception as exc:
                st.error(f"Split failed: {exc}")
            bar.empty()

        sres = st.session_state.get("split_result")
        if sres:
            st.success(f"All {TOTAL_TILES} tiles exported ({sres['info']}, {_human_size(len(sres['zip']))}).")
            st.download_button("⬇️ Download radar_tiles.zip", sres["zip"], "radar_tiles.zip",
                               "application/zip", key="split_download", **DL)

    # ---------- PNG -> TXD ----------
    with tab_txd:
        st.markdown("Convert PNG files into **RenderWare TXD** textures for GTA San Andreas "
                    "(D3D9, A8R8G8B8).")
        txd_uploads = st.file_uploader("PNG files", type=["png"], accept_multiple_files=True,
                                       key="txd_uploads")
        zero_pad = st.checkbox("Zero-pad tile numbers (radar5 → radar05)", value=True, key="txd_pad")
        if st.button("📦 Convert to TXD", type="primary", disabled=not txd_uploads, **BTN):
            bar = st.progress(0.0, text="Converting…")
            results, rows, errors = convert_pngs_to_txd(
                txd_uploads, zero_pad, lambda i, n: bar.progress(i / n, text=f"{i}/{n}"))
            bar.empty()
            st.session_state.txd_result = {"zip": _zip_bytes(results) if results else None,
                                           "rows": rows, "errors": errors}
        tres = st.session_state.get("txd_result")
        if tres:
            if tres["zip"]:
                st.success(f"Converted {len(tres['rows'])} PNG file(s) to TXD.")
                st.dataframe(tres["rows"], hide_index=True)
                st.download_button("⬇️ Download txd_files.zip", tres["zip"], "txd_files.zip",
                                   "application/zip", key="txd_download", **DL)
            if tres["errors"]:
                st.error("Some files could not be converted:\n\n" + "\n".join(f"- {e}" for e in tres["errors"]))
        elif not txd_uploads:
            st.info("Upload one or more PNG files to convert.")

    # ---------- TXD -> PNG ----------
    with tab_extract:
        st.markdown("Extract the textures from **TXD** files (for example the game's original "
                    "`radarNN.txd`) into PNG.")
        ex_uploads = st.file_uploader("TXD files", type=["txd"], accept_multiple_files=True,
                                      key="extract_uploads")
        if st.button("📤 Extract to PNG", type="primary", disabled=not ex_uploads, **BTN):
            with st.spinner("Reading TXD files…"):
                files, rows, previews, errors = extract_txds(ex_uploads)
            st.session_state.extract_result = {"zip": _zip_bytes(files) if files else None,
                                               "rows": rows, "previews": previews, "errors": errors}
        eres = st.session_state.get("extract_result")
        if eres:
            if eres["zip"]:
                st.success(f"Extracted {len(eres['rows'])} texture(s).")
                st.dataframe(eres["rows"], hide_index=True)
                cols = st.columns(4)
                for i, (name, img) in enumerate(eres["previews"]):
                    with cols[i % 4]:
                        st.image(img, caption=name, **IMG)
                st.download_button("⬇️ Download png_files.zip", eres["zip"], "png_files.zip",
                                   "application/zip", key="extract_download", **DL)
            if eres["errors"]:
                st.error("Some files could not be read:\n\n" + "\n".join(f"- {e}" for e in eres["errors"]))
        elif not ex_uploads:
            st.info("Upload one or more TXD files.")

    # ---------- Color Replace ----------
    with tab_color:
        st.markdown("Pick a **target color**, choose a **replacement** and watch the preview. "
                    "Pixels near the target blend smoothly toward the replacement.")
        color_upload = st.file_uploader("Map image", type=["png"], key="color_upload")
        st.session_state.setdefault("target_hex", "#888888")
        st.session_state.setdefault("replace_hex", "#ff0000")

        image_bytes = None
        if color_upload:
            image_bytes = color_upload.getvalue()
            try:
                size = Image.open(io.BytesIO(image_bytes)).size
                if size[0] * size[1] > MAX_CANVAS_PIXELS:
                    st.error("This image is too large to process safely.")
                    image_bytes = None
                else:
                    st.caption(f"Loaded **{color_upload.name}** - {size[0]}×{size[1]} px.")
            except Exception as exc:
                st.error(f"Could not read image: {exc}")
                image_bytes = None

        ct, cr = st.columns(2)
        with ct:
            st.color_picker("🎯 Target color", key="target_hex")
            st.caption(_rgb_label(st.session_state.target_hex))
        with cr:
            st.color_picker("🎨 Replacement color", key="replace_hex")
            st.caption(_rgb_label(st.session_state.replace_hex))

        if image_bytes:
            colors = dominant_colors(image_bytes)
            st.caption("Dominant colors in the image - click one to use it as the target:")
            for col, hx in zip(st.columns(len(colors)), colors):
                with col:
                    st.markdown(f'<div style="background:{hx};height:26px;border-radius:6px;'
                                f'border:1px solid #8884"></div>', unsafe_allow_html=True)
                    st.button(hx, key=f"swatch_{hx}", on_click=_set_target, args=(hx,))

        tolerance = st.slider("Tolerance", 1, 150, 40,
                              help="1 = near-exact matches only. 150 = very wide, soft blend.")
        mode = st.radio("Mode", [MODE_SHIFT, MODE_BLEND], horizontal=True,
                        help="Shift keeps shading and detail. Blend pulls pixels to one flat color.")

        if image_bytes:
            tgt, rep = st.session_state.target_hex, st.session_state.replace_hex
            left, right = st.columns(2)
            with left:
                st.markdown("**Original**")
                st.image(preview_png(image_bytes, LIVE_PREVIEW_DIM), **IMG)
            with right:
                st.markdown("**Recolored** (preview)")
                st.image(recolor_png(image_bytes, tgt, rep, tolerance, mode, LIVE_PREVIEW_DIM), **IMG)

            key = (color_upload.name, len(image_bytes), tgt, rep, tolerance, mode)
            if st.button("🖼️ Render full resolution", **BTN):
                with st.spinner("Rendering full-resolution image…"):
                    st.session_state.color_full = {
                        "key": key, "png": recolor_png(image_bytes, tgt, rep, tolerance, mode, 0)}
            full = st.session_state.get("color_full")
            if full and full["key"] == key:
                st.download_button("⬇️ Download map_recolored.png", full["png"],
                                   "map_recolored.png", "image/png", key="color_download", **DL)
            elif full:
                st.caption("Settings changed - render full resolution again to download.")
        else:
            st.info("Upload a map image to preview recoloring live.")

    # ---------- Help ----------
    with tab_help:
        st.markdown(HELP_TEXT)

    st.divider()
    st.caption("🛰️ GTA SA Radar Tool v2 · 12×12 grid · 144 tiles · Streamlit + Pillow + NumPy")


if st.runtime.exists():
    main()
