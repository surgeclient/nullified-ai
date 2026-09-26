"""Placeholder textures for mod models that reference textures the mod doesn't ship.

The AI writes code and JSON, not art, so without this every new item/block renders as the
purple-and-black missing texture. Each placeholder is a 16x16 PNG with a color derived from its name.
"""
import hashlib
import json
import re
import struct
import zlib

RES = "src/main/resources/assets"
TEXTURE_REF = re.compile(r"^(?:(?P<ns>[a-z0-9_.-]+):)?(?P<path>[a-z0-9_./-]+)$")


def png_16(name: str) -> bytes:
    """A 16x16 RGBA PNG: name-colored fill, darker border, light diagonal so it reads as a placeholder."""
    h = hashlib.md5(name.encode()).digest()
    base = (80 + h[0] % 150, 80 + h[1] % 150, 80 + h[2] % 150)
    rows = []
    for y in range(16):
        row = bytearray([0])  # filter type 0
        for x in range(16):
            if x in (0, 15) or y in (0, 15):
                r, g, b = (c // 2 for c in base)
            elif x == y:
                r, g, b = (min(255, c + 60) for c in base)
            else:
                shade = 1.0 - 0.02 * (x + y) / 2
                r, g, b = (int(c * shade) for c in base)
            row += bytes((r, g, b, 255))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 16, 16, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


def missing_textures(files: dict[str, str]) -> list[str]:
    """Resource paths of mod-namespace textures referenced by models but not present."""
    missing = []
    for path, body in files.items():
        if not (path.startswith(RES) and "/models/" in path and path.endswith(".json")):
            continue
        try:
            textures = json.loads(body).get("textures", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        for ref in textures.values() if isinstance(textures, dict) else []:
            m = TEXTURE_REF.match(ref) if isinstance(ref, str) and not ref.startswith("#") else None
            if not m or (m.group("ns") or "minecraft") == "minecraft":
                continue
            png = f"{RES}/{m.group('ns')}/textures/{m.group('path')}.png"
            if png not in files and png not in missing:
                missing.append(png)
    return missing


def placeholder_textures(files: dict[str, str]) -> dict[str, bytes]:
    return {png: png_16(png) for png in missing_textures(files)}
