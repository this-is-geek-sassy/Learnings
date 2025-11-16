#!/usr/bin/env python3
import struct
from pathlib import Path
p = Path("sift/sift_base.fvecs")
if not p.exists():
    raise SystemExit("sift_base.fvecs not found in current folder.")

with p.open("rb") as f:
    data = f.read(4 + 128*4)

if len(data) < 4 + 128*4:
    raise SystemExit(f"File too short: {len(data)} bytes")

dim = struct.unpack('<i', data[0:4])[0]
print(f"Dimension header: {dim}\n")

print(f"{'idx':>3}  {'float':>12}   {'hex (little-endian)'}")
print("-"*48)
floats = []
for i in range(dim):
    chunk = data[4+4*i:4+4*(i+1)]
    val = struct.unpack('<f', chunk)[0]
    floats.append(val)
    hx = ' '.join(f"{b:02x}" for b in chunk)
    print(f"{i:3d}  {val:12.6f}   {hx}")

# Save compact Python list to file
out = Path("first_vector_decoded.txt")
with out.open("w") as fo:
    fo.write("[" + ", ".join(str(x) for x in floats) + "]\n")
print(f"\nSaved compact list to {out.resolve()}")

