"""Download and convert enwik8 and hg38 into $YGGDRASIL_DATA (default data/).

    python -m tasks.prepare enwik8 hg38
"""
import argparse
import gzip
import hashlib
import json
import urllib.request
import zipfile

import numpy as np

from .text import DATA, N_TOKEN, SPLIT_FILE, VOCAB

ENWIK8 = ("http://mattmahoney.net/dc/enwik8.zip", "a1fa5ffddb56f4953e226637dabbb36a")  # md5 of the unzipped file
HG38_FASTA = ("https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz", "014fd36b333691315c8f4a8a7b2570ff")
HG38_BED = ("https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed", "2c256f1c667fe3318865303248e70659")

MD5_BLOCK = 1 << 24
BASES = b"ACGT"
LUT = np.full(256, N_TOKEN, np.uint8)  # byte -> token, upper and lower case
for i, c in enumerate(BASES):
    LUT[c] = LUT[c + 32] = i


def md5(path):
    """Hex md5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(MD5_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, digest=None):
    """Download url to dest unless dest exists with the right md5; returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and (digest is None or md5(dest) == digest)):
        print("download", url)
        urllib.request.urlretrieve(url, dest)
    if digest is not None and md5(dest) != digest:
        raise RuntimeError(f"md5 mismatch for {dest}")
    return dest


def prepare_enwik8():
    """data/enwik8: the unzipped 100M bytes."""
    out = DATA / "enwik8"
    if out.exists() and md5(out) == ENWIK8[1]:
        return
    with zipfile.ZipFile(fetch(ENWIK8[0], DATA / "raw" / "enwik8.zip")) as z:
        out.write_bytes(z.read("enwik8"))
    if md5(out) != ENWIK8[1]:
        raise RuntimeError(f"md5 mismatch for {out}")


def read_fasta(path):
    """Gzipped FASTA -> {chrom: uint8 array}."""
    with gzip.open(path, "rb") as f:
        text = f.read()
    genome = {}
    for rec in text.split(b">")[1:]:
        head, _, body = rec.partition(b"\n")
        genome[head.split()[0].decode()] = np.frombuffer(body.replace(b"\n", b"").replace(b"\r", b""), np.uint8)
    return genome


def union_length(iv):
    """Length of the union of half-open (chrom, start, end) intervals."""
    total, last = 0, {}
    for c, s, e in sorted(iv):
        lo = max(s, last.get(c, s))
        total += max(0, e - lo)
        last[c] = max(e, last.get(c, e))
    return total


def write_split(genome, intervals, path):
    """Write the tokens of the (chrom, start, end) intervals to path in order; returns the count of each token."""
    counts = np.zeros(VOCAB["hg38"], np.int64)
    with open(path, "wb") as f:
        for c, s, e in intervals:
            if s < 0 or e > len(genome[c]):
                raise RuntimeError(f"interval {c}:{s}-{e} out of bounds")
            codes = LUT[genome[c][s:e]]
            counts += np.bincount(codes, minlength=len(counts))
            f.write(codes.tobytes())
    return counts


def prepare_hg38():
    """Each split is its BED intervals sliced from hg38.ml.fa and concatenated in file order, case-insensitive."""
    raw = DATA / "raw" / "hg38_basenji"
    fa = fetch(HG38_FASTA[0], raw / "hg38.ml.fa.gz", HG38_FASTA[1])
    bed = fetch(HG38_BED[0], raw / "sequences_human.bed", HG38_BED[1])  # the intervals HyenaDNA loads
    rows = [ln.split("\t") for ln in bed.read_text().splitlines() if ln.strip()]
    by_split = {s: [(r[0], int(r[1]), int(r[2])) for r in rows if r[3].strip() == s] for s in SPLIT_FILE}
    genome = read_fasta(fa)
    out = DATA / "hg38"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"vocab": VOCAB["hg38"], "tokens": f"ACGT = 0..3, other = {N_TOKEN}", "splits": {}}
    for split, iv in by_split.items():
        counts = write_split(genome, iv, out / f"{split}.u8")
        meta["splits"][split] = {"bases": int(counts.sum()), "intervals": len(iv),
                                 "counts": dict(zip("ACGTN", map(int, counts)))}
        print(split, int(counts.sum()))
    meta["overlap_bases"] = {f"{a}&{b}": union_length(by_split[a]) + union_length(by_split[b])
                             - union_length(by_split[a] + by_split[b])
                             for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="download and convert the text and DNA datasets")
    ap.add_argument("names", nargs="+", choices=("enwik8", "hg38"))
    for name in ap.parse_args().names:
        {"enwik8": prepare_enwik8, "hg38": prepare_hg38}[name]()
