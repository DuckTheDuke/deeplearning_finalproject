"""Sanity-check the Hakka Wikipedia Parquet file before building the tone-restoration dataset.

Usage: python check_hakwiki.py path/to/train-00000-of-00001.parquet
"""
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

# Pha̍k-fa-sṳ tone marks (combining characters after NFD). U+0324 (the dots under ṳ) is NOT a tone.
TONE_MARKS = {"̂": "circumflex (陰平)", "̀": "grave (陽平)",
              "́": "acute (上聲)", "̍": "vertical line (陽入)"}
MARKUP = ["[[", "]]", "{{", "}}", "<ref", "'''", "Category:"]

df = pd.read_parquet(sys.argv[1])

print("== 1. Basic shape ==")
print(f"rows: {len(df):,}   columns: {list(df.columns)}")
print(f"empty texts: {(df['text'].str.strip() == '').sum():,}")
lengths = df["text"].str.len()
print(f"article length (chars): median {lengths.median():.0f}, mean {lengths.mean():.0f}, max {lengths.max():,}")

print("\n== 2. Two sample articles ==")
for _, row in df.sample(2, random_state=0).iterrows():
    print(f"--- {row['title']} ---\n{row['text'][:300]}\n")

print("== 3. Script mix ==")
all_text = "\n".join(df["text"])
han = len(re.findall(r"[一-鿿]", all_text))
latin = len(re.findall(r"[A-Za-z]", all_text))
print(f"Chinese characters: {han:,}   Latin letters: {latin:,}")
han_share = df["text"].apply(lambda t: len(re.findall(r"[一-鿿]", t)) / max(len(t), 1))
print(f"articles that are mostly Chinese characters (>50%): {(han_share > 0.5).sum():,}")

print("\n== 4. Tone marks ==")
nfd = unicodedata.normalize("NFD", all_text)
marks = Counter(ch for ch in nfd if ch in TONE_MARKS)
for ch, name in TONE_MARKS.items():
    print(f"U+{ord(ch):04X} {name:28s} {marks[ch]:>9,}")
print(f"ṳ (U+0324, keep this!):                 {nfd.count(chr(0x0324)):>9,}")

print("\n== 5. Syllable tokens ==")
tokens = re.findall(r"[A-Za-z̀-ͯ]+", nfd)  # hyphens split syllables
toned = sum(1 for t in tokens if any(m in t for m in TONE_MARKS))
print(f"latin syllable tokens: {len(tokens):,}   with a tone mark: {toned:,} ({toned / max(len(tokens), 1):.0%})")

print("\n== 6. Leftover wiki markup ==")
for m in MARKUP:
    n = df["text"].str.contains(m, regex=False).sum()
    print(f"{m!r:12s} in {n:,} articles")

print("\n== 7. Repeated sentences (template warning) ==")
sentences = [s.strip() for t in df["text"] for s in re.split(r"(?<=[.!?。])\s+", t) if len(s.strip()) > 20]
counts = Counter(sentences)
dup = sum(c for c in counts.values() if c > 1)
print(f"sentences: {len(sentences):,}   unique: {len(counts):,}   in duplicated groups: {dup:,}")
for s, c in counts.most_common(5):
    print(f"  x{c}: {s[:90]}")
