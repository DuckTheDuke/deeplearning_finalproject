"""Plot the Hakka Wikipedia data: tone balance, tone ambiguity, article sizes, and repeated lines.

Usage: uv run python visualize_hakwiki.py path/to/train-00000-of-00001.parquet [out_dir]
Writes PNGs to out_dir (default: figures/).
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Pha̍k-fa-sṳ tone marks (combining characters after NFD). U+0324 (the dots under ṳ) is NOT a tone.
TONES = [  # (combining mark, label); order fixes each tone's color
    ("̂", "â  circumflex · 陰平"),
    ("̀", "à  grave · 陽平"),
    ("́", "á  acute · 上聲"),
    ("̍", "a̍  vertical line · 陽入"),
    (None, "a  no mark · 去聲 / 陰入"),
]
TONE_MARKS = {mark for mark, _ in TONES if mark}

# Reference data-viz palette, light mode: categorical slots 1-5 in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Arial Unicode MS"],  # second font covers the Chinese tone names
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def tone_of(token_nfd):
    marks = [ch for ch in token_nfd if ch in TONE_MARKS]
    return marks[0] if marks else None


def base_of(token_nfd):
    """Toneless, lowercased syllable — the model's input form. Keeps ṳ."""
    return unicodedata.normalize("NFC", "".join(ch for ch in token_nfd if ch not in TONE_MARKS)).lower()


# Pha̍k-fa-sṳ letters: no b, d, j, q, r, w, x, z, and g only in "ng". Tone marks sit on a vowel or a syllabic m/n.
HAKKA_LETTERS = re.compile(r"(?:ng|[acefhiklmnopstuvy̤])+")
TONE_CARRIERS = set("aeioumn")


def looks_hakka(token_nfd):
    """Rough shape filter for one syllable; foreign names usually break one of these rules."""
    tok = token_nfd.lower()  # casing says nothing: sentence starts and place names are capitalized
    marks = [i for i, ch in enumerate(tok) if ch in TONE_MARKS]
    base = "".join(ch for ch in tok if ch not in TONE_MARKS)
    if not 1 <= len(base.replace("̤", "")) <= 6 or not HAKKA_LETTERS.fullmatch(base):
        return False
    if len(marks) > 1:  # Frédéric, Pyrénées, Cầu
        return False
    carrier = tok[:marks[0]].replace("̤", "")[-1:] if marks else ""  # ṳ́ is u, U+0324, then the tone
    return not marks or carrier in TONE_CARRIERS


def finish(fig, out_dir, name):
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def plot_tone_balance(tone_counts, out_dir):
    labels = [label for _, label in TONES]
    counts = [tone_counts[mark] for mark, _ in TONES]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, height=0.55, color=SERIES[0])
    for yi, c in zip(y, counts):
        ax.text(c + total * 0.005, yi, f"{c:,}  ({c / total:.0%})", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(y, labels)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("syllable tokens")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_title("How often each tone mark appears", loc="left")
    finish(fig, out_dir, "1_tone_balance.png")


def plot_ambiguity(base_tones, out_dir, top_n=25):
    totals = Counter({b: sum(c.values()) for b, c in base_tones.items()})
    top = [b for b, _ in totals.most_common(top_n)]
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(top))[::-1]
    left = np.zeros(len(top))
    for (mark, label), color in zip(TONES, SERIES):
        share = np.array([base_tones[b][mark] / totals[b] for b in top])
        # 2px surface gap between stacked segments
        ax.barh(y, share, left=left, height=0.7, color=color, label=label, edgecolor=SURFACE, linewidth=1.5)
        left += share
    ax.set_yticks(y, [f"{b}  ({totals[b]:,})" for b in top])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of occurrences")
    ax.set_title(f"Tone split for the {top_n} most common toneless syllables", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=9)
    finish(fig, out_dir, "2_tone_ambiguity_top_syllables.png")


def plot_ambiguity_summary(base_tones, out_dir, min_share=0.05):
    # Weighted by tokens: what share of running text has a syllable seen with 1, 2, 3... different tones?
    # A tone only counts if it covers >= min_share of the syllable's uses, so stray typos don't inflate this.
    by_n = Counter()
    for counts in base_tones.values():
        total = sum(counts.values())
        n_tones = sum(1 for c in counts.values() if c / total >= min_share)
        by_n[n_tones] += total
    total = sum(by_n.values())
    ns = sorted(by_n)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([str(n) for n in ns], [by_n[n] / total for n in ns], width=0.55, color=SERIES[0])
    for i, n in enumerate(ns):
        ax.text(i, by_n[n] / total + 0.01, f"{by_n[n] / total:.0%}", ha="center", color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(f"number of tones the syllable takes (each ≥ {min_share:.0%} of its uses)")
    ax.set_ylabel("share of syllable tokens")
    ax.set_title("How ambiguous is each syllable's tone?", loc="left")
    finish(fig, out_dir, "3_tone_ambiguity_summary.png")


def plot_article_lengths(syllables_per_article, out_dir):
    s = np.array(syllables_per_article)
    s = s[s > 0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.logspace(0, np.log10(s.max()), 40)
    ax.hist(s, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    med = np.median(s)
    ax.axvline(med, color=INK_2, linewidth=1)
    ax.text(med * 0.94, ax.get_ylim()[1] * 0.92, f"median {med:.0f}", color=INK_2, fontsize=9, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Hakka syllable tokens per article (log scale)")
    ax.set_ylabel("articles")
    ax.set_title("Most articles are short stubs", loc="left")
    finish(fig, out_dir, "4_article_lengths.png")


def plot_repeated_lines(df, out_dir, top_n=12):
    lines = Counter(line.strip() for t in df["text"] for line in t.split("\n") if len(line.strip()) > 3)
    top = lines.most_common(top_n)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(top))[::-1]
    ax.barh(y, [c for _, c in top], height=0.6, color=SERIES[0])
    ax.set_yticks(y, [(s[:40] + "…") if len(s) > 40 else s for s, _ in top], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("number of articles containing the line")
    ax.set_title("Most repeated lines (headings and templates to remove)", loc="left")
    finish(fig, out_dir, "5_repeated_lines.png")


def main():
    df = pd.read_parquet(sys.argv[1])
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    tone_counts = Counter()
    base_tones = defaultdict(Counter)
    syllables_per_article = []
    for text in df["text"]:
        n = 0
        for tok in re.findall(r"[A-Za-z̀-ͯ]+", unicodedata.normalize("NFD", text)):
            if not looks_hakka(tok):
                continue
            tone = tone_of(tok)
            tone_counts[tone] += 1
            base_tones[base_of(tok)][tone] += 1
            n += 1
        syllables_per_article.append(n)

    print(f"articles: {len(df):,}   Hakka-looking syllable tokens: {sum(tone_counts.values()):,}   "
          f"distinct toneless syllables: {len(base_tones):,}")
    plot_tone_balance(tone_counts, out_dir)
    plot_ambiguity(base_tones, out_dir)
    plot_ambiguity_summary(base_tones, out_dir)
    plot_article_lengths(syllables_per_article, out_dir)
    plot_repeated_lines(df, out_dir)


if __name__ == "__main__":
    main()
