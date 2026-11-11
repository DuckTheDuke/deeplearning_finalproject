"""Turn the Hakka Wikipedia dump into sentence-level tone-tagging data.

Each output line (JSONL) is one sentence:
  syllables  toneless, lowercased syllables — the model input
  tags       one tone tag per syllable — the target ("X" = not Hakka, never scored)
  caps       1 if the syllable was capitalized in the original text
  word_ids   which hyphen-joined word each syllable belongs to (for the word-lookup baseline)
  template   True if the same sentence appears in >= TEMPLATE_MIN_ARTICLES articles

Usage: uv run python -m hakka_tones.data [--input data/hakwiki.parquet] [--out data/processed]
"""
import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Pha̍k-fa-sṳ tone marks as combining characters. U+0324 (the dots under ṳ) is part of the letter, not a tone.
TONE_TAGS = {"̂": "circumflex", "̀": "grave", "́": "acute", "̍": "vertical"}
TONE_MARKS = set(TONE_TAGS)
TAGS = ["circumflex", "grave", "acute", "vertical", "none", "X"]

# Pha̍k-fa-sṳ letters: no b, d, j, q, r, w, x, z, and g only in "ng". Tone marks sit on a vowel or a syllabic m/n.
HAKKA_LETTERS = re.compile(r"(?:ng|[acefhiklmnopstuvy̤])+")
TONE_CARRIERS = set("aeioumn")

LETTER = r"(?:[^\W\d_]|[̀-ͯ])"  # \w misses combining marks like U+030D, so add them
WORD = re.compile(rf"{LETTER}+(?:-{LETTER}+)*")  # digits split "1943-ngièn" into just "ngièn"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")
SENTENCE_PUNCT = re.compile(r"[.!?。！？]")
EMPTY_PARENS = re.compile(r"\(\s*\)|（\s*）")

MIN_HAKKA_SYLLABLES = 3
MIN_MARKED_SHARE = 0.2  # Hakka text is ~60% tone-marked; English that slips past looks_hakka is ~0%
MAX_SYLLABLE_LETTERS = 7  # chhiung, chhiang
TEMPLATE_MIN_ARTICLES = 5
MAX_HAN_SHARE = 0.5


def looks_hakka(token_nfd):
    """Rough shape filter for one syllable; foreign names usually break one of these rules."""
    tok = token_nfd.lower()  # casing says nothing: sentence starts and place names are capitalized
    marks = [i for i, ch in enumerate(tok) if ch in TONE_MARKS]
    base = "".join(ch for ch in tok if ch not in TONE_MARKS)
    if not 1 <= len(base.replace("̤", "")) <= MAX_SYLLABLE_LETTERS or not HAKKA_LETTERS.fullmatch(base):
        return False
    if len(marks) > 1:  # Frédéric, Pyrénées, Cầu
        return False
    carrier = tok[:marks[0]].replace("̤", "")[-1:] if marks else ""  # ṳ́ is u, U+0324, then the tone
    return not marks or carrier in TONE_CARRIERS


def tone_tag(token_nfd):
    marks = [ch for ch in token_nfd if ch in TONE_MARKS]
    return TONE_TAGS[marks[0]] if marks else "none"


def strip_tones(text):
    """Remove only the four tone marks; NFC afterwards so ṳ stays one character."""
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in nfd if ch not in TONE_MARKS))


def syllable_info(raw):
    """raw syllable as written -> (model input, tone tag, was capitalized)."""
    nfd = unicodedata.normalize("NFD", raw)
    tag = tone_tag(nfd) if looks_hakka(nfd) else "X"
    return strip_tones(raw).lower(), tag, int(raw[:1].isupper())


def han_share(text):
    han = len(re.findall(r"[一-鿿]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return han / max(han + latin, 1)


def clean_lines(text):
    text = unicodedata.normalize("NFC", EMPTY_PARENS.sub("", text.replace("‹", "").replace("›", "")))
    return [" ".join(line.split()) for line in text.split("\n") if line.strip()]


def is_boilerplate(line, repeated_lines):
    """Section titles and footers ("Li̍t-sṳ́", "Ngoi-phu lièn-kiet"): no sentence punctuation, many articles.

    Unique lines without punctuation (lists, names) are real Hakka and are kept.
    """
    return line in repeated_lines and not SENTENCE_PUNCT.search(line)


def repeated_lines(texts):
    articles_per_line = Counter(line for text in texts for line in set(clean_lines(text)))
    return {line for line, n in articles_per_line.items() if n >= TEMPLATE_MIN_ARTICLES}


def is_hakka_sentence(rec):
    tags = [t for t in rec["tags"] if t != "X"]
    marked = sum(t != "none" for t in tags)
    return len(tags) >= MIN_HAKKA_SYLLABLES and marked / len(tags) >= MIN_MARKED_SHARE


def sentence_record(sentence):
    syllables, tags, caps, word_ids = [], [], [], []
    for word_id, word in enumerate(WORD.findall(sentence)):
        for raw in filter(None, word.split("-")):
            syl, tag, cap = syllable_info(raw)
            syllables.append(syl)
            tags.append(tag)
            caps.append(cap)
            word_ids.append(word_id)
    return {"syllables": syllables, "tags": tags, "caps": caps, "word_ids": word_ids, "raw": sentence}


def article_sentences(text, repeated=frozenset()):
    for line in clean_lines(text):
        if is_boilerplate(line, repeated):
            continue
        for sentence in SENTENCE_SPLIT.split(line):
            rec = sentence_record(sentence.strip())
            if is_hakka_sentence(rec):
                yield rec


def load_articles(parquet_path):
    """Read the dump, drop exact-duplicate and mostly-Chinese-character articles."""
    df = pd.read_parquet(parquet_path)
    n_raw = len(df)
    df = df.drop_duplicates("text")
    n_dedup = len(df)
    df = df[df["text"].map(han_share) <= MAX_HAN_SHARE]
    stats = {"raw": n_raw, "duplicates_dropped": n_raw - n_dedup, "han_dropped": n_dedup - len(df)}
    return df, stats


def split_articles(article_ids, seed=0, fractions=(0.8, 0.1, 0.1)):
    """Split by article (never by sentence) so no article's text lands in two splits."""
    ids = sorted(article_ids)
    random.Random(seed).shuffle(ids)
    n_train = int(fractions[0] * len(ids))
    n_val = int(fractions[1] * len(ids))
    return {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}


def flag_templates(records):
    """Mark sentences that recur across many articles (bot stubs); they are scored separately."""
    articles_per_sentence = defaultdict(set)
    for rec in records:
        articles_per_sentence[" ".join(rec["syllables"])].add(rec["article_id"])
    for rec in records:
        rec["template"] = len(articles_per_sentence[" ".join(rec["syllables"])]) >= TEMPLATE_MIN_ARTICLES


def build(parquet_path, seed=0):
    df, stats = load_articles(parquet_path)
    repeated = repeated_lines(df["text"])
    records = []
    for article_id, title, text in zip(df["id"], df["title"], df["text"]):
        for i, rec in enumerate(article_sentences(text, repeated)):
            records.append({"article_id": article_id, "title": title, "sent_id": i, **rec})
    stats["articles_with_sentences"] = len({r["article_id"] for r in records})
    flag_templates(records)
    splits = split_articles({r["article_id"] for r in records}, seed=seed)
    split_of = {a: name for name, ids in splits.items() for a in ids}
    by_split = {name: [r for r in records if split_of[r["article_id"]] == name] for name in splits}
    return by_split, stats


def words_of(rec):
    """Toneless hyphen-joined Hakka words in a sentence, e.g. 'fa-hok-ka'."""
    words = defaultdict(list)
    for syl, tag, wid in zip(rec["syllables"], rec["tags"], rec["word_ids"]):
        if tag != "X":
            words[wid].append(syl)
    return ["-".join(sylls) for sylls in words.values()]


def data_card(by_split, stats, seed):
    lines = ["# Data card: Hakka Wikipedia tone tagging", "",
             "Source: `wikimedia/wikipedia`, config `20231101.hak` (Hugging Face).",
             f"Built by `python -m hakka_tones.data` with seed {seed}; split by article 80/10/10.", "",
             "## Article filtering", "",
             f"- articles in dump: {stats['raw']:,}",
             f"- exact-duplicate texts dropped: {stats['duplicates_dropped']:,}",
             f"- mostly Chinese characters (>{MAX_HAN_SHARE:.0%} Han) dropped: {stats['han_dropped']:,}",
             f"- articles with at least one usable sentence: {stats['articles_with_sentences']:,}", "",
             "## Splits", "",
             "| split | articles | sentences | Hakka syllables | X tokens | template sentences |",
             "|---|---|---|---|---|---|"]
    for name, recs in by_split.items():
        tags = Counter(t for r in recs for t in r["tags"])
        hakka = sum(v for k, v in tags.items() if k != "X")
        lines.append(f"| {name} | {len({r['article_id'] for r in recs}):,} | {len(recs):,} | {hakka:,} | "
                     f"{tags['X']:,} | {sum(r['template'] for r in recs) / max(len(recs), 1):.1%} |")

    lines += ["", "## Tone tags (Hakka syllables only)", "", "| tag | " + " | ".join(by_split) + " |",
              "|---|" + "---|" * len(by_split)]
    counts = {name: Counter(t for r in recs for t in r["tags"] if t != "X") for name, recs in by_split.items()}
    for tag in TAGS[:-1]:
        cells = [f"{counts[n][tag] / max(sum(counts[n].values()), 1):.1%}" for n in by_split]
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    train_words = {w for r in by_split["train"] for w in words_of(r)}
    test_words = [w for r in by_split["test"] for w in words_of(r)]
    unseen = [w for w in test_words if w not in train_words]
    unseen_syl = sum(w.count("-") + 1 for w in unseen)
    all_syl = sum(w.count("-") + 1 for w in test_words)
    lines += ["", "## Seen vs. unseen words (test set)", "",
              f"- test words (hyphen-joined): {len(test_words):,}; not seen in train: {len(unseen):,} "
              f"({len(unseen) / max(len(test_words), 1):.1%})",
              f"- test syllables inside unseen words: {unseen_syl:,} ({unseen_syl / max(all_syl, 1):.1%})", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/hakwiki.parquet")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--card", default="docs/data_card.md")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    by_split, stats = build(args.input, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in by_split.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {out / f'{name}.jsonl'}: {len(recs):,} sentences")

    card = Path(args.card)
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(data_card(by_split, stats, args.seed), encoding="utf-8")
    print(f"wrote {card}")


if __name__ == "__main__":
    main()
