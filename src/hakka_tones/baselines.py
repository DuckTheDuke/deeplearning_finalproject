"""Non-deep baselines for tone restoration, scored with hakka_tones.evaluate.

1. syllable_majority  each toneless syllable gets its most common tone in train
2. word_majority      look up the whole hyphen-joined word first, fall back to (1)
3. linear_crf         sklearn-crfsuite with handcrafted context features (a CRF without a neural network)

Usage: uv run python -m hakka_tones.baselines [--data data/processed] [--out docs/baselines.md]
"""
import argparse
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import mlflow
import sklearn_crfsuite

from hakka_tones.data import looks_hakka
from hakka_tones.evaluate import TONE_TAGS, evaluate, load_split, train_word_set, word_keys


class SyllableMajority:
    def fit(self, records):
        counts = defaultdict(Counter)
        for rec in records:
            for syl, tag in zip(rec["syllables"], rec["tags"]):
                if tag != "X":
                    counts[syl][tag] += 1
        self.best = {syl: c.most_common(1)[0][0] for syl, c in counts.items()}
        self.default = Counter(t for rec in records for t in rec["tags"] if t != "X").most_common(1)[0][0]
        return self

    def predict_one(self, syl):
        return self.best.get(syl, self.default)

    def predict(self, records):
        return [[t if t == "X" else self.predict_one(s) for s, t in zip(r["syllables"], r["tags"])]
                for r in records]


class WordMajority:
    """Most common tag sequence for the whole word; unseen words back off to SyllableMajority."""

    def fit(self, records):
        self.backoff = SyllableMajority().fit(records)
        counts = defaultdict(Counter)
        for rec in records:
            for key, tags in self._words(rec):
                counts[key][tags] += 1
        self.best = {key: c.most_common(1)[0][0] for key, c in counts.items()}
        return self

    @staticmethod
    def _words(rec):
        """(word key, tuple of its tags) for each Hakka word, in order."""
        positions = defaultdict(list)
        for i, key in enumerate(word_keys(rec)):
            if key:
                positions[(rec["word_ids"][i], key)].append(i)
        return [(key, tuple(rec["tags"][i] for i in idx)) for (_, key), idx in positions.items()]

    def predict(self, records):
        out = []
        for rec in records:
            pred = list(rec["tags"])  # X positions stay X (never scored)
            positions = defaultdict(list)
            for i, key in enumerate(word_keys(rec)):
                if key:
                    positions[(rec["word_ids"][i], key)].append(i)
            for (_, key), idx in positions.items():
                tags = self.best.get(key) or [self.backoff.predict_one(rec["syllables"][i]) for i in idx]
                for i, tag in zip(idx, tags):
                    pred[i] = tag
            out.append(pred)
        return out


def final_class(syl):
    if syl[-1] in "ptk":
        return "stop"  # only "none" (陰入) or "vertical" (陽入) are possible
    return "nasal" if syl[-1] in "mn" or syl.endswith("ng") else "open"


def token_features(rec, i, keys):
    syls = rec["syllables"]
    syl = syls[i]
    wid = rec["word_ids"][i]
    word_positions = [j for j, w in enumerate(rec["word_ids"]) if w == wid]
    pos = word_positions.index(i)
    feats = {
        "bias": 1.0,
        "syl": syl,
        "final": final_class(syl),
        "last2": syl[-2:],
        "first2": syl[:2],
        "hakka_shape": looks_hakka(unicodedata.normalize("NFD", syl)),
        "cap": bool(rec["caps"][i]),
        "word": keys[i] or "<foreign>",
        "word_len": min(len(word_positions), 4),
        "pos_in_word": "S" if len(word_positions) == 1 else "B" if pos == 0 else
                       "E" if pos == len(word_positions) - 1 else "I",
    }
    for offset in (-2, -1, 1, 2):
        j = i + offset
        feats[f"syl[{offset}]"] = syls[j] if 0 <= j < len(syls) else ("<s>" if j < 0 else "</s>")
    feats["bigram[-1]"] = feats["syl[-1]"] + "|" + syl
    feats["bigram[+1]"] = syl + "|" + feats["syl[1]"]
    return feats


def sentence_features(rec):
    keys = word_keys(rec)
    return [token_features(rec, i, keys) for i in range(len(rec["syllables"]))]


class LinearCRF:
    def __init__(self, c1=0.1, c2=0.1, max_iterations=100):
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations)

    def fit(self, records):
        self.crf.fit([sentence_features(r) for r in records], [r["tags"] for r in records])
        return self

    def predict(self, records):
        return [list(tags) for tags in self.crf.predict([sentence_features(r) for r in records])]


BASELINES = {"syllable_majority": SyllableMajority, "word_majority": WordMajority, "linear_crf": LinearCRF}

COLUMNS = [("acc_all", "all"), ("acc_seen", "seen words"), ("acc_unseen", "unseen words"),
           ("acc_non_template", "non-template"), ("macro_f1", "macro-F1"), ("sentence_exact", "sentence exact")]


def results_table(results, split):
    lines = [f"| model | " + " | ".join(name for _, name in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for model, by_split in results.items():
        m = by_split[split]
        lines.append(f"| {model} | " + " | ".join(f"{m[k]:.1%}" for k, _ in COLUMNS) + " |")
    return lines


def per_tone_table(results, split):
    lines = ["| model | " + " | ".join(TONE_TAGS) + " |", "|---|" + "---|" * len(TONE_TAGS)]
    for model, by_split in results.items():
        lines.append(f"| {model} | " + " | ".join(f"{by_split[split][f'f1_{t}']:.3f}" for t in TONE_TAGS) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="docs/baselines.md")
    args = parser.parse_args()

    data = Path(args.data)
    splits = {name: load_split(data / f"{name}.jsonl") for name in ("train", "val", "test")}
    train_words = train_word_set(splits["train"])

    mlflow.set_experiment("baselines")
    results = {}
    for name, cls in BASELINES.items():
        start = time.time()
        model = cls().fit(splits["train"])
        results[name] = {s: evaluate(splits[s], model.predict(splits[s]), train_words) for s in ("val", "test")}
        seconds = time.time() - start
        with mlflow.start_run(run_name=name):
            for s, metrics in results[name].items():
                mlflow.log_metrics({f"{s}_{k}": v for k, v in metrics.items()})
            mlflow.log_metric("fit_predict_seconds", seconds)
        print(f"{name:18s} test acc {results[name]['test']['acc_all']:.1%}  "
              f"unseen {results[name]['test']['acc_unseen']:.1%}  ({seconds:.0f}s)")

    test = results["syllable_majority"]["test"]
    lines = ["# Baselines", "",
             "Non-deep baselines from `python -m hakka_tones.baselines`. Accuracy is over Hakka syllables only "
             "(foreign X tokens are not scored). \"Unseen words\" are hyphen-joined words that never appear in train.",
             "", f"Test set: {test['n_all']:,} syllables, {test['n_unseen']:,} in unseen words, "
             f"{test['n_non_template']:,} in non-template sentences.", "",
             "## Test", "", *results_table(results, "test"), "",
             "## Validation", "", *results_table(results, "val"), "",
             "## Per-tone F1 (test)", "", *per_tone_table(results, "test"), ""]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
