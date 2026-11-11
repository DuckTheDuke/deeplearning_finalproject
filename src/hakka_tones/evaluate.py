"""Shared scoring for every model (baselines and neural), so all numbers are computed the same way.

Only Hakka syllables are scored: positions whose gold tag is "X" (foreign words) are skipped.
"""
import json
from collections import Counter

from hakka_tones.data import TAGS

TONE_TAGS = TAGS[:-1]  # everything except "X"


def load_split(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def word_keys(rec):
    """Per position: the toneless hyphen-joined word it belongs to ('fa-hok-ka'), or None for X tokens."""
    words = {}
    for syl, tag, wid in zip(rec["syllables"], rec["tags"], rec["word_ids"]):
        if tag != "X":
            words.setdefault(wid, []).append(syl)
    return ["-".join(words[wid]) if tag != "X" else None for tag, wid in zip(rec["tags"], rec["word_ids"])]


def train_word_set(train_records):
    return {key for rec in train_records for key in word_keys(rec) if key}


def evaluate(records, predictions, train_words):
    """Accuracy overall and on the subsets the experiments care about, plus per-tone F1."""
    correct, total = Counter(), Counter()
    tp, fp, fn = Counter(), Counter(), Counter()
    exact = 0
    for rec, pred in zip(records, predictions):
        assert len(pred) == len(rec["tags"]), "one predicted tag per syllable"
        all_right = True
        for gold, guess, key in zip(rec["tags"], pred, word_keys(rec)):
            if gold == "X":
                continue
            hit = gold == guess
            all_right &= hit
            groups = ["all", "seen" if key in train_words else "unseen",
                      "template" if rec["template"] else "non_template"]
            for g in groups:
                correct[g] += hit
                total[g] += 1
            if hit:
                tp[gold] += 1
            else:
                fn[gold] += 1
                fp[guess] += 1
        exact += all_right

    metrics = {f"acc_{g}": correct[g] / total[g] for g in total}
    metrics.update({f"n_{g}": total[g] for g in total})
    f1s = []
    for tag in TONE_TAGS:
        if not tp[tag] + fp[tag] + fn[tag]:
            continue  # tag never occurs and is never predicted: leave it out of the macro average
        precision = tp[tag] / max(tp[tag] + fp[tag], 1)
        recall = tp[tag] / max(tp[tag] + fn[tag], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics[f"f1_{tag}"] = f1
        f1s.append(f1)
    metrics["macro_f1"] = sum(f1s) / len(f1s)
    metrics["sentence_exact"] = exact / max(len(records), 1)
    return metrics
