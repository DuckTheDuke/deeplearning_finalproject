"""Vocabularies, Dataset, padding, and length-bucketed batches for the neural models.

A batch is a dict of tensors (B = sentences, T = longest sentence, C = longest syllable in characters):
  syllables   (B, T)     long   syllable ids (0 = padding, 1 = unknown)
  chars       (B, T, C)  long   character ids per syllable, for the char CNN
  caps        (B, T)     long   1 if the syllable was capitalized
  tags        (B, T)     long   tag ids in TAGS order; padding positions hold 0 but are masked
  mask        (B, T)     bool   real tokens (what the CRF needs)
  hakka_mask  (B, T)     bool   real Hakka syllables, i.e. not X (what gets scored)
  index       (B,)       long   position of each sentence in the split, to line predictions up with records

X (foreign) tokens stay in the sequence with their own tag so the CRF sees whole sentences; they are never scored.
"""
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from hakka_tones.data import TAGS
from hakka_tones.evaluate import load_split

PAD, UNK = 0, 1
MAX_CHARS = 12  # Hakka syllables have at most 7 letters; only long foreign words get truncated
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}


class Vocab:
    """String <-> id map with padding at 0 and unknown at 1."""

    def __init__(self, tokens):
        self.itos = ["<pad>", "<unk>", *tokens]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def build(cls, counts, min_count=1):
        return cls(sorted(tok for tok, n in counts.items() if n >= min_count))

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, UNK)


def build_vocabs(train_records, min_syllable_count=2):
    """Built from TRAIN ONLY. Syllables seen once in train become <unk>, so the model learns an
    <unk> embedding it can use for unseen syllables at test time (the char CNN still sees their letters).
    Characters come from Hakka syllables only; otherwise every Chinese character and foreign script in the
    X tokens would get its own id (~2,800 instead of ~30)."""
    syllables = Counter(s for r in train_records for s in r["syllables"])
    chars = Counter(ch for r in train_records for s, t in zip(r["syllables"], r["tags"]) if t != "X" for ch in s)
    return Vocab.build(syllables, min_syllable_count), Vocab.build(chars)


def save_vocabs(syllable_vocab, char_vocab, path):
    Path(path).write_text(json.dumps({"syllables": syllable_vocab.itos[2:], "chars": char_vocab.itos[2:]},
                                     ensure_ascii=False), encoding="utf-8")


def load_vocabs(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    return Vocab(saved["syllables"]), Vocab(saved["chars"])


class TaggingDataset(Dataset):
    """Converts each sentence record to id tensors once, up front."""

    def __init__(self, records, syllable_vocab, char_vocab):
        self.records = records
        self.items = []
        for i, rec in enumerate(records):
            self.items.append({
                "index": i,
                "syllables": torch.tensor([syllable_vocab[s] for s in rec["syllables"]]),
                "chars": [torch.tensor([char_vocab[ch] for ch in s[:MAX_CHARS]]) for s in rec["syllables"]],
                "caps": torch.tensor(rec["caps"]),
                "tags": torch.tensor([TAG_TO_ID[t] for t in rec["tags"]]),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def lengths(self):
        return [len(item["syllables"]) for item in self.items]


def collate(items):
    """Pad a list of dataset items into one batch (padding on the right)."""
    B = len(items)
    T = max(len(it["syllables"]) for it in items)
    C = max(len(ch) for it in items for ch in it["chars"])
    batch = {
        "syllables": torch.full((B, T), PAD),
        "chars": torch.full((B, T, C), PAD),
        "caps": torch.zeros(B, T, dtype=torch.long),
        "tags": torch.zeros(B, T, dtype=torch.long),
        "mask": torch.zeros(B, T, dtype=torch.bool),
        "index": torch.tensor([it["index"] for it in items]),
    }
    for b, it in enumerate(items):
        n = len(it["syllables"])
        batch["syllables"][b, :n] = it["syllables"]
        batch["caps"][b, :n] = it["caps"]
        batch["tags"][b, :n] = it["tags"]
        batch["mask"][b, :n] = True
        for t, ch in enumerate(it["chars"]):
            batch["chars"][b, t, :len(ch)] = ch
    batch["hakka_mask"] = batch["mask"] & (batch["tags"] != TAG_TO_ID["X"])
    return batch


class BucketBatchSampler(Sampler):
    """Batches of similar-length sentences, so little compute is spent on padding.

    Training: shuffle, cut into pools of `pool` batches, sort each pool by length, batch, shuffle the batches.
    Evaluation (shuffle=False): sort everything by length; the batch["index"] field maps results back.
    """

    def __init__(self, lengths, batch_size, shuffle=True, pool=50, seed=0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = pool
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if not self.shuffle:
            idx.sort(key=lambda i: self.lengths[i])
            yield from (idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size))
            return
        self.rng.shuffle(idx)
        pool_size = self.batch_size * self.pool
        batches = []
        for start in range(0, len(idx), pool_size):
            chunk = sorted(idx[start:start + pool_size], key=lambda i: self.lengths[i])
            batches += [chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        self.rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_loader(dataset, batch_size=32, shuffle=True, seed=0):
    sampler = BucketBatchSampler(dataset.lengths(), batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)


def get_loaders(data_dir="data/processed", batch_size=32, seed=0, train_fraction=1.0):
    """Train/val/test loaders plus the vocabularies.

    train_fraction < 1 keeps that share of training ARTICLES (for the learning-curve experiment);
    the vocabulary is built from whatever training data is kept.
    """
    data_dir = Path(data_dir)
    splits = {name: load_split(data_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    if train_fraction < 1.0:
        articles = sorted({r["article_id"] for r in splits["train"]})
        random.Random(seed).shuffle(articles)
        keep = set(articles[:int(train_fraction * len(articles))])
        splits["train"] = [r for r in splits["train"] if r["article_id"] in keep]

    syllable_vocab, char_vocab = build_vocabs(splits["train"])
    datasets = {name: TaggingDataset(recs, syllable_vocab, char_vocab) for name, recs in splits.items()}
    loaders = {name: make_loader(ds, batch_size, shuffle=(name == "train"), seed=seed)
               for name, ds in datasets.items()}
    return loaders, syllable_vocab, char_vocab


def ids_to_tags(tag_ids):
    """One sentence's predicted tag ids -> tag strings, for hakka_tones.evaluate."""
    return [TAGS[i] for i in tag_ids]
