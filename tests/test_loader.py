import json

import torch

from hakka_tones.data import TAGS, sentence_record
from hakka_tones.loader import (PAD, TAG_TO_ID, UNK, BucketBatchSampler, TaggingDataset, Vocab, build_vocabs,
                                collate, get_loaders, ids_to_tags, load_vocabs, save_vocabs)


def rec(text, article_id="a"):
    return {**sentence_record(text), "template": False, "article_id": article_id}


TRAIN = [rec("Kì he yit-chak fa-ho̍k-kâ."), rec("Kì chhai Nobel Fa-ho̍k-kâ."), rec("Sṳ̀n he sṳ́.")]


def test_vocab_padding_and_unknown():
    v = Vocab(["he", "ki"])
    assert v["<pad>"] == PAD == 0 and v["never-seen"] == UNK == 1
    assert v["he"] == 2 and len(v) == 4


def test_vocabs_built_from_train_with_min_count():
    syllables, chars = build_vocabs(TRAIN, min_syllable_count=2)
    assert syllables["ki"] > UNK and syllables["fa"] > UNK  # seen twice
    assert syllables["chhai"] == UNK                       # seen once -> <unk>, char CNN still sees it
    assert chars["ṳ"] > UNK                           # ṳ is one character
    assert chars["b"] == UNK                               # only from Hakka syllables, not "Nobel"


def test_vocab_round_trip(tmp_path):
    syllables, chars = build_vocabs(TRAIN)
    save_vocabs(syllables, chars, tmp_path / "vocab.json")
    s2, c2 = load_vocabs(tmp_path / "vocab.json")
    assert s2.itos == syllables.itos and c2.itos == chars.itos


def test_collate_pads_and_masks():
    syllables, chars = build_vocabs(TRAIN, min_syllable_count=1)
    ds = TaggingDataset(TRAIN, syllables, chars)
    batch = collate([ds[0], ds[2]])
    n0, n2 = len(TRAIN[0]["syllables"]), len(TRAIN[2]["syllables"])
    assert batch["syllables"].shape == (2, max(n0, n2))
    assert batch["chars"].shape[:2] == (2, max(n0, n2))
    assert batch["mask"].sum(1).tolist() == [n0, n2]
    assert (batch["syllables"][1, n2:] == PAD).all() and (batch["chars"][1, n2:] == PAD).all()
    assert batch["index"].tolist() == [0, 2]
    # characters of the first syllable "ki", then padding
    assert batch["chars"][0, 0, :2].tolist() == [chars["k"], chars["i"]]
    assert (batch["chars"][0, 0, 2:] == PAD).all()


def test_hakka_mask_excludes_foreign_tokens():
    syllables, chars = build_vocabs(TRAIN)
    batch = collate([TaggingDataset(TRAIN, syllables, chars)[1]])  # "Kì chhai Nobel Fa-ho̍k-kâ."
    assert batch["tags"][0].tolist()[2] == TAG_TO_ID["X"]
    assert batch["hakka_mask"][0].tolist() == [True, True, False, True, True, True]
    assert ids_to_tags(batch["tags"][0].tolist()) == TRAIN[1]["tags"]


def test_bucket_sampler_covers_every_sentence_once():
    lengths = [((i * 37) % 50) + 1 for i in range(1000)]
    for shuffle in (True, False):
        batches = list(BucketBatchSampler(lengths, 32, shuffle=shuffle, pool=5))
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(1000))
        assert len(batches) == len(BucketBatchSampler(lengths, 32)) == 32


def test_bucket_sampler_groups_similar_lengths_and_is_seeded():
    lengths = [((i * 37) % 50) + 1 for i in range(1000)]
    batches = list(BucketBatchSampler(lengths, 32, pool=10, seed=0))
    spread = sum(max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches) / len(batches)
    assert spread < 10  # random batches of 32 would spread ~45
    assert batches == list(BucketBatchSampler(lengths, 32, pool=10, seed=0))
    assert batches != list(BucketBatchSampler(lengths, 32, pool=10, seed=1))


def test_get_loaders_end_to_end(tmp_path):
    for name in ("train", "val", "test"):
        with open(tmp_path / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(TRAIN * 4):
                f.write(json.dumps({**r, "article_id": f"{name}{i}"}, ensure_ascii=False) + "\n")
    loaders, syllables, _ = get_loaders(tmp_path, batch_size=5)
    seen = sorted(i for batch in loaders["val"] for i in batch["index"].tolist())
    assert seen == list(range(12))
    small, _, _ = get_loaders(tmp_path, batch_size=5, train_fraction=0.5)
    assert len(small["train"].dataset) == 6 and len(small["val"].dataset) == 12
    batch = next(iter(loaders["train"]))
    assert batch["tags"].max() < len(TAGS) and batch["syllables"].dtype == torch.long
