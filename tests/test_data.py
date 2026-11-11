import unicodedata

import pandas as pd
import pytest

from hakka_tones.data import (TONE_MARKS, article_sentences, build, is_boilerplate, looks_hakka,
                              repeated_lines, sentence_record, split_articles, syllable_info)


def nfd(s):
    return unicodedata.normalize("NFD", s)


@pytest.mark.parametrize("raw, base, tag, cap", [
    ("lî", "li", "circumflex", 0),
    ("ngìn", "ngin", "grave", 0),
    ("chióng", "chiong", "acute", 0),
    ("Li̍t", "lit", "vertical", 1),     # U+030D has no precomposed form
    ("he", "he", "none", 0),
    ("sṳ́", "sṳ", "acute", 0),          # tone on ṳ; the dots under ṳ must survive
    ("ńg", "ng", "acute", 0),           # syllabic ng
    ("chhiùng", "chhiung", "grave", 0),  # 7 letters, the longest Hakka syllables
    ("Frédéric", "frederic", "X", 1),  # foreign name: accents stripped from input, never scored
    ("Cầu", "cau", "X", 1),            # two marks -> not Hakka
    ("Abruzzo", "abruzzo", "X", 1),    # letters Hakka spelling doesn't use
])
def test_syllable_info(raw, base, tag, cap):
    assert syllable_info(raw) == (base, tag, cap)


def test_u_with_dots_stays_one_character():
    base, _, _ = syllable_info("sṳ́")
    assert base == "sṳ" and len(base) == 2


@pytest.mark.parametrize("word", ["Paramaribo", "Weinberg", "Gdynia", "Pyrénées"])
def test_foreign_words_rejected(word):
    assert not looks_hakka(nfd(word))


def test_numbers_split_off_and_words_grouped():
    rec = sentence_record("Kì chhai 1943-ngièn tet-to Nobel Fa-ho̍k-chióng.")
    assert rec["syllables"] == ["ki", "chhai", "ngien", "tet", "to", "nobel", "fa", "hok", "chiong"]
    assert rec["tags"] == ["grave", "none", "grave", "none", "none", "X", "none", "vertical", "acute"]
    assert rec["word_ids"] == [0, 1, 2, 3, 3, 4, 5, 5, 5]
    assert rec["caps"][0] == 1 and rec["caps"][6] == 1


def test_no_tone_marks_leak_into_input():
    text = "Mario Molina (1943-ngièn 3-ngie̍t 19-ngit) he Me̍t-sî-kô lâu Mî-koet ke yit-chak fa-ho̍k-kâ."
    for rec in article_sentences(text):
        for syl in rec["syllables"]:
            assert not set(nfd(syl)) & TONE_MARKS, syl


def test_headings_and_junk_removed():
    text = "Li̍t-sṳ́\n\nThi-lî\n‹\n›\nKì () he yit-chak fa-ho̍k-kâ. Kì chhai Mî-koet。Ngìn-khiéu he thai-to.\n"
    sents = list(article_sentences(text))
    assert [s["raw"] for s in sents] == ["Kì he yit-chak fa-ho̍k-kâ.", "Kì chhai Mî-koet。",
                                         "Ngìn-khiéu he thai-to."]


def test_repeated_headings_dropped_but_unique_lists_kept():
    texts = [f"Li̍t-sṳ́ thi-lî hì-hèu\nKì he yit-chak fa-ho̍k-kâ {i}." for i in range(5)]
    repeated = repeated_lines(texts)
    assert is_boilerplate("Li̍t-sṳ́ thi-lî hì-hèu", repeated)
    assert not is_boilerplate("Koet-li̍p Thòi-vân Thí-yuk Thai-ho̍k", repeated)  # unique: keep
    assert not is_boilerplate("Kì he yit-chak fa-ho̍k-kâ 0.", repeated)         # has punctuation
    raws = [s["raw"] for s in article_sentences(texts[0] + "\nKoet-li̍p Thòi-vân Thí-yuk Thai-ho̍k", repeated)]
    assert raws == ["Kì he yit-chak fa-ho̍k-kâ 0.", "Koet-li̍p Thòi-vân Thí-yuk Thai-ho̍k"]


def test_english_sentences_dropped():
    text = "Among Us is a 2018 online multiplayer social deduction game. Kì he yit-chak fa-ho̍k-kâ."
    assert [s["raw"] for s in article_sentences(text)] == ["Kì he yit-chak fa-ho̍k-kâ."]


def test_split_is_by_article_and_deterministic():
    ids = [str(i) for i in range(100)]
    a, b = split_articles(ids, seed=0), split_articles(ids, seed=0)
    assert a == b
    assert sorted(a["train"] + a["val"] + a["test"]) == sorted(ids)
    assert not set(a["train"]) & set(a["test"]) and not set(a["train"]) & set(a["val"])
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (80, 10, 10)


def test_build_drops_duplicates_and_chinese_articles(tmp_path):
    hakka = "Kì he yit-chak fa-ho̍k-kâ. Kì chhai Mî-koet chhut-sên."
    df = pd.DataFrame({
        "id": ["1", "2", "3"],
        "title": ["A", "A copy", "Han"],
        "text": [hakka, hakka, "香港客家話係香港原居民講嘅客家話。"],
    })
    path = tmp_path / "tiny.parquet"
    df.to_parquet(path)
    by_split, stats = build(path)
    assert stats["duplicates_dropped"] == 1 and stats["han_dropped"] == 1
    all_ids = {r["article_id"] for recs in by_split.values() for r in recs}
    assert all_ids == {"1"}
