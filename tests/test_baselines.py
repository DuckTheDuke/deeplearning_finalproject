import pytest

from hakka_tones.baselines import LinearCRF, SyllableMajority, WordMajority, final_class
from hakka_tones.data import sentence_record
from hakka_tones.evaluate import evaluate, train_word_set, word_keys


def rec(text, template=False):
    return {**sentence_record(text), "template": template}


TRAIN = [rec("Kì he yit-chak fa-ho̍k-kâ."), rec("Kì he yit-chak fa-ho̍k-kâ."), rec("Ngì he Thòi-vân-ngìn.")]


def test_word_keys_skip_foreign_tokens():
    r = rec("Kì chhai Nobel Fa-ho̍k-kâ.")
    assert word_keys(r) == ["ki", "chhai", None, "fa-hok-ka", "fa-hok-ka", "fa-hok-ka"]


def test_syllable_majority_and_unknown_default():
    model = SyllableMajority().fit(TRAIN)
    assert model.predict_one("he") == "none"
    assert model.predict_one("ki") == "grave"
    assert model.predict_one("zzz") == model.default == "none"


def test_word_majority_uses_whole_word_then_backs_off():
    # "ngin" alone is grave in train; inside the seen word "thoi-van-ngin" it is also grave,
    # but the unseen word "ngin-hok" must fall back to per-syllable majority.
    model = WordMajority().fit(TRAIN)
    pred = model.predict([rec("Thòi-vân-ngìn he fa-ho̍k-kâ.")])[0]
    assert pred == ["grave", "circumflex", "grave", "none", "none", "vertical", "circumflex"]
    pred = model.predict([rec("Ngìn-ho̍k.")])[0]
    assert pred == ["grave", "vertical"]


def test_perfect_predictions_score_100():
    test = [rec("Kì he Thòi-vân-ngìn.", template=True), rec("Kì chhai Nobel Fa-ho̍k-kâ.")]
    m = evaluate(test, [r["tags"] for r in test], train_word_set(TRAIN))
    assert m["acc_all"] == m["macro_f1"] == m["sentence_exact"] == 1.0
    # X (Nobel) is not scored: 5 + 5 Hakka syllables
    assert m["n_all"] == 10 and m["n_template"] == 5
    # unseen words: "chhai" (1 syllable); "thoi-van-ngin", "fa-hok-ka", "ki", "he" are in train
    assert m["n_unseen"] == 1


def test_errors_counted_per_subset():
    test = [rec("Kì chhai fa-ho̍k-kâ.")]
    wrong = ["grave", "grave", "none", "vertical", "circumflex"]  # "chhai" wrong (unseen word)
    m = evaluate(test, [wrong], train_word_set(TRAIN))
    assert m["acc_all"] == pytest.approx(4 / 5)
    assert m["acc_unseen"] == 0.0 and m["acc_seen"] == 1.0
    assert m["sentence_exact"] == 0.0


@pytest.mark.parametrize("syl, cls", [("hok", "stop"), ("yit", "stop"), ("ngin", "nasal"),
                                      ("thung", "nasal"), ("ka", "open")])
def test_final_class(syl, cls):
    assert final_class(syl) == cls


def test_linear_crf_learns_training_data():
    model = LinearCRF(max_iterations=50).fit(TRAIN * 5)
    assert model.predict(TRAIN[:1])[0] == TRAIN[0]["tags"]
