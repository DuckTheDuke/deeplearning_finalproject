"""Correctness harness for the from-scratch CRF in hakka_tones.crf.

Three independent kinds of evidence:
  brute force   enumerate every tag sequence on tiny inputs (the definition of the CRF)
  library       same numbers as pytorch-crf (torchcrf)
  gradients     autograd agrees with finite differences and with the textbook formulas

Everything runs in float64 so tolerances can be tight.
"""
import itertools

import pytest
import torch
from torchcrf import CRF as LibraryCRF

from hakka_tones.crf import CRF

DTYPE = torch.float64
TOL = 1e-8


def make_crf(num_tags, seed=0):
    torch.manual_seed(seed)
    crf = CRF(num_tags).to(DTYPE)
    with torch.no_grad():  # larger than the default init so mistakes in any term show up
        crf.transitions.copy_(torch.randn(num_tags, num_tags, dtype=DTYPE))
        crf.start.copy_(torch.randn(num_tags, dtype=DTYPE))
        crf.end.copy_(torch.randn(num_tags, dtype=DTYPE))
    return crf


def make_batch(batch, max_len, num_tags, seed=0, lengths=None):
    g = torch.Generator().manual_seed(seed)
    emissions = torch.randn(batch, max_len, num_tags, dtype=DTYPE, generator=g)
    tags = torch.randint(num_tags, (batch, max_len), generator=g)
    if lengths is None:
        lengths = torch.randint(1, max_len + 1, (batch,), generator=g)
        lengths[0] = max_len
    lengths = torch.as_tensor(lengths)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    return emissions, tags, mask, lengths.tolist()


def path_score(crf, emissions, seq):
    """Score of one tag sequence for one sentence, straight from the definition (no batching, no mask)."""
    score = crf.start[seq[0]] + emissions[0, seq[0]]
    for t in range(1, len(seq)):
        score = score + crf.transitions[seq[t - 1], seq[t]] + emissions[t, seq[t]]
    return score + crf.end[seq[-1]]


def all_paths(crf, emissions, length):
    seqs = list(itertools.product(range(crf.num_tags), repeat=length))
    return seqs, torch.stack([path_score(crf, emissions, s) for s in seqs])


def library_copy(crf):
    lib = LibraryCRF(crf.num_tags, batch_first=True).to(DTYPE)
    with torch.no_grad():
        lib.transitions.copy_(crf.transitions)
        lib.start_transitions.copy_(crf.start)
        lib.end_transitions.copy_(crf.end)
    return lib


# ---------- score_sequence ----------

def test_score_by_hand():
    crf = CRF(2).to(DTYPE)
    with torch.no_grad():
        crf.start.copy_(torch.tensor([1.0, 2.0]))
        crf.end.copy_(torch.tensor([10.0, 20.0]))
        crf.transitions.copy_(torch.tensor([[100.0, 200.0], [300.0, 400.0]]))
    emissions = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], dtype=DTYPE)
    tags = torch.tensor([[0, 1, 1]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    # start[0] + E[0,0] + trans[0,1] + E[1,1] + trans[1,1] + E[2,1] + end[1]
    expected = 1.0 + 0.1 + 200.0 + 0.4 + 400.0 + 0.6 + 20.0
    assert crf.score_sequence(emissions, tags, mask).item() == pytest.approx(expected, abs=TOL)


def test_score_matches_definition_with_padding():
    crf = make_crf(4)
    emissions, tags, mask, lengths = make_batch(6, 7, 4, seed=1)
    got = crf.score_sequence(emissions, tags, mask)
    assert got.shape == (6,)
    for b, n in enumerate(lengths):
        expected = path_score(crf, emissions[b], tags[b, :n].tolist())
        assert got[b].item() == pytest.approx(expected.item(), abs=TOL), f"sentence {b}, length {n}"


# ---------- log_partition (forward algorithm) ----------

def test_log_partition_matches_brute_force():
    crf = make_crf(3)
    emissions, _, mask, lengths = make_batch(5, 4, 3, seed=2, lengths=[4, 3, 1, 2, 4])  # up to 3^4 = 81 paths
    got = crf.log_partition(emissions, mask)
    assert got.shape == (5,)
    for b, n in enumerate(lengths):
        _, scores = all_paths(crf, emissions[b], n)
        assert got[b].item() == pytest.approx(torch.logsumexp(scores, 0).item(), abs=TOL), f"length {n}"


def test_probabilities_sum_to_one():
    crf = make_crf(3)
    emissions, _, _, _ = make_batch(1, 4, 3, seed=3, lengths=[4])
    seqs = torch.tensor(list(itertools.product(range(3), repeat=4)))
    nll = crf.neg_log_likelihood(emissions.expand(len(seqs), -1, -1), seqs, torch.ones(len(seqs), 4, dtype=torch.bool))
    assert (nll >= 0).all()
    assert torch.exp(-nll).sum().item() == pytest.approx(1.0, abs=TOL)


def test_long_sentence_is_numerically_stable():
    crf = make_crf(6)
    emissions, tags, mask, _ = make_batch(2, 300, 6, seed=4, lengths=[300, 300])
    nll = crf.neg_log_likelihood(emissions * 50, tags, mask)  # would overflow without log-space
    assert torch.isfinite(nll).all() and (nll >= 0).all()


# ---------- agreement with pytorch-crf ----------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nll_matches_library(seed):
    crf = make_crf(6, seed)
    emissions, tags, mask, _ = make_batch(8, 12, 6, seed=seed)
    expected = -library_copy(crf)(emissions, tags, mask=mask, reduction="none")
    torch.testing.assert_close(crf.neg_log_likelihood(emissions, tags, mask), expected, atol=1e-6, rtol=0)
    assert crf(emissions, tags, mask).item() == pytest.approx(expected.mean().item(), abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_viterbi_matches_library(seed):
    crf = make_crf(6, seed)
    emissions, _, mask, _ = make_batch(8, 12, 6, seed=seed)
    assert crf.viterbi_decode(emissions, mask) == library_copy(crf).decode(emissions, mask=mask)


# ---------- viterbi ----------

def test_viterbi_matches_brute_force():
    crf = make_crf(3)
    emissions, _, mask, lengths = make_batch(5, 4, 3, seed=5, lengths=[4, 3, 1, 2, 4])
    paths = crf.viterbi_decode(emissions, mask)
    assert [len(p) for p in paths] == lengths
    for b, n in enumerate(lengths):
        seqs, scores = all_paths(crf, emissions[b], n)
        assert tuple(paths[b]) == seqs[scores.argmax().item()], f"sentence {b}, length {n}"


def test_viterbi_path_never_scores_below_gold():
    crf = make_crf(5)
    emissions, tags, mask, lengths = make_batch(6, 9, 5, seed=6)
    best = torch.zeros_like(tags)
    for b, path in enumerate(crf.viterbi_decode(emissions, mask)):
        best[b, :len(path)] = torch.tensor(path)
    assert (crf.score_sequence(emissions, best, mask) >= crf.score_sequence(emissions, tags, mask) - TOL).all()


# ---------- masking and batching ----------

def test_padding_content_is_ignored():
    crf = make_crf(4)
    emissions, tags, mask, _ = make_batch(5, 8, 4, seed=7, lengths=[8, 5, 1, 3, 6])
    noisy_e = torch.where(mask[..., None], emissions, torch.full_like(emissions, 1e3))
    noisy_t = torch.where(mask, tags, torch.full_like(tags, 3))
    torch.testing.assert_close(crf.neg_log_likelihood(noisy_e, noisy_t, mask),
                               crf.neg_log_likelihood(emissions, tags, mask), atol=TOL, rtol=0)
    assert crf.viterbi_decode(noisy_e, mask) == crf.viterbi_decode(emissions, mask)


def test_batch_equals_one_sentence_at_a_time():
    crf = make_crf(4)
    emissions, tags, mask, lengths = make_batch(5, 8, 4, seed=8)
    batched = crf.neg_log_likelihood(emissions, tags, mask)
    paths = crf.viterbi_decode(emissions, mask)
    for b, n in enumerate(lengths):
        e, t, m = emissions[b:b + 1, :n], tags[b:b + 1, :n], mask[b:b + 1, :n]
        assert crf.neg_log_likelihood(e, t, m).item() == pytest.approx(batched[b].item(), abs=TOL)
        assert crf.viterbi_decode(e, m) == [paths[b]]


# ---------- gradients ----------

def test_gradcheck():
    crf = make_crf(3)
    emissions, tags, mask, _ = make_batch(3, 5, 3, seed=9, lengths=[5, 2, 4])

    def loss(e, transitions, start, end):
        params = {"transitions": transitions, "start": start, "end": end}
        return torch.func.functional_call(crf, params, (e, tags, mask))

    inputs = [emissions.clone().requires_grad_(), *(p.detach().clone().requires_grad_()
                                                    for p in (crf.transitions, crf.start, crf.end))]
    assert torch.autograd.gradcheck(loss, inputs, eps=1e-6, atol=1e-5)


def test_emission_gradient_is_marginal_minus_gold():
    """dNLL/dE[t, j] = P(y_t = j | x) - 1[gold y_t = j]: the 'belief minus truth' form to derive for the quiz."""
    crf = make_crf(3)
    emissions, tags, mask, _ = make_batch(1, 4, 3, seed=10, lengths=[4])
    emissions.requires_grad_()
    crf.neg_log_likelihood(emissions, tags, mask).sum().backward()

    seqs, scores = all_paths(crf, emissions[0].detach(), 4)
    probs = torch.softmax(scores, 0)
    expected = torch.zeros(4, 3, dtype=DTYPE)
    for seq, p in zip(seqs, probs):
        for t, j in enumerate(seq):
            expected[t, j] += p
    for t, j in enumerate(tags[0].tolist()):
        expected[t, j] -= 1
    torch.testing.assert_close(emissions.grad[0], expected.detach(), atol=TOL, rtol=0)


def test_transition_gradient_is_expected_minus_observed_counts():
    """dNLL/dA[i, j] = E[count of i->j under the model] - count of i->j in the gold sequence."""
    crf = make_crf(3)
    emissions, tags, mask, _ = make_batch(1, 4, 3, seed=11, lengths=[4])
    crf.neg_log_likelihood(emissions, tags, mask).sum().backward()

    seqs, scores = all_paths(crf, emissions[0], 4)
    probs = torch.softmax(scores.detach(), 0)
    expected = torch.zeros(3, 3, dtype=DTYPE)
    for seq, p in zip(seqs, probs):
        for i, j in zip(seq, seq[1:]):
            expected[i, j] += p
    gold = tags[0].tolist()
    for i, j in zip(gold, gold[1:]):
        expected[i, j] -= 1
    torch.testing.assert_close(crf.transitions.grad, expected, atol=TOL, rtol=0)
