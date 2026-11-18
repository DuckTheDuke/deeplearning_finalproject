"""Linear-chain CRF layer, written from the equations (the project's new mechanism).

Shapes used everywhere:
  emissions  (B, T, K) float   E[b, t, j] = score of tag j at position t (from the BiLSTM)
  tags       (B, T)    long    gold tag ids
  mask       (B, T)    bool    True for real tokens; padding is on the right, and mask[:, 0] is always True

Parameters:
  transitions[i, j]  score of tag i followed by tag j
  start[j]           score of starting a sentence with tag j
  end[j]             score of ending a sentence with tag j

Check your work with: uv run pytest tests/test_crf.py
"""
import torch
from torch import nn


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))

    def score_sequence(self, emissions, tags, mask):
        """Score of one given tag sequence per sentence -> (B,).

        s(x, y) = start[y_1] + sum_t E[t, y_t] + sum_{t>=2} transitions[y_{t-1}, y_t] + end[y_L]
        where L is the sentence's real length. Padded positions add nothing.
        """
        score = self.start[tags[:, 0]]
        emit = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        score = score + torch.where(mask,emit, 0.0).sum(dim=1)

        trans = self.transitions[tags[:, :-1], tags[:,1:]]
        score = score + torch.where(mask[:, 1:], trans, 0.0).sum(dim=1)

        lengths = mask.sum(dim=1)                                         # (B,)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1) # (B,)
        score = score + self.end[last_tags]

        return score 

    def log_partition(self, emissions, mask):
        """log Z: log of the summed exp-scores of ALL K^L tag sequences -> (B,). The forward algorithm.

        alpha_1(j) = start[j] + E[1, j]
        alpha_t(j) = E[t, j] + logsumexp_i( alpha_{t-1}(i) + transitions[i, j] )
        log Z      = logsumexp_j( alpha_L(j) + end[j] )
        At padded positions, keep the previous alpha (torch.where on the mask).
        """
        B, T, K = emissions.shape
        alpha = self.start + emissions[:, 0]                              # (B, K)

        for t in range(1, T):
            # [b, i, j] = alpha_{t-1}(i) + transitions[i, j]
            scores = alpha.unsqueeze(2) + self.transitions                # (B, K, K)
            # alpha_t(j) = E[t, j] + logsumexp over previous tag i (dim 1)
            new_alpha = torch.logsumexp(scores, dim=1) + emissions[:, t]  # (B, K)
            # padded position: keep the previous alpha
            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)

            # log Z = logsumexp_j( alpha_L(j) + end[j] )
        return torch.logsumexp(alpha + self.end, dim=1)
        

    def neg_log_likelihood(self, emissions, tags, mask):
        """-log P(y | x) = log Z - s(x, y), per sentence -> (B,). Always >= 0."""
        return self.log_partition(emissions, mask) - self.score_sequence(emissions, tags, mask)

    def forward(self, emissions, tags, mask):
        """Training loss: mean negative log-likelihood over the batch (a scalar)."""
        return self.neg_log_likelihood(emissions, tags, mask).mean()

    @torch.no_grad()
    def viterbi_decode(self, emissions, mask):
        """Highest-scoring tag sequence per sentence -> list of B lists of ints (each of its real length).

        Same recursion as log_partition with max in place of logsumexp:
        delta_t(j) = E[t, j] + max_i( delta_{t-1}(i) + transitions[i, j] ),  backpointer_t(j) = argmax_i(...)
        Then pick argmax_j( delta_L(j) + end[j] ) and follow the backpointers from L back to 1.
        """
        B, T, K = emissions.shape
        delta = self.start + emissions[:, 0]                              # (B, K)
        backpointers = []                                                 # T-1 tensors of (B, K)

        for t in range(1, T):
            scores = delta.unsqueeze(2) + self.transitions                # (B, K, K)
            best_scores, best_prev = scores.max(dim=1)                    # max over previous tag i
            new_delta = best_scores + emissions[:, t]
            delta = torch.where(mask[:, t].unsqueeze(1), new_delta, delta)
            backpointers.append(best_prev)

        best_last = (delta + self.end).argmax(dim=1)                      # (B,)
        lengths = mask.sum(dim=1)

        paths = []
        for b in range(B):
            tag = best_last[b].item()
            path = [tag]
            for t in range(lengths[b].item() - 1, 0, -1):                 # positions L-1 down to 1
                tag = backpointers[t - 1][b, tag].item()                  # backpointers[t-1] belongs to position t
                path.append(tag)
            paths.append(path[::-1])
        return paths
