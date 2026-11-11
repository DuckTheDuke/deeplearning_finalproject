# Data card: Hakka Wikipedia tone tagging

Source: `wikimedia/wikipedia`, config `20231101.hak` (Hugging Face).
Built by `python -m hakka_tones.data` with seed 0; split by article 80/10/10.

## Article filtering

- articles in dump: 10,246
- exact-duplicate texts dropped: 171
- mostly Chinese characters (>50% Han) dropped: 463
- articles with at least one usable sentence: 9,430

## Splits

| split | articles | sentences | Hakka syllables | X tokens | template sentences |
|---|---|---|---|---|---|
| train | 7,544 | 18,443 | 316,354 | 18,222 | 15.3% |
| val | 943 | 2,243 | 38,824 | 2,146 | 15.2% |
| test | 943 | 2,363 | 40,299 | 2,221 | 13.9% |

## Tone tags (Hakka syllables only)

| tag | train | val | test |
|---|---|---|---|
| circumflex | 23.6% | 24.1% | 23.9% |
| grave | 18.4% | 18.3% | 17.9% |
| acute | 11.3% | 11.3% | 11.3% |
| vertical | 5.5% | 5.6% | 5.4% |
| none | 41.2% | 40.6% | 41.6% |

## Seen vs. unseen words (test set)

- test words (hyphen-joined): 23,954; not seen in train: 1,602 (6.7%)
- test syllables inside unseen words: 3,774 (9.4%)
