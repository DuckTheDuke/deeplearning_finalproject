# Baselines

Non-deep baselines from `python -m hakka_tones.baselines`. Accuracy is over Hakka syllables only (foreign X tokens are not scored). "Unseen words" are hyphen-joined words that never appear in train.

Test set: 40,299 syllables, 3,774 in unseen words, 36,918 in non-template sentences.

## Test

| model | all | seen words | unseen words | non-template | macro-F1 | sentence exact |
|---|---|---|---|---|---|---|
| syllable_majority | 78.9% | 80.2% | 65.9% | 78.0% | 77.8% | 17.7% |
| word_majority | 92.1% | 94.8% | 65.9% | 91.4% | 91.9% | 53.1% |
| linear_crf | 93.4% | 95.7% | 71.4% | 92.8% | 93.3% | 58.9% |

## Validation

| model | all | seen words | unseen words | non-template | macro-F1 | sentence exact |
|---|---|---|---|---|---|---|
| syllable_majority | 79.5% | 80.5% | 69.7% | 78.6% | 78.7% | 18.0% |
| word_majority | 92.4% | 94.7% | 69.7% | 91.6% | 92.3% | 53.7% |
| linear_crf | 93.7% | 95.6% | 74.8% | 93.1% | 93.6% | 60.2% |

## Per-tone F1 (test)

| model | circumflex | grave | acute | vertical | none |
|---|---|---|---|---|---|
| syllable_majority | 0.747 | 0.770 | 0.607 | 0.917 | 0.849 |
| word_majority | 0.901 | 0.909 | 0.877 | 0.964 | 0.943 |
| linear_crf | 0.921 | 0.920 | 0.904 | 0.967 | 0.952 |
