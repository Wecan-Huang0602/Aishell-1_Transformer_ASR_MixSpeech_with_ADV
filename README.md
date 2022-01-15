# Aishell-1_Transformer_ASR_MixSpeech_with_ADV

## Baseline Methods
| Baseline                    | Dev CER | Test CER |
| :-----:                     | :----:  | :----:   |
| Base                        | 6.34    | 7.07     |
| SpecAugment                 | 5.43    | 5.90     |
| Speed Perturb               | 5.95    | 6.63     |
| Speed Perturb + SpecAugment | 5.18    | 5.74     |
---

## MixSpeech Methods
| MixSpeech (Accumulation)    | Dev CER | Test CER |
| :-----:                     | :----:  | :----:   |
| Max 𝜆=0.3                   | 6.19    | 6.88     |
| Max 𝜆=0.5                   | 5.74    | 6.37     |
---

## MixSpeech with ADV Methods
| MixSpeech with ADV                                        | Dev CER | Test CER |
| :-----:                                                   | :----:  | :----:   |
| Accumulation (Max 𝜆=0.5)                                  | 5.81    | 6.37     |
| 1:Clean -> 2: MixSpeech -> MixSpeech with ADV (Max 𝜆=0.5) | 5.63    | 6.15     |
---

