# ORIEN Neural Ecosystem: Dataset Split Report

## Global Summary
| Split | Vision (IDs) | Emotion (Samples) | Behavior (Sessions) | Voice (Samples) |
|---|---|---|---|---|
| **Train** | 4025 | 27473 | 571 | 1020 |
| **Val** | 862 | 3293 | 122 | 240 |
| **Test** | 863 | 3268 | 123 | 185 |

## Emotion Class Distribution (FER2013)
### Train Emotion Set
| Class | Count | Percentage |
|---|---|---|
| Angry | 3840 | 13.98% |
| Disgust | 381 | 1.39% |
| Fear | 3897 | 14.18% |
| Happy | 7086 | 25.79% |
| Sad | 4724 | 17.20% |
| Surprise | 2675 | 9.74% |
| Neutral | 4870 | 17.73% |

### Val Emotion Set
| Class | Count | Percentage |
|---|---|---|
| Angry | 435 | 13.21% |
| Disgust | 41 | 1.25% |
| Fear | 445 | 13.51% |
| Happy | 865 | 26.27% |
| Sad | 625 | 18.98% |
| Surprise | 296 | 8.99% |
| Neutral | 586 | 17.80% |

### Test Emotion Set
| Class | Count | Percentage |
|---|---|---|
| Angry | 452 | 13.83% |
| Disgust | 38 | 1.16% |
| Fear | 478 | 14.63% |
| Happy | 845 | 25.86% |
| Sad | 573 | 17.53% |
| Surprise | 286 | 8.75% |
| Neutral | 596 | 18.24% |

## Leakage Audit Status
- Vision: Zero identity overlap (Group-based split enforced).
- Emotion: Deduplicated at source; zero sample leakage.
- Behavior: Zero session overlap (Stratified session split).
- Voice: Zero speaker overlap (Actor-based split enforced).

## Real-World Simulation Constraints
- Test Set Isolation: The /test directory is strictly isolated and should only be accessed during final evaluation.
- Temporal Split: Behavioral data preserves session integrity.
- Identity-Unseen: Vision model will be tested on 863 identities it has never seen during training.
