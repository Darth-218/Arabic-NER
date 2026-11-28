# AraBERT — Hyperparameter Search (issue 5.3)

**Owner:** Member 3 · **Selection split:** validation F1 (test is untouched until issue 5.4)

`batch_size` is not searched: 8 is already the largest the 4 GB RTX 3050 holds at `max_len 128` with gradient checkpointing, and accumulation already yields an effective batch of 16. Every other setting (bf16, gradient checkpointing, warmup, weight decay, seed) is identical to the baseline so the comparison isolates the tuned knobs.

| Stage | Run | LR | Epochs | Max len | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `lr2e5_e3_bs8_len128` | 2e-05 | 3 | 128 | 0.9056 | 0.9137 | **0.9097** |
| 1 | `lr3e5_e3_bs8_len128` | 3e-05 | 3 | 128 | 0.9074 | 0.9162 | **0.9118** |
| 1 | `lr5e5_e3_bs8_len128` | 5e-05 | 3 | 128 | 0.9097 | 0.9192 | **0.9144** |
| 2 | `lr5e5_e4_bs8_len128` | 5e-05 | 4 | 128 | 0.9162 | 0.9233 | **0.9197** |
| 3 | `lr5e5_e4_bs8_len96` | 5e-05 | 4 | 96 | 0.9130 | 0.9230 | **0.9180** |

**Best:** `lr5e5_e4_bs8_len128` with validation F1 **0.9197** (precision 0.9162, recall 0.9233).

Baseline for reference: lr 3e-5 / epochs 3 / max_len 128, validation F1 0.9129 (issue 5.1 full run).

Which run wins is measured on **validation**; the number that goes into the phase-1 report is the **test** score from a single retrain with the winning config (issue 5.4).
