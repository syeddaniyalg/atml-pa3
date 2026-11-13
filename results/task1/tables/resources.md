| method | trainable params (M) | train time (h) | peak GPU mem (GB) | weights/adapter size (MB) | recovery ckpt (GB) | skipped updates | overhead |
|---|---|---|---|---|---|---|---|
| Full FT | 596.05 | 2.37 | 11.69 | 2273.9 | 6.66 | 0 | ref 2.22 GB (CPU) |
| Full FT + L2-SP | 596.05 | 2.533 | 11.69 | 2273.9 | 6.66 | 0 | ref 2.22 GB (CPU); penalty 6.0 min |
| LoRA-4 | 2.523 | 2.629 | 3.14 | 9.7 | 0.03 | 0 |  |
| LoRA-16 | 10.093 | 2.658 | 3.25 | 38.6 | 0.11 | 0 |  |
| OPLoRA-16 | 10.093 | 3.448 | 3.55 | 38.6 | 0.11 | 0 | bases 308 MB, svd 33s, export 0.1s |