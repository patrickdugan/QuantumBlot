# QB-COMPACT-06-R: materialized-bound reference recheck

The original QB06 landed at fa7de62a88649076b88d99d402b8981d5f3e34c3. Its local receipts were unavailable when the continuation began, and its GitHub commit was discovered after this recheck ran. Preserve the original production API and its bounded-tile, int16 projection, source-row mapping and optional parallel-read implementation. This directory contains a separate experimental reference and raw-receipt hashes, with no changes to qctx/disk_screen.py. The same synthetic corpus seeds 58911–58913 are reused, so this is an implementation recheck rather than an independent-corpus replication.

## Reference construction

Stream normalized binary64 backing rows into a new store with SHA-256 manifest and verify all file lengths/digests on open. Ordinary backing-file changes are checked around search. Trusted local immutable-store convention, not adversarial authentication, concurrent-write safety or crash durability. Full int8 vector codes and int8 rank-32 projections both store binary32 row scales and outward-rounded reconstruction errors. Projection adds an orthogonal residual bound. For reconstructed coefficients c_hat,p_hat with error norms e_x,e_y, dot-product error is bounded in exact arithmetic by e_x||y||+(||x||+e_x)e_y. Add gamma_(2r+8) for conventional FP32 GEMM, with engineering FP64 slack. This remains conditional on valid radii, ordinary accumulation and numerical margins; not a certified floating-point proof.

Unlike the main implementation, materialize one M-by-query-block upper-bound array. Maintain the kth largest lower bound across all distinct corpus rows, discard only upper endpoints below it, optionally refine projected candidates by full codes, then coalesce backing reads by sorted row ID. Final contenders receive a canonical float64 reduction on stored original rows, stable store-row ties. The same reduction is used for the full mmap control. Score equality below refers to this convention. Default mmap; explicit read/codes/projection/cascade modes. No threaded reader in this reference. The main implementation retains the preferable bounded tile memory behavior.

M=16,384, d=512, r=32: explicit persistent index arrays 9,371,648 bytes (8.9375 MiB) plus 64 MiB original backing file. Arrays occupy 13.96% of original corpus bytes, a reduction of 86.04% in resident application-array storage. Query block 24 adds 3 MiB for upper endpoints; tile panels, temporary buffers, Python overhead and OS cache are additional. No peak RSS claim. Opening with verification reads the full backing file.

## Same-seed recheck

Three synthetic cohorts, 1,536 independent training rows, 16,384 corpus rows and 24 held-out queries each. Rank 32 for every cohort (original QB06 used rank 64 for mixed), int8 projections, tiles 1,024, batch sizes 1/8/24, warm and best-effort cold advice, two randomized timing repetitions per arm. Six arms: mmap, full explicit read, projection, codes, cascade, cascade gap=2. One BLAS thread. Cold advice via POSIX_FADV_DONTNEED before whole call; later query blocks may reuse pages. Linux virtualized overlay filesystem; physical device unknown. Kernel /proc/self/io read_bytes includes readahead and is not physical SSD telemetry. Fit/build/open and cache preparation excluded from search time.

| family | codes requested bytes vs mmap b1 | b24 | cold kernel read ratio b24 | warm b1 mmap / codes ms | cold b24 mmap / codes ms |
|---|---:|---:|---:|---:|---:|
| centroid | .1966% | 4.6753% | 5.4321% | 94.44 / 65.68 | 37.09 / 183.78 |
| mixed | .1844% | 4.3457% | 5.0964% | 91.20 / 65.76 | 37.57 / 166.61 |
| dense | .3204% | 7.3608% | 9.1675% | 90.60 / 76.61 | 42.28 / 266.05 |

All times are for 24 total test queries. Codes and cascade requested identical backing bytes at every measured setting. Full-vector codes also screen dense vectors, where rank-32 projection alone reads the full corpus. Cascade generally adds latency; gap=2 not consistently faster. Cold batch24 codes performs 693–1,116 small read calls. I/O volume improvement does not imply elapsed-time improvement. Results agree qualitatively with the main QB06 campaign, with differences in layout, projection storage/rank, tile, algorithm and repetitions. Do not pool the same-seed cohorts or treat timing differences as controlled head-to-head speedups.

72 distinct queries; 5,184 timed query/arm/cache/batch/repetition checks including 3,456 screening checks and 1,728 full-read controls. All rankings identical; maximum score discrepancy zero. 4,718,592 bound endpoint comparisons with no observed violation; minimum observed endpoint margin .004687877. Eight new tests; combined isolated QB04/QB05/reference suite 30 passed, native QB05 panel loaded. Main QB06 suite is separate (its committed report records 31 combined passes). Legacy CI and Windows unverified. No actual language embedding corpus available here. No model, GPU or QPU calls. All raw fresh recheck receipts in QB_COMPACT06_reference_recheck.zip; source blobs and receipt hashes in AUDIT_SUMMARY.json.

## Reproduce

From repo root, Python 3.13.5, NumPy 2.3.5 and pytest 9.0.2. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.

```sh
python -m pytest -q experiments/qb_compact06/recheck/tests
python -m experiments.qb_compact06.recheck.benchmark --family centroid --seed 58911 --out centroid_58911.json
python -m experiments.qb_compact06.recheck.benchmark --family mixed --seed 58912 --out mixed_58912.json
python -m experiments.qb_compact06.recheck.benchmark --family dense --seed 58913 --out dense_58913.json
python -m experiments.qb_compact06.recheck.benchmark --split-file qctx/vectors_pca_topk.npy --rank 32 --workdir /path/to/local/ssd --out local_audit.json
```

Use --warm-only on hosts without Linux cold advice. The local split is disjoint by row ID, not proven independent documents. Keep real corpus receipts private. Next: physically cluster backing rows, preserve original stable logical IDs, measure block read amplification and latency on target device. Main API and defaults unchanged.
