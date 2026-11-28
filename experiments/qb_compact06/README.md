# QB-COMPACT-06: compressed RAM index and exact disk-backed retrieval

CPU-only opt-in extension of QB-COMPACT-05 at 6f5bde112894c494f4da6350c592276ef4cd0083. Full normalized vectors are streamed into a flat little-endian binary64 backing file; full-vector int8 or int16 codes, per-row binary32 scales and outward error radii are loaded in memory. Optional learned projection uses int16 by default, with its residual and error radii. No persistent original vector matrix or corpus-by-query score panel in the search object. Original high-precision vectors remain on disk. The build accepts an npy memmap and optional source row IDs; returned IDs are store row numbers, with optional `source_rows` mapping. Read-only trusted store convention, manifest written last, file length/SHA256 checks on open by default, file-stat mutation check around search. This is accidental-corruption detection, not authenticated storage or safe concurrent modification. Opening with `verify=False` skips hashes. Search on one instance is serial; optional internal positional reads share one descriptor safely.

## Bounds and algorithm

For unit vectors x,y and a contraction U, c=U^T x and p=U^T y satisfy |x^T y-c^T p| <= sqrt(1-||c||^2)sqrt(1-||p||^2). For reconstructed packed vectors h(c),h(p) with Euclidean errors e_c,e_p, |c^T p-h(c)^T h(p)| <= e_c ||p||+(||c||+e_c)e_p. Apply the QB05 binary32 dot-product gamma bound to the scaled integer panel, using actual query norms and a corpus norm upper estimate. The full-vector code has no projection residual term. Intersect the resulting two enclosures. First pass across bounded tiles maintains the kth largest lower bound across distinct rows; every true top-k score is at least this threshold. Cascade uses a provisional threshold from previous rows plus current coarse lower bounds to refine only unions of possible survivors across queries. Second pass recomputes bounds, loads only rows whose upper bound can reach the threshold, and updates it with exact kth scores. Strict exclusion preserves ties. Every output score uses the original stored normalized binary64 row, with canonical per-row summation near the GEMM top-k boundary and stable store-row ties. Algebraic guarantees are conditional on valid enclosures, standard round-to-nearest FP32 accumulation (no reduced precision or overflow), and numerical guard assumptions. FP64 Gram, normalization, radius, and final threshold safeguards remain engineering margins rather than interval-certified arithmetic.

Read runs are sorted/coalesced within each tile and query block, then exact-score one gathered union matrix; gap_rows may read extra rows to reduce requests. Optional 1..8 positional-read workers (requires os.pread, Linux tested), at most twice as many pending runs as workers. No automatic hardware use. Threaded variant does not change requested row order or scores. Individually bounded score panels O(tile_rows*query_block); temporary row buffers O(tile_rows*d). This is not a measured peak RSS limit. The OS may cache original pages. `mmap` baseline maps original rows and uses the same tiled GEMM and canonical scores, `full` baseline uses explicit sequential reads, `projection` and `code` ablate the cascade.

## Main matched benchmark

Three fresh seeds 58911 centroid, 58912 mixed, 58913 dense. 16,384 corpus rows of dimension512, 1536 independent training, 8 development, 24 held-out queries per seed. Learned rank32 for centroid/dense and64 for mixed; int8 full codes and int16 projection, 4096-row tiles, batch1/8/24. Six arms, randomized order, two warm trials after touching backing file, one best-effort POSIX_FADV_DONTNEED trial per arm/setting. Warm dev choice requires 5% gain against mmap. No concurrency across cohorts. Pilot seed58901 excluded. Linux container on virtualized overlay filesystem, NumPy2.3.5, one BLAS thread. Read volumes: logical requested row bytes plus mmap-scored bytes; page visits include repeated visits. /proc/self/io read_bytes deltas are kernel-accounted storage traffic; not physical SSD bytes/latency. Cold advice is not guaranteed eviction, and subsequent query blocks in one timed call reuse pages. No actual hardware SSD identification. Timing excludes build/hash verification, includes decoding, read calls, allocation, canonical scoring. Raw receipts preserve individual trials and counters.

| family | resident arrays MiB | cascade requested bytes / full mmap, batch1 | batch8 | batch24 | cold-advised kernel read ratio, batch24 | cold batch24 speedup |
|---|---:|---:|---:|---:|---:|---:|
| centroid |9.4375|.199%|1.589%|4.724%|5.50%|.172x|
| mixed |10.5625|.187%|1.493%|4.413%|5.18%|.185x|
| dense |9.4375|.332%|2.627%|7.617%|9.73%|.122x|

Each backing file is 64MiB, plus 9.44 or 10.56MiB of packed metadata on disk. Persistent application arrays shrink by 85.25% (rank32) or 83.50% (rank64) vs retaining 64MiB original vectors. Includes basis; no claim of equivalent peak RSS reduction or net on-disk compression. Warm centroid batch1 chose cascade_gap2: 54.871 vs 78.826ms for 24 total queries (1.44x); all other eight dev-selected main settings chose mmap. Cold batch24 mmap read_bytes 64MiB in all three; cascade 3,690,496 / 3,477,504 / 6,529,024 bytes. Cold cascade slower despite lower traffic. Rank32 projection alone screens poorly on dense; full-code interval yields useful I/O reduction even there. Cascade computes 2.08%,17.74%,100% of the two-pass full-code panel products in centroid/mixed/dense batch1 respectively. Compare code-only on dense. The storage benefit is meaningful, but current implementation pays for irregular reads and FP32 decode/refinement.

## Read scheduling follow-up

Fresh seeds 58914 centroid and 58915 mixed, same sizes/splits, cold advice only. One or four workers and gap_rows 0/16/64, against mmap. Development selection of six arm/setting groups picked four cascade paths that regressed on held-out latency; remaining two picked mmap. One post-hoc centroid b24 setting (gap64, one worker) used 39 read calls, 70.1ms vs 122.1ms mmap in that particular trial, but requested much more data; not a reliable speedup and not the frozen choice. Parallel reads/gap merging are optional only; next stage should reorder backing pages using learned cluster locality with original stable ID mapping and freeze cost controller on a larger development set. Main default sequential receipts were collected before adding optional parallel reads and source-map verification; final sequential behavior and parity tested.

## Audit

Five independent cohorts, 120 held-out queries, 40 development queries; 3,024 held-out arm-versus-mmap ranking checks (2,592 screening comparisons and 432 explicit sequential full-scan comparisons), 648 development checks, exact agreement in all. Max observed score discrepancy zero. Nine new tests pass: int8/int16, dense/structured, extreme values, zero projections, duplicate/negative scores, tiled scores against direct canonical sum, shape/size/hash failure, read accounting, source mapping, parallel parity. Combined isolated QB04/QB05/QB06 tests 31 passed with native QB05 kernel loaded. Full legacy CI, Windows execution and formal floating-point certification outstanding. Actual QuantumBlot language-embedding binary could not be retrieved here; no real-corpus performance claim. No model, GPU or QPU calls.

## Reproduce

From repository root, Python3.13.5, NumPy2.3.5, pytest. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Use a local workdir on the intended backing device, with enough free disk space. All generated stores are temporary in the benchmark.

```sh
python -m pytest -q experiments/qb_compact06/tests
python -m experiments.qb_compact06.benchmark --family centroid --seed 58911 --out centroid_58911.json
python -m experiments.qb_compact06.benchmark --family mixed --seed 58912 --rank 64 --out mixed_58912.json
python -m experiments.qb_compact06.benchmark --family dense --seed 58913 --out dense_58913.json
python -m experiments.qb_compact06.io_followup --family centroid --seed 58914 --out io_58914.json
python -m experiments.qb_compact06.io_followup --family mixed --seed 58915 --rank 64 --out io_58915.json
python -m experiments.qb_compact06.benchmark --split-file qctx/vectors_pca_topk.npy --rank 32 --tile 4096 --workdir /path/to/local/ssd --out local_audit.json
```

On Windows use a local SSD directory and sequential mode; POSIX cold advice, Linux counters and pread follow-up are unavailable. Disjoint row splits do not prove independence of documents. Keep actual corpus and receipts private. For the API use `build_store(new_directory, source_memmap, basis)` then `with DiskScreen(new_directory) as index: index.search(queries, k=20, mode='cascade')`; use `mode='mmap'` as latency/control fallback. Legacy defaults unchanged.
