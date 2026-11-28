# QB-COMPACT-02: implicit learned rotation and signed-subspace screening

Parent QB-COMPACT-01, CPU only. Exact cosine search in real arithmetic with an optional low-rank screening index. No QPU, model inference, or change to legacy retrieval defaults.

For normalized real x,y and an orthonormal d-by-r frame U, p=U^T x and q=U^T y. Decompose the dot product into p^T q plus the dot product of orthogonal residuals. Therefore x^T y <= p^T q + sqrt(1-||p||^2)sqrt(1-||q||^2). This also holds for any contraction U by Cauchy-Schwarz in the positive semidefinite form I-UU^T. Grouping p and q into dyadic blocks replaces their signed dot product with a sum of norm products and weakens the bound. For r a power of two, extend U to an orthonormal basis, place its columns in dyadic blocks 1,1,2,4,...,r/2, then use the inverse normalized periodic sum/difference fold from QB-COMPACT-01; this constructs a full rotation realizing those Fourier shell energies. The complement can remain one merged block. The implemented signed screen needs only the r coordinates and residual norm, not a full rotation. This is an elementary projection bound, not a new general QFT algorithm.

Training: uncentered PCA of normalized training rows, random orthogonal controls, and supervised class-vs-rest second moment contrast eigenvectors with equal class quotas and polar orthogonalization. No test query enters training. Screen computes cheap bounds, evaluates 64 or k seed candidates, rejects candidates strictly below the exact kth pilot score with numerical slack, then scores remaining survivors. Contiguous full-scan fallback if at least half the corpus survives. Fallback includes repeated pilot work. Float64 safety margins are engineering safeguards rather than a directed-rounding proof. Exact score ties use row index; original vectors retained in memory. User-facing API is opt-in.

Held-out synthetic experiments: roots 58301,58302,58303, each dense Gaussian, centroid and class-subspace families, 1536 train,8192 corpus,48 queries, dimension512 rank32. Structured distributions have eight latent four-dimensional class subspaces plus an eight-dimensional shared nuisance subspace and isotropic noise; disjoint samples share the planted latent basis. Natural-data control: sklearn bundled handwritten digits with disjoint stratified train/corpus/query partitions (714/898/185), dimension64 rank16. Baseline is full single-thread BLAS plus identical deterministic ranking. All six screens match all 617 held-out top20s, 3702 ranking matches, 22230444 checked bounds with no violation, max score error2.22e-16. Median of run medians (not pooled):
| Dataset | PCA signed exact unique scores | PCA signed speedup | Class signed unique scores | Class signed speedup |
|---|---:|---:|---:|---:|
| centroid |119.5|5.067|965|1.717|
| subspace |88|5.194|212.5|4.047|
| dense |8192|.840|8192|.862|
| digits |65|.469|72|.468|

PCA signed beats the supervised frame on these controls. All dense queries trigger full scan fallback; those calls also incur 64 pilot scores. Digits prunes but loses on runtime. No semantic recall or language embedding speedup has been established. Timing uses one warm-up then three complete timed queries per arm; median of query medians, randomized arm order, one BLAS thread. Training/index creation excluded from query latency and separately recorded in local full receipts. Extra index memory O(Mr+dr), original O(Md) vectors retained. No claim of data compression or GPU advantage. 16 new tests passed, one inapplicable rank-one supervised case skipped; combined isolated QB01/QB02 suite 92 passed, 3 skipped (two unavailable Qiskit checks plus the rank-one case). Full legacy integration still unverified.

## Reproduce

From repo root with NumPy, pytest, sklearn for bundled digits, set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. On PowerShell use $env:PYTHONPATH='.' or invoke module from root with equivalent Python path.

```
python -m pytest -q experiments/qb_compact01/tests experiments/qb_compact02/tests
python -m experiments.qb_compact02.benchmark --root 58301
python -m experiments.qb_compact02.benchmark --root 58302
python -m experiments.qb_compact02.benchmark --root 58303
python -m experiments.qb_compact02.benchmark --digits
python -m experiments.qb_compact02.benchmark_local --train train.npy --corpus corpus.npy --queries queries.npy --rank 32 --k 20
```

Use three genuinely disjoint local embedding arrays, same embedding model and dimension. Local script emits aggregate metrics only, validates each ranking against full scan, reports fit/build/query latency. Do not enable the optional path without target-workload measurements. Full corpus was unavailable in this environment; no data download or new credentials used. Raw timings in local reproducibility package; deterministic input seeds and aggregate summary committed. Next: test actual semantic embeddings, then calibrate whether to use screening on a separate development query split and investigate progressive residual blocks.
