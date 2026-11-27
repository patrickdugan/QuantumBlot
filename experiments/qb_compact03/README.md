# QB-COMPACT-03: progressive residual screening and query rank selection

CPU-only continuation of QB-COMPACT-02. Experimental opt-in APIs. No QPU, GPU, model inference, corpus uploads, or Fable. Full cosine remains the legacy default.

## Exact-arithmetic result

For unit real vectors x,y and a nested orthonormal frame U, define a=U^T x, b=U^T y and R_r=I-U_r U_r^T. The signed prefix s_r=sum_{i<=r} a_i b_i and residual product e_r=sqrt(x^T R_r x)*sqrt(y^T R_r y) give lower/upper bounds L_r=s_r-e_r, B_r=s_r+e_r. The statement extends to a contraction U with U U^T <= I. For a new block V, the old residual energies split into the block energies and new residual energies. Cauchy-Schwarz over the two nonnegative norm components gives |<V^T x,V^T y>|+e_new <= e_old. Therefore L_r is nondecreasing, B_r nonincreasing. This proves safe permanent exclusion below a valid lower bound on the kth true score. The kth largest among distinct-row lower bounds is such a bound. Exact pilot scores are included as lower bounds for their own rows only. Strict comparison preserves equal-score candidates and deterministic row-ID ties.

The NumPy implementation scales an approximately orthonormal frame to a contraction using a computed Gram defect and engineering slack; it also preserves the minimum previously computed upper bound. These safeguards are not a directed-rounding or interval certificate. Returned scores come from the full normalized vectors. Rank ladder 16,32,64,128, early exact switch at 1.5% unresolved (at least k), pilot 64, dense block choice at 45%, full fallback at 50% survivors. The C++ variant fuses row-local prefix refinement with a heap of valid distinct-row lower bounds and a final upper-bound pass, with no full-score approximation. It loads only an explicitly built local shared library. C++ compiled without fast-math; no native binary distributed. Linux x86_64 tested; Windows/macOS build not tested. Pure NumPy path remains available.

## Matched experiments

Training 1536 normalized synthetic rows, corpus 8192, development 24, test 48, dimension512, roots 58401..58403. Dense, centroid, subspace, heterogeneous families (heterogeneous class subspace dimensions 2,2,4,4,8,16,32,64, plus shared nuisance and noise). Separate natural-data digits control: 714 train,898 corpus,90 dev,95 test, d64 with ranks8/16/32/64. Fit one maximum-rank uncentered PCA frame from train only. Fixed ranks use prior LearnedScreen. One warm-up and three timed calls, randomized arm order, single BLAS thread. Corpus-level policy selects by development median latency (requires >=5% gain vs full). Test results never change it. The table reports median of root medians, not pooled per-query means.

| family | development choice | selected speedup | early progressive speedup | early progressive coordinate work / full |
|---|---|---:|---:|---:|
| centroid | fixed32, all roots | 4.60 | 3.75 | .0535 |
| subspace | fixed32, all roots | 4.77 | 2.27 | .0868 |
| heterogeneous | fixed32 once, fixed64 twice | 3.23 | 2.38 | .0580 |
| dense | full, all roots | 1.00 | .57 | 1.2734 |
| digits | full | 1.00 | .26 | .2316 |

Coordinate work counts query-projection dot-product dimensions, corpus projection dimensions actually scored (including any dense block operation), and full-score dimensions including repeated pilot work on fallback. It excludes sorting, norm operations, memory allocation, cache behavior and other costs. It is not measured FLOPs or bytes transferred. Mixed-subspace fixed64 uses .1444 of full coordinate work; early progressive .0580, but higher elapsed time. Fixed32 yields 122.5 median full comparisons on centroid and 77 on subspace. All dense screenings fall back. For the mixed-subspace workload, full progressive had zero full-scan fallbacks and median 100 full-vector comparisons versus 95 for fixed64; early progressive had 141.5. Lower-bound threshold helped subspace cases. Extra index 8.75 MiB for 32 MiB float64 corpus, 27.34% additional stored array bytes, excluding temporary memory and Python overhead.

Native follow-up uses fresh roots 58501 for all four synthetic families and 58502 for heterogeneous. First combined 58501 native run exceeded local tool timeout; completed results were regenerated individually and only those receipts counted. All 240 held-out queries match 14 screening arms. Heterogeneous median native-progressive speedup 2.73 vs full and 2.24 for NumPy progressive; median native coordinate ratio .0655. Development-selected fixed rank still won on the measured latency metric. All five native cohorts selected full or fixed NumPy, so keep native opt-in. No general latency or memory-bandwidth claim.

## Query-energy gate

Fit one decision stump using rank16 retained query energy and actual development mean latency over full, rank16/32/64/128. Include measured gate time, at least six development rows per leaf, require 10% improvement over fastest static development mean. Fit uses only 24 development queries and no labels. If no qualifying split, use static with no feature calculation. Separate roots 58601..58603; dense and heterogeneous. On all three dense roots policy chose full. On all three heterogeneous roots choose rank16 for high retained energy, rank128 otherwise. Gate overhead included in held-out timing. Results (mean ms):

| root | static rank128 | adaptive | adaptive speedup vs static | adaptive rank16 choices /48 |
|---|---:|---:|---:|---:|
| 58601 | .6311 | .5071 | 1.245 | 27 |
| 58602 | .7018 | .6123 | 1.146 | 27 |
| 58603 | .6361 | .4936 | 1.289 | 28 |

P90 latency improved in two roots and slightly worsened in 58602. All 288 held-out queries match full scan, 144 development queries. A threshold boundary robustness correction (right observed energy endpoint instead of midpoint) was made after initial pilot; all three roots rerun with final code. Timings are descriptive, sample is small, no semantic recall or language-embedding deployment claim. Benchmark holds multiple separate indexes in RAM; a production implementation would share storage and retain only selected arms. This policy is a cost controller, not a learned similarity metric or proof of lower latency on arbitrary inputs.

## Audit

NumPy: 13 cohorts,671 test queries,4697 ranking comparisons against full. Native: 5 cohorts,240 test queries,3360 comparisons. Gate: 288 test queries,576 comparisons. Total1199 held-out queries,8633 ranking matches; 642 development queries. 54,159,856 numerical prefix lower/upper checks, no bound or monotonicity violation observed. Max observed score error6.66e-16 for fixed/progressive/native comparison. Combined isolated QB01/QB02/QB03 suite 106 passed,3 skipped (two Qiskit tests unavailable, one supervised rank-one control inapplicable). New native tests ran using local GCC shared library. Full legacy integration and CI remain unverified. No actual QuantumBlot corpus available; benchmark uses synthetic rows and sklearn digits only. Timing excludes basis fit and index build, recorded separately. All full rows are resident; no storage compression claimed.

## Reproduce

From repository root, Python3.13.5, NumPy2.3.5, sklearn and pytest. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Use Python module invocation from repo root. Native optional on Linux:

```sh
mkdir -p build
g++ -O3 -std=c++17 -fPIC -shared native/progressive_scan.cpp -o build/libqb_scan.so
export QB_SCAN_LIBRARY="$PWD/build/libqb_scan.so"
python -m pytest -q experiments/qb_compact01/tests experiments/qb_compact02/tests experiments/qb_compact03/tests
python -m experiments.qb_compact03.benchmark --root 58401
python -m experiments.qb_compact03.benchmark --root 58402
python -m experiments.qb_compact03.benchmark --root 58403
python -m experiments.qb_compact03.benchmark --digits
python -m experiments.qb_compact03.benchmark --root 58501 --native-library build/libqb_scan.so
python -m experiments.qb_compact03.benchmark --root 58502 --family heterogeneous --native-library build/libqb_scan.so
python -m experiments.qb_compact03.benchmark_policy --root 58601
python -m experiments.qb_compact03.benchmark_policy --root 58602
python -m experiments.qb_compact03.benchmark_policy --root 58603
```

For local actual embedding data (four genuinely disjoint matrices, same embedding model and preprocessing; paths alone cannot prove disjointness):

```sh
python -m experiments.qb_compact03.benchmark --train train.npy --corpus corpus.npy --dev dev.npy --test queries.npy --ranks 16 32 64 128 --out local_audit.json
```

Optional native mode expects at least two ranks for its second-rank-start comparison. The benchmark stores multiple baseline indexes for timing; start with a bounded corpus on RAM-limited machines. Receipts include per-query work/latency and hashed top-k row IDs, so keep actual corpus receipts local if sensitive. No network requests or model downloads occur. Public repository has summary and executable code; full generated synthetic receipts are in the reproduction bundle. Re-run timing on target hardware before choosing a retrieval path. Next: use real local language embeddings and measure batched/blocked query schedules; keep the gate frozen on a distinct development split. All code in this stage is experimental and opt-in.
