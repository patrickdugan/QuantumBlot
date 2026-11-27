# QB-COMPACT-05: quantized, error-bounded projection panels

CPU-only extension of QB04. Full normalized float64 vectors remain for exact-score reranking; new projection formats are opt-in.

For a contraction U and unit vectors x,y, let c=U^T x, p=U^T y, R_x=sqrt(1-||c||^2), R_y=sqrt(1-||p||^2). In exact arithmetic, |x^T y-c^T p| <= R_x R_y. If c_hat,p_hat are quantized reconstructions and ||c-c_hat|| <= e_x, ||p-p_hat|| <= e_y, then |c^T p-c_hat^T p_hat| <= e_x ||p|| + ||c_hat|| e_y <= e_x ||y||+(||x||+e_x)e_y. Add this to the residual product. For binary32 panel arithmetic add gamma_(2r+8)*(||x||+e_x)*(||y||+e_y), gamma_n=n*u/(1-n*u), u=2^-24, assuming ordinary rounding, no reduced-precision GEMM or overflow. Store per-prefix residual and reconstruction-error radii rounded upward to FP32. The FP64 contraction scaling, norm/projection operations and threshold comparisons retain conservative engineering slack; this is not a machine-certified floating-point enclosure. The algebraic bound holds conditional on valid enclosed quantities.

FP32, int16, int8 storage; integer formats use one FP32 scale per row. FP32 query projections and temporary integer unpacking included in timing. Optional C++ kernel uses int8 document codes and int16 query codes; rank <=512, symmetric code ranges give absolute accumulator <=512*127*32767=2,130,641,408 < INT32_MAX, exact integer products and accumulation. Two final FP64 scale multiplications have their own gamma term. Native compilation tested on Linux/GCC, other OS builds untested. Load trusted local libraries only; no binary distributed.

## Matched synthetic benchmark

Five fresh roots 58801..58805, dimension512, 1536 train,16 dev,32 test each. Centroid roots 58801/58804/58805 have 16384 corpus rows, rank32, one full tile; mixed58802 rank64 and 16384 rows, dense58803 rank32 and8192 rows, both tile4096. Nested rank ladder16/32/64/128. Full, auto, lower modes, batch1/8/32, one warmup two timed repetitions, randomized arm order, one BLAS thread. Native included only in roots58804/58805. Initial non-native cohorts may overlap on shared host; timings descriptive. Development selection needs >5% improvement over full; compressed-only comparator excludes native, all-arm comparator includes it.

| format | additional index MiB | centroid median auto speedup vs full (batch1/8/32) | median exact-score pair ratio |
|---|---:|---|---:|
| float64 |17|3.672/1.634/1.180|.012056|
| float32 |9|3.202/1.449/.979|.012060|
| int16 |5.0625|2.894/1.423/1.003|.012068|
| int8 |3.0625|2.874/1.552/.965|.014193|
| native int8 |3.0625|3.260/1.427/.723|.014136|

Root-level medians; native has two roots, others three. Includes basis and metadata, excludes 64 MiB original corpus, temporary allocations and overhead. Int8 saves 82% of extra index bytes and about 17% of total corpus-plus-index bytes. Int16 saves 70% of extra bytes with almost unchanged selectivity. Quantized kernels show no consistent latency advantage over float64 screening. Full batched BLAS selected on mixed b8/b32 and all dense conditions. One centroid dev selection regressed on held-out b32. No actual language-embedding performance claim.

160 held-out queries; 80 development queries; all4224 screening ranking comparisons match; 32,505,856 numeric endpoint comparisons against FP64 scores/projected products without observed violation; maximum score discrepancy9.992e-16. 22 combined QB04/QB05 tests passed (native panel parity against int64, all compressed formats, zero projections, duplicate rows, negative similarities, tiles). Full legacy integration and CI unverified. No GPU, QPU, Fable or model calls. Full raw synthetic receipts available in reproduction archive QB_COMPACT05_quantized.zip (SHA256 6d38c7ab6c7cf8d31c35ace8a7a1b7e9980a2ac89b8f353f62d6a5591560d053).

## Reproduce from repo root

Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. NumPy2.3.5 and pytest. Optional native Linux build:

```sh
mkdir -p build
g++ -O3 -march=native -std=c++17 -fPIC -shared native/quantized_panel.cpp -o build/libqb_int8.so
export QB_INT8_LIBRARY="$PWD/build/libqb_int8.so"
python -m pytest -q experiments/qb_compact04/tests experiments/qb_compact05/tests
python -m experiments.qb_compact05.benchmark --family centroid --seed 58801 --out centroid_58801.json
python -m experiments.qb_compact05.benchmark --family mixed --seed 58802 --rank 64 --tile 4096 --out mixed_58802.json
python -m experiments.qb_compact05.benchmark --family dense --seed 58803 --corpus-size 8192 --tile 4096 --out dense_58803.json
python -m experiments.qb_compact05.benchmark --family centroid --seed 58804 --native-library build/libqb_int8.so --out centroid_58804_native.json
python -m experiments.qb_compact05.benchmark --family centroid --seed 58805 --native-library build/libqb_int8.so --out centroid_58805_native.json
```

Local real-corpus binary unavailable here; run on user's machine with rank compatible with dimension:
```sh
python -m experiments.qb_compact05.benchmark --split-file qctx/vectors_pca_topk.npy --ranks 16 32 64 128 --rank 32 --tile 4096 --out local_audit.json
```
Disjoint row IDs do not establish independence of documents. Benchmark holds several indexes in RAM; reduce corpus size on constrained hosts. Keep local corpus receipts private. Next investigate quantizing the full corpus with a high-precision backing store and exact reranking of survivors, with actual disk I/O measurement. Legacy full-search default unchanged.
