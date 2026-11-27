# QB-COMPACT-01 — exact observable reduction and selected-bin arithmetic

Base 57ca3e26b2fb225923f7c36262c6de35f5642a41. Transfer of the HAD-QFT preparation-equivalent principle and valuation-shell arithmetic to QuantumBlot. CPU-only; no hardware job, no new embedding model inference, no cryptographic assurance or real corpus benchmark.

## Bifocal audit

The pinned BifocalQFT initializes every register in a uniform superposition; its two QFTs with swaps map the coarse and fine states to zero. The intervening codec phases and CNOT gates fix zero. Shadow 0 (temporal only) flips all coarse bits; shadow 1 flips the lowest min(n_fine,4) fine bits. Shadow 2 CZ (genealogical) fixes the zero coarse register; shadow 3 controlled phases (conceptual) act diagonally and do not change computational-basis data probabilities. Trace out shadows and the measured law is the XOR of one or two independent fair-bit masks. No corpus vector, timestamp, topic ID, abstraction level, parent ID, or theme enters the legacy state preparation. This is a distribution equivalence for that circuit; no claim about a generic QFT or other observables.

Default 8+8+4: 20 wires, 56 controlled phases and eight swaps within the QFTs. Temporal also uses 12 shadow CNOTs. With ideal two-CNOT controlled-phase and three-CNOT swap decomposition (before routing), temporal baseline is 148 CNOTs, compact circuit 16 measured wires, two H and 10 CNOTs. One can sample the exact law on a CPU without a circuit. Other codecs use one H and three CNOTs in compact mode. 20 independent statevector reconstructions across four codecs and five width pairs, including the full 20-wire default; measured laws agree to 1e-12. Original Qiskit path remains default. Optional direct Qiskit comparison skipped locally.

## Shell-energy identity

For N=2^b, normalized Fourier transform and unit-normalized x, cumulative mass through shell j (frequencies of order dividing 2^j) is E_j=(2^j/N) sum_r |sum_(t congruent r mod 2^j) x_t|^2. Apply Parseval to the folded vector. If z has length 2^j, split z=u||v. E_j-E_(j-1)=2^(j-1-b)||u-v||^2/||x||^2 in the original unnormalized convention. The fold uses u+v at each level, hence O(N) arithmetic, log2(N)+1 energies. Code scales input for finite dynamic range, accumulates nonnegative squared norms. This is exact in exact arithmetic for arbitrary real/complex x, distinct from contiguous bands and from projecting amplitudes onto the invariant subspace. Counts can be directly aggregated using additive order, with caller-specified bit reversal.

For invariant amplitudes expressed in orthonormal shell basis, p=c0, write output coordinate b+1-j=(p-c_j)/sqrt(2), update p=(p+c_j)/sqrt(2), DC=p after j=b. This is the weighted Ramanujan transform in O(b) operations on native shell coordinates. Projection of an arbitrary x is explicitly exposed separately with squared relative residual epsilon. Normalized projected state, if nonzero, has measurement TV <=sqrt(epsilon) by the pure-state trace-distance formula. No hidden projection is used for energy fingerprints. The information in energy alone is insufficient to reconstruct an arbitrary embedding or authenticate it.

## Coarse-to-exact retrieval

Parseval and Cauchy-Schwarz on each frequency shell yield |<x,y>| <= sum_j sqrt(p_j q_j). Sort candidates by this upper bound, evaluate exact normalized-vector scores in batches, stop only if the remaining bound lies below the kth score (strict inequality for ties). Exact-arithmetic guarantee; engineering float64 margin, not interval-certified floating point. Stable row index ties, full-scan option. Normalized vectors remain in RAM; the descriptor adds O(M log N) storage. No automatic replacement of existing FFT band or mixer semantics.

64 queries, 4096 synthetic 1024-dimensional vectors, top20: 64/64 matches to full BLAS. Dense: 4096/4096 exact evaluations, 3.668ms vs 1.706ms. Single-shell synthetic: median448 (384..448), .876ms vs 1.711ms, 89.06% fewer exact evaluations. These data were deliberately structured; no real-corpus semantic recall claim. Generic query invariant projection loss median .9891; single-shell query .9753. Batched fold agrees with grouped real FFT up to 5.78e-15 through N65536, but slower for every measured size. Keep optional until profiling actual corpus and optimized native implementation.

## Selected-bin watermark

The prior semantic_proof.py modifies FFT bins f and -f by s and conjugate(s), then returns real(IFFT). By linearity direct result is real(x_t)+(2/T)real(s exp(2pi i f t/T)), including self-conjugate DC/Nyquist bins when both updates hit the same bin. Verification uses a single unnormalized DFT sum instead of FFT of all bins. Same mathematical signature, phase, thresholds, and existing crypto-attestation implementation. Changes roundoff and potentially a threshold-edge decision; no cryptographic security implication. For T65536, K16 real trace, 15 timing repetitions, injection 42.089->5.830ms (7.22x), selected bin 22.955->5.501ms (4.17x); max injection error 2.23e-15, max bin abs error 3.73e-12. PCA and I/O excluded. Short T128 verification slower. The local suite checks the new arithmetic against prior formulas, not end-to-end integration in the complete cloned repository.

## Reproduce

From repo root, NumPy 2.3.5 and pytest (Python3.13.5 tested):
- python -m pytest -q experiments/qb_compact01/tests
- python experiments/qb_compact01/benchmark.py
- python experiments/qb_compact01/watermark_benchmark.py

Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before benchmarking. 76 local tests passed, 2 skipped (Qiskit absent). Public source blob hashes of the new helpers, test files and benchmark scripts match the executed local files. Integrated legacy class edits are patch-reviewed, existing runner regression suite and Qiskit comparison remain to run in full checkout. Legacy qctx package imported a missing fourier_fingerprint export eagerly; new lazy exports enable the NumPy-only path, but missing legacy function and pre-existing invalid import in raw_codec.py remain out of scope. CI unverified. No new QPU time or Fable.

Next: profile actual 1024-dimensional corpus with matched retrieval recall, exploit the exact shell filter only when it screens; compare block/streamed index memory and measure whether meaningful Fourier profiles require learning a fixed representation. Do not claim uniform-state bifocal sampling performs semantic retrieval.
