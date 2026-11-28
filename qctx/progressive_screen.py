"""Nested signed-subspace screening for real cosine retrieval.

Uses an opt-in index; original rows remain available for exact reranking.
Bounds and monotonicity hold in exact arithmetic. Float64 slack is an
engineering safeguard, not an interval-certified implementation.
"""
from __future__ import annotations
import numpy as np
from .learned_screen import normalize, rank_top


class ProgressiveScreen:
    """A fixed nested frame and contiguous blocks, learned on separate data.

    A scaled orthonormal frame is a contraction. Every prefix satisfies the
    residual Cauchy-Schwarz bound, and every added block tightens it in exact
    arithmetic. Screening uses exact-score seeds and optional lower bounds.
    """
    def __init__(self, vectors, basis, ranks=(16, 32, 64, 128)):
        self.x = normalize(vectors)
        if self.x.ndim != 2 or not len(self.x):
            raise ValueError('expected nonempty corpus matrix')
        self.m, self.d = self.x.shape
        self.ids = np.arange(self.m)
        self.ranks = tuple(ranks)
        if (not self.ranks or any(not isinstance(r, (int, np.integer)) for r in self.ranks)
                or not 0 < self.ranks[0] or self.ranks[-1] > self.d
                or any(a >= b for a, b in zip(self.ranks, self.ranks[1:]))):
            raise ValueError('ranks must be strictly increasing positive integers <= dimension')
        u = np.asarray(basis)
        if np.iscomplexobj(u):
            raise ValueError('real frame required')
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.d, self.ranks[-1]) or not np.all(np.isfinite(u)):
            raise ValueError('frame shape does not match corpus and final rank')
        self.slack = 256*np.finfo(np.float64).eps*max(self.d, 1)
        defect = float(np.linalg.norm(u.T@u-np.eye(u.shape[1]), ord=2))
        if defect > 1e-8:
            raise ValueError('frame must be orthonormal')
        u = u/np.sqrt(1+defect+self.slack)
        self.defect = defect
        self.basis_blocks = []
        self.blocks = []
        self.residuals = []
        mass = np.zeros(self.m)
        start = 0
        for r in self.ranks:
            v = np.ascontiguousarray(u[:, start:r])
            c = np.ascontiguousarray(self.x @ v)
            mass += np.einsum('ij,ij->i', c, c)
            self.basis_blocks.append(v)
            self.blocks.append(c)
            self.residuals.append(np.sqrt(np.maximum(0., 1-mass+self.slack)))
            start = r

    @property
    def index_bytes(self):
        return sum(x.nbytes for xs in (self.basis_blocks, self.blocks, self.residuals) for x in xs)

    def prefix_bounds(self, query):
        """Audit path: all lower/upper bounds, no survivor filtering."""
        q = normalize(query)
        if q.shape != (self.d,):
            raise ValueError('query shape mismatch')
        p = np.zeros(self.m)
        mass = 0.
        out = []
        for u, c, r in zip(self.basis_blocks, self.blocks, self.residuals):
            z = q@u
            mass += float(z@z)
            p += c@z
            e = r*np.sqrt(max(0., 1-mass+self.slack))
            out.append((p-e-self.slack, p+e+self.slack))
        return out

    def search(self, query, k=20, *, pilot=64, early_fraction=0.,
               fallback_fraction=.5, lower_bound=True, dense_fraction=.45,
               screen=True):
        """Progressively score candidate blocks, then evaluate original rows.

        `early_fraction` is a cost heuristic: stop refinements once at most
        max(k,ceil(fraction*M)) unresolved rows remain. Zero disables the
        heuristic (empty active sets still stop). A dense block operation may
        be cheaper than gathering most rows; work telemetry counts it in full.
        `lower_bound` strengthens the threshold using all current lower bounds
        and the exact top-k seeds. Strict exclusion and stable tie breaks.
        """
        q = normalize(query)
        if (q.shape != (self.d,) or not isinstance(k, int) or not 1 <= k <= self.m
                or not isinstance(pilot, int) or pilot < 1 or not 0 <= early_fraction <= 1
                or not 0 < fallback_fraction <= 1 or not 0 < dense_fraction <= 1):
            raise ValueError('invalid query parameters')
        if not screen:
            ids, scores = rank_top(self.x@q, self.ids, k)
            return dict(indices=ids.tolist(), scores=scores.tolist(), evaluated=self.m,
                        unique_evaluated=self.m, fallback=False, stages=[],
                        projected_coordinates=0, query_coordinates=0,
                        coordinate_work=self.m*self.d, stop_reason='full')
        active = self.ids.copy()
        partial = np.zeros(self.m)
        prior = np.full(self.m, np.inf)
        mass = 0.
        seed_ids = None
        seed_scores = None
        threshold = -np.inf
        trace = []
        work = 0
        query_work = 0
        stop = 'final_rank'
        nseed = min(self.m, max(k, pilot))
        for level, (rank, u, block, residual) in enumerate(zip(
                self.ranks, self.basis_blocks, self.blocks, self.residuals)):
            if not len(active):
                stop = 'empty'; break
            w = u.shape[1]
            z = q@u
            query_work += self.d*w
            mass += float(z@z)
            dense = len(active) >= dense_fraction*self.m
            if dense:
                delta = (block@z)[active]
                evaluated_block = self.m
            else:
                delta = block[active]@z
                evaluated_block = len(active)
            work += evaluated_block*w
            partial += delta
            e = residual[active]*np.sqrt(max(0., 1-mass+self.slack))
            upper = np.minimum(prior, partial+e+self.slack)
            lower = partial-e-self.slack
            before = len(active)
            if level == 0:
                # Only the exact seeds can serve as direct result candidates.
                seed_ids, _ = rank_top(partial, active, nseed)
                seed_scores = self.x[seed_ids]@q
                _, best = rank_top(seed_scores, seed_ids, k)
                threshold = float(best[-1])
                mask = np.ones(self.m, dtype=bool); mask[seed_ids] = False
                active, partial, upper, lower = active[mask], partial[mask], upper[mask], lower[mask]
            if lower_bound and len(active):
                # These are distinct original candidates. A kth lower bound
                # is a valid lower bound on the full corpus's kth true score.
                all_lower = np.concatenate((seed_scores, lower))
                threshold = max(threshold, float(np.partition(all_lower,len(all_lower)-k)[len(all_lower)-k]))
            mask = upper >= threshold-self.slack
            active, partial, prior = active[mask], partial[mask], upper[mask]
            trace.append(dict(rank=rank, width=w, active_before=before,
                              active_after=len(active), block_rows=evaluated_block,
                              dense=bool(dense), threshold=threshold,
                              query_retained_energy=mass))
            if not len(active):
                stop='empty';break
            if early_fraction and len(active) <= max(k, int(np.ceil(early_fraction*self.m))):
                stop='early_exact';break
        fallback = nseed+len(active) >= fallback_fraction*self.m
        if fallback:
            ids,scores=rank_top(self.x@q,self.ids,k)
            evaluated=self.m+nseed; unique=self.m; stop='full_fallback'
        else:
            ids,scores=rank_top(np.concatenate((seed_scores,self.x[active]@q)),
                                np.concatenate((seed_ids,active)),k)
            evaluated=unique=nseed+len(active)
        return dict(indices=ids.tolist(), scores=scores.tolist(), evaluated=int(evaluated),
                    unique_evaluated=int(unique), fallback=bool(fallback), stages=trace,
                    projected_coordinates=int(work), query_coordinates=int(query_work),
                    coordinate_work=int(work+query_work+evaluated*self.d),stop_reason=stop)
