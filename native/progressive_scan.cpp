// Fused per-row screening kernel. No model, hardware or external dependencies.
// Caller validates shapes, finite data, rank ordering and contraction bounds.
// The heap contains lower bounds for distinct rows and exact seed scores.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#if defined(_WIN32)
#define QB_EXPORT __declspec(dllexport)
#else
#define QB_EXPORT __attribute__((visibility("default")))
#endif

// Independent accumulators permit vectorization without fast-math.
static double dot(const double* a, const double* b, int lo, int hi) {
    double s0=0,s1=0,s2=0,s3=0;
    int j=lo;
    for (; j+3<hi; j+=4) {
        s0+=a[j]*b[j];s1+=a[j+1]*b[j+1];s2+=a[j+2]*b[j+2];s3+=a[j+3]*b[j+3];
    }
    double s=(s0+s1)+(s2+s3);
    for (;j<hi;++j) s+=a[j]*b[j];
    return s;
}

extern "C" QB_EXPORT int64_t qb_scan_v1(
    int64_t m, int32_t stride, int32_t levels, int32_t start, int32_t end,
    const int32_t* ranks, const double* coeff, const double* residual,
    const double* p, const double* query_res, const double* initial,
    const uint8_t* seed_mask, int32_t k, const double* seed_lower,
    double slack, int32_t update_lower, int64_t* output, double* upper_out,
    int64_t* visited, double* final_threshold) {
    std::vector<double> heap(seed_lower, seed_lower+k);
    std::make_heap(heap.begin(), heap.end(), std::greater<double>());
    double threshold=heap.front();
    int64_t kept=0;
    std::fill(visited,visited+levels,0);
    for (int64_t i=0;i<m;++i) {
        if(seed_mask[i]) continue;
        double s=initial[i];
        double ub=1e300,lb=-1e300;
        bool reject=false;
        const double* row=coeff+i*stride;
        for(int32_t j=start;j<=end;++j){
            ++visited[j];
            if(j!=start) s+=dot(row,p,ranks[j-1],ranks[j]);
            double e=residual[i*levels+j]*query_res[j];
            ub=std::min(ub,s+e+slack);
            lb=std::max(lb,s-e-slack);
            if(ub<threshold-slack){reject=true;break;}
        }
        if(reject)continue;
        output[kept]=i;upper_out[kept]=ub;++kept;
        // One lower bound per newly visited row. No repeated heap entries for
        // a row, even though several prefixes may have been examined.
        if(update_lower && lb>threshold){
            std::pop_heap(heap.begin(),heap.end(),std::greater<double>());
            heap.back()=lb;
            std::push_heap(heap.begin(),heap.end(),std::greater<double>());
            threshold=heap.front();
        }
    }
    // Earlier rows can now be rejected by the tightened global threshold.
    int64_t n=0;
    for(int64_t i=0;i<kept;++i)if(upper_out[i]>=threshold-slack)output[n++]=output[i];
    *final_threshold=threshold;
    return n;
}
