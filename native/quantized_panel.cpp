// Exact signed integer panel products. Safe up to rank 512:
// 512*127*32767 < INT32_MAX. Scales are applied by Python in FP64.
#include <cstdint>
#if defined(_WIN32)
#define QB_EXPORT __declspec(dllexport)
#else
#define QB_EXPORT __attribute__((visibility("default")))
#endif
extern "C" QB_EXPORT int qb_int8_i16_panel(int64_t rows, int32_t cols, int32_t stride,
        int32_t rank, const int8_t* a, const int16_t* b, int32_t* out) {
    if(rows<0 || cols<1 || rank<1 || rank>512 || stride<rank) return -1;
    for(int64_t i=0;i<rows;++i){
        const auto* x=a+i*stride;
        for(int32_t j=0;j<cols;++j){
            const auto* y=b+int64_t(j)*rank;
            int32_t acc=0;
            for(int32_t k=0;k<rank;++k) acc+=int32_t(x[k])*int32_t(y[k]);
            out[i*cols+j]=acc;
        }
    }
    return 0;
}
