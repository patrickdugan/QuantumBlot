"""Local CPU selected-bin comparison with the previous full-FFT formula."""
import sys, numpy as np,time,statistics,json
from pathlib import Path
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p.parents[1]/'newtype/qft-runner'));from spectral_bin import inject_conjugate_pair, selected_bin


def run():
    r=np.random.default_rng(732);result=[]
    def med(f):
        f();v=[]
        for _ in range(15):
            t=time.perf_counter_ns();f();v.append((time.perf_counter_ns()-t)/1e6)
        return statistics.median(v)
    for T in [128,1024,8192,65536]:
        x=r.normal(size=(T,16));s=.08*np.exp(1j*r.uniform(0,2*np.pi,16));f=3
        def old():
            a=np.fft.fft(x,axis=0);a[f]+=s;a[-f%T]+=s.conj();return np.fft.ifft(a,axis=0).real
        def old_verify():return np.fft.fft(x,axis=0)[f]
        a=old();b=inject_conjugate_pair(x,f,s);assert np.max(abs(a-b))<1e-12
        c=old_verify();d=selected_bin(x,f);assert np.max(abs(c-d))<1e-8
        result.append(dict(T=T,K=16,old_embed_ms=med(old),new_embed_ms=med(lambda:inject_conjugate_pair(x,f,s)),old_verify_ms=med(old_verify),new_verify_ms=med(lambda:selected_bin(x,f)),max_embed_error=float(np.max(abs(a-b))),max_bin_error=float(np.max(abs(c-d)))))
    (p/'WATERMARK_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
