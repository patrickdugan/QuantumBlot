"""Batch retrieval pilot; held-out synthetic or local embeddings, no network."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
from qctx.learned_screen import fit_basis,normalize
from qctx.batch_screen import BatchScreen


def synthetic(seed,mode,m=16384,d=512):
    rng=np.random.default_rng(seed); widths=[4]*8 if mode!='mixed' else [2,2,4,4,8,16,32,64]
    off=np.cumsum([0]+widths);r=int(off[-1]);u=np.linalg.qr(rng.normal(size=(d,r+8)),mode='reduced')[0]
    def sample(n):
        labels=rng.integers(8,size=n);noise=normalize(rng.normal(size=(n,d)))
        if mode=='dense':return noise,labels
        a=np.zeros((n,r))
        for g,w in enumerate(widths):
            sel=np.flatnonzero(labels==g);z=rng.normal(size=(len(sel),w))
            if mode=='centroid' or (mode=='mixed' and g<4):z*=.22;z[:,0]+=1
            a[sel,off[g]:off[g+1]]=normalize(z)
        return normalize(.80*a@u[:,:r].T+.50*normalize(rng.normal(size=(n,8)))@u[:,r:].T+.25*noise),labels
    tr,_=sample(1536);x,_=sample(m);dev,dl=sample(32);test,tl=sample(64)
    return tr,x,dev,test,dl,tl

