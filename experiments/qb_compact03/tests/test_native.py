"""Native tests use an explicit prebuilt library, otherwise skip."""
import os
import numpy as np
import pytest
from qctx.learned_screen import fit_basis
from qctx.progressive_screen import ProgressiveScreen
from qctx.native_screen import NativeProgressiveScreen


@pytest.mark.parametrize('d,ranks',[(8,(1,2,4,8)),(64,(4,16,32,64)),(128,(16,32,64,128))])
def test_native(d,ranks):
    lib=os.environ.get('QB_SCAN_LIBRARY')
    if not lib:pytest.skip('set QB_SCAN_LIBRARY to locally compiled trusted kernel')
    rng=np.random.default_rng(7800+d);train=rng.normal(size=(180,d));x=rng.normal(size=(300,d));x[1]=x[0];x[2]=-x[0]
    u=fit_basis(train,ranks[-1]);ix=NativeProgressiveScreen(x,u,lib,ranks);ref=ProgressiveScreen(x,u,ranks)
    for q in [*rng.normal(size=(10,d)),x[0],x[2]]:
        for k in (1,7,20,300):
            truth=ref.search(q,k,screen=False)
            for start,end in [(ranks[0],ranks[-1]),(ranks[1],ranks[-1]),(ranks[-1],ranks[-1])]:
                for lower in (False,True):
                    got=ix.search(q,k,pilot=16,start_rank=start,end_rank=end,lower_bound=lower)
                    assert got['indices']==truth['indices']
                    assert np.allclose(got['scores'],truth['scores'],atol=1e-14)
                    assert got['projected_coordinates']==sum(s['width']*s['block_rows'] for s in got['stages'])
                    assert got['unique_evaluated']<=len(x)
