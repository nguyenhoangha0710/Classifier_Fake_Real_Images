"""CPU checks for the changed augmentation, crop and resumable batch contracts."""
import random
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train_codec import Views, codec, Batches
from src.core import ForensicResidualExtractor, ForensicTwins, barlow_twins_loss
from src.data import _rectangles_overlap

def main():
    image=Image.fromarray(np.random.default_rng(42).integers(0,256,(256,270,3),dtype=np.uint8))
    counts={}
    for i in range(1000):
        out,rec=codec(image,random.Random(i))
        assert out.size==image.size
        counts[rec['policy']]=counts.get(rec['policy'],0)+1
        if rec['quality'] is not None: assert 80<=rec['quality']<=100
    assert 430<counts['jpeg']<570 and 290<counts['resize_jpeg']<410 and 100<counts['identity']<200,counts
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'real.png'; image.save(path)
        row={'sample_id':'test','image_path':str(path)}
        for epoch in range(8):
            entries=[(row,epoch,7)]
            a,b,records=Views()(entries)
            aa,bb,rr=Views()(entries)
            assert torch.equal(a,aa) and torch.equal(b,bb) and records==rr
            assert not _rectangles_overlap(records[0]['box1'],records[0]['box2'])
            assert a.min()>=0 and a.max()<=1
        # Halo computation equals full-image FRE then crop, including native context.
        gray=torch.from_numpy(np.array(image.convert('L'),dtype=np.float32))[None,None]/255
        fre=ForensicResidualExtractor().freeze()
        from src.data import _extract_halo
        box=(8,16,64,64)
        local=fre(_extract_halo(gray[0],box,7)[None])[...,7:-7,7:-7]
        full=fre(gray)[...,16:80,8:72]
        assert torch.allclose(local,full,atol=1e-6)
    all_batches=list(Batches(range(128),16,2))
    assert list(Batches(range(128),16,2,3))==all_batches[3:]
    model=ForensicTwins(ForensicResidualExtractor(),projector_hidden=32,projector_out=32)
    _,_,e1,e2=model(torch.rand(2,1,78,78),torch.rand(2,1,110,110))
    loss,_=barlow_twins_loss(e1,e2); loss.backward()
    assert torch.isfinite(loss) and model.encoder.patch_embed.weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in model.fre.parameters())
    print('Codec CPU checks passed:',counts)

if __name__=='__main__': main()
