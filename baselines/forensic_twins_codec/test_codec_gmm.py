"""CPU integration checks for reused Gaussian math and codec-matched test inputs."""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from evaluate_codec_gmm import g,b,ProtocolDataset

def main():
    rng=np.random.default_rng(42)
    fit=rng.normal(size=(500,8)); cal=rng.normal(size=(1000,8))
    model=g.fit_single_gaussian(fit,1e-6)
    scores=g.gaussian_nll(cal,model)
    centered=cal-model['mean']
    expected=.5*(8*np.log(2*np.pi)+np.linalg.slogdet(model['covariance'])[1]+np.einsum('ni,ij,nj->n',centered,np.linalg.inv(model['covariance']),centered))
    assert np.allclose(scores,expected)
    t=g.calibrate_thresholds(scores,[90,95,99],95)
    assert abs(np.mean(scores>t['percentiles']['95'])-.05)<.002
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'image.png'; Image.fromarray(rng.integers(0,256,(300,310,3),dtype=np.uint8)).save(path)
        row={'sample_id':'fake','image_path':str(path),'label':1,'generator':'test','jpeg_reencode':True,'jpeg_quality':80,'jpeg_subsampling':2}
        transformed=ProtocolDataset([row],b.ProtocolSpec('fake_codec_matched'),256)[0][0]
        expected=g.ProtocolImageDataset([row],256)[0]['gray_halo']
        native=ProtocolDataset([dict(row,jpeg_reencode=False)],b.ProtocolSpec('native'),256)[0][0]
        assert torch.equal(transformed,expected) and not torch.equal(native,transformed)
        dest=Path(temp)/'model.npz'; g.save_gmm(model,dest,{'source':'test'})
        assert np.allclose(g.gaussian_nll(cal,g.load_gmm(dest)),scores)
    print('Codec GMM CPU integration checks passed')
if __name__=='__main__': main()
