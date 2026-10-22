"""Two-process CPU/Gloo check of shared basis and Barlow/DDP gradient scaling."""
import os
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from model import semantic_basis, barlow_loss


def worker(rank, url):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=url,rank=rank,world_size=2)
    try:
        torch.manual_seed(31)
        x=torch.randn(8,5);y=torch.randn(8,5);patches=torch.randn(8,4,5)
        model=torch.nn.Linear(5,4)
        reference=torch.nn.Linear(5,4);reference.load_state_dict(model.state_dict())
        v,_=semantic_basis(patches[rank*4:(rank+1)*4],2,'batch')
        gathered=[torch.empty_like(v),torch.empty_like(v)];dist.all_gather(gathered,v)
        torch.testing.assert_close(gathered[0],gathered[1])
        _,_,vh=torch.linalg.svd(patches.reshape(-1,5),full_matrices=False)
        torch.testing.assert_close(v@v.T,vh[:2].T@vh[:2])
        model=DistributedDataParallel(model)
        e1=model(x[rank*4:(rank+1)*4]);e2=model(y[rank*4:(rank+1)*4])
        loss,_=barlow_loss(e1,e2);loss.backward()
        # Reference computed after collectives close, so it uses the full batch once.
        dist.barrier();dist.destroy_process_group()
        full,_=barlow_loss(reference(x),reference(y));full.backward()
        torch.testing.assert_close(loss,full)
        for p,q in zip(model.module.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


class DistributedContracts(unittest.TestCase):
    def test_two_ranks(self):
        with tempfile.TemporaryDirectory(prefix='gsd-ddp-') as tmp:
            url=Path(tmp,'rendezvous').resolve().as_uri()
            mp.spawn(worker,args=(url,),nprocs=2,join=True)


if __name__=='__main__': unittest.main()
