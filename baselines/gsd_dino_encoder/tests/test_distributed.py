"""Two-rank CPU/Gloo check of global target balancing and student/EMA synchronization."""
from datetime import timedelta
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from baselines.gsd_dino_encoder.losses import sinkhorn_targets, ssl_loss
from baselines.gsd_dino_encoder.model import update_teacher
from baselines.gsd_dino_encoder.tests.test_contracts import tiny, views


def worker(rank, rendezvous, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(44)
        all_logits=torch.randn(7,5)*.1
        logits=all_logits[:4] if rank==0 else all_logits[4:]
        balanced=sinkhorn_targets(logits,.07)
        student,teacher,config=tiny()
        model=DDP(student,broadcast_buffers=False)
        torch.manual_seed(1000+rank)
        g,l,m=views()
        loss,_=ssl_loss(model(g,l,m),teacher.teacher_forward(g,m),m,config,.07)
        loss.backward()
        torch.optim.SGD(student.parameters(),lr=1e-4).step()
        update_teacher(student,teacher,.9)
        torch.save({'q':balanced, 'student':torch.cat([p.detach().flatten() for p in student.parameters()]),
                    'teacher':torch.cat([p.detach().flatten() for p in teacher.parameters()])},
                   Path(folder)/f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


class DistributedContracts(unittest.TestCase):
    def test_two_rank_targets_and_ema(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with tempfile.TemporaryDirectory() as folder:
            rendezvous=(Path(folder)/'rendezvous').resolve().as_uri()
            mp.spawn(worker,args=(rendezvous,folder),nprocs=2,join=True)
            a=torch.load(Path(folder)/'rank0.pt',weights_only=True)
            b=torch.load(Path(folder)/'rank1.pt',weights_only=True)
            torch.manual_seed(44)
            expected=sinkhorn_targets(torch.randn(7,5)*.1,.07)
            torch.testing.assert_close(torch.cat([a['q'],b['q']]),expected,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(a['student'],b['student'],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(a['teacher'],b['teacher'],atol=1e-6,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
