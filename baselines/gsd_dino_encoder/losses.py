"""DINO cross-view CE, masked iBOT CE and per-global-view KoLeo."""
import math

import torch
import torch.distributed as dist
from torch.nn import functional as F


def distributed():
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def sinkhorn_targets(logits, temperature, iterations=3):
    """Log-domain Sinkhorn: same row/column balancing without exp overflow."""
    if logits.ndim != 2 or temperature <= 0 or iterations < 1:
        raise ValueError('Invalid Sinkhorn inputs')
    log_q = logits.detach().float() / temperature
    count = torch.tensor(float(len(log_q)), device=log_q.device)
    if distributed():
        dist.all_reduce(count)
    if count.item() < 1:
        raise ValueError('No teacher tokens for Sinkhorn')
    log_count = math.log(count.item())
    log_k = math.log(log_q.shape[1])
    for _ in range(iterations):
        if distributed():
            maximum = log_q.max(0).values if len(log_q) else log_q.new_full((log_q.shape[1],), -float('inf'))
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            totals = (log_q - maximum).exp().sum(0)
            dist.all_reduce(totals)
            log_totals = maximum + totals.clamp_min(1e-30).log()
        else:
            log_totals = torch.logsumexp(log_q, dim=0)
        log_q = log_q - log_totals - log_k
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True) - log_count
    return (log_q + log_count).exp()


def dino_loss(student_logits, teacher_targets, batch_size, local_crops, temperature=0.1):
    b = batch_size
    if student_logits.shape[0] != (2 + local_crops) * b or teacher_targets.shape[0] != 2 * b:
        raise ValueError('DINO expects crop-major global/local logits')
    student = F.log_softmax(student_logits.float() / temperature, dim=-1).split(b)
    teacher = teacher_targets.detach().split(b)
    terms = [-(teacher[g] * student[v]).sum(-1).mean()
             for g in range(2) for v in range(2 + local_crops) if v != g]
    return torch.stack(terms).mean()


def ibot_loss(student_logits, teacher_targets, masks, temperature=0.1):
    if masks.ndim != 2 or masks.dtype != torch.bool or int(masks.sum()) != len(student_logits):
        raise ValueError('Masked logits must match row-major masked indices')
    if teacher_targets.shape != student_logits.shape:
        raise ValueError('Teacher/student patch logits differ')
    weights = masks.sum(-1).clamp_min(1).reciprocal().float()
    weights = weights[:, None].expand_as(masks)[masks]
    cross_entropy = -(teacher_targets.detach() * F.log_softmax(student_logits.float() / temperature, dim=-1)).sum(-1)
    return (cross_entropy * weights).sum() / len(masks)


def koleo_loss(cls):
    if cls.shape[0] < 2:
        raise ValueError('KoLeo needs >=2 source images per GPU, not merely global batch >=2')
    with torch.autocast(device_type=cls.device.type, enabled=False):
        u = F.normalize(cls.float(), dim=-1, eps=1e-8)
        with torch.no_grad():
            similarity = u @ u.T
            similarity.fill_diagonal_(-float('inf'))
            nearest = similarity.argmax(dim=1)
        distances = F.pairwise_distance(u, u[nearest], p=2, eps=1e-8)
        return -(distances + 1e-8).log().mean()


def ssl_loss(student, teacher, masks, config, teacher_temperature):
    b = masks.shape[0] // 2
    q_dino = sinkhorn_targets(teacher['dino'], teacher_temperature, config['sinkhorn_iterations'])
    q_ibot = sinkhorn_targets(teacher['ibot'], teacher_temperature, config['sinkhorn_iterations'])
    dino = dino_loss(student['dino'], q_dino, b, config['local_crops'], config['student_temperature'])
    ibot = ibot_loss(student['ibot'], q_ibot, masks, config['student_temperature'])
    koleo = sum(koleo_loss(x) for x in student['cls'].split(b))
    total = dino + ibot + config['koleo_weight'] * koleo
    return total, {'dino': dino.detach(), 'ibot': ibot.detach(), 'koleo': koleo.detach()}
