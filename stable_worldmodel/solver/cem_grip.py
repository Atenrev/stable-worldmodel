"""CEM with a discrete gripper: the gripper command of every action block is sampled as one of two values
(Bernoulli per block, probability refit from the elites), the other dimensions as in CEMSolver."""

import time

import torch

from .cem import CEMSolver


class GripperBernoulliCEMSolver(CEMSolver):
    """Args (in addition to CEMSolver's):
        grip_lo, grip_hi: the two gripper values in the planner's (z-scored) action space (open, close).
        grip_dim: index of the gripper within one env-step action (LIBERO OSC: 6).
        env_action_dim: env-step action size (7); the block holds action_block of them.
        p_min: clamp of the refit probability (keeps some exploration).
    """

    def __init__(self, *args, grip_lo: float = -0.7234, grip_hi: float = 1.4037, grip_dim: int = 6,
                 env_action_dim: int = 7, p_min: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.grip_lo, self.grip_hi = float(grip_lo), float(grip_hi)
        self.grip_dim, self.env_action_dim, self.p_min = grip_dim, env_action_dim, p_min

    def _gdims(self):
        return list(range(self.grip_dim, self.action_dim, self.env_action_dim))

    def _apply(self, cand, bits):
        """Set the gripper dims of every block from bits (..., H) in {0, 1}."""
        val = self.grip_lo + (self.grip_hi - self.grip_lo) * bits.to(cand.dtype)
        cand[..., self._gdims()] = val.unsqueeze(-1).expand(*val.shape, len(self._gdims()))
        return cand

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {'costs': [], 'mean': [], 'var': []}
        total_envs = len(next(iter(info_dict.values())))
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean, var = mean.to(self.device), var.to(self.device)
        p = torch.full((total_envs, self.horizon), 0.5, device=self.device, dtype=self.dtype)
        if init_action is not None and init_action.shape[1] > 0:  # warm start: mode of the previous plan
            gd = self._gdims()
            prev = init_action.to(self.device)[..., gd].mean(-1)
            k = init_action.shape[1]
            p[:, :k] = torch.where(prev > 0.5 * (self.grip_lo + self.grip_hi), 1 - self.p_min, self.p_min)
        for s0 in range(0, total_envs, self.batch_size):
            s1 = min(s0 + self.batch_size, total_envs)
            bs = s1 - s0
            bm, bv, bp = mean[s0:s1], var[s0:s1], p[s0:s1]
            exp = {}
            for k, v in info_dict.items():
                vb = v[s0:s1]
                if torch.is_tensor(v):
                    vb = vb.to(device=self.device, dtype=self.dtype if vb.is_floating_point() else None)
                    vb = vb.unsqueeze(1).expand(bs, self.num_samples, *vb.shape[1:])
                exp[k] = vb
            final = None
            for step in range(self.n_steps):
                cand = torch.randn(bs, self.num_samples, self.horizon, self.action_dim, generator=self.torch_gen,
                                   device=self.device, dtype=self.dtype) * bv.unsqueeze(1) + bm.unsqueeze(1)
                u = torch.rand(bs, self.num_samples, self.horizon, generator=self.torch_gen, device=self.device,
                               dtype=self.dtype)
                bits = (u < bp.unsqueeze(1)).to(self.dtype)
                bits[:, 0] = (bp > 0.5).to(self.dtype)  # first sample = current mean / mode
                cand[:, 0] = bm
                cand = self._apply(cand, bits)
                costs = self._score(exp, cand)
                topv, topi = torch.topk(costs, k=self.topk, dim=1, largest=False)
                bi = torch.arange(bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                el = cand[bi, topi]
                bm, bv = el.mean(1), el.std(1)
                bp = bits[bi, topi].mean(1).clamp(self.p_min, 1 - self.p_min)
                final = topv.mean(1).cpu().tolist()
            bm = self._apply(bm, (bp > 0.5).to(self.dtype))
            mean[s0:s1], var[s0:s1], p[s0:s1] = bm, bv, bp
            outputs['costs'].extend(final)
        outputs['actions'] = mean.detach().cpu()
        outputs['mean'] = [mean.detach().cpu()]
        outputs['var'] = [var.detach().cpu()]
        outputs['grip_p'] = p.detach().cpu()
        print(f'CEM (discrete gripper) solve time: {time.time() - start_time:.4f} seconds')
        return outputs


class PriorInitCEMSolver(CEMSolver):
    """CEM whose initial distribution is the model's action prior along its prior-mean imagined rollout
    (``model.prior_plan(info_dict, horizon) -> mean, std``), with the std scaled by ``std_scale``."""

    def __init__(self, *args, std_scale: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.std_scale = float(std_scale)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        mu, sd = self.model.prior_plan(info_dict, self.horizon)
        mu, sd = mu.to(self.dtype).cpu(), (self.std_scale * sd).to(self.dtype).cpu()
        self.init_action_distrib = lambda n, actions=None: (mu, sd)
        try:
            return super().solve(info_dict, None)
        finally:
            del self.init_action_distrib
