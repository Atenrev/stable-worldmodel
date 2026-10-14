"""CEM with a discrete gripper: the gripper command of every action block is sampled as one of two values
(Bernoulli per block, probability refit from the elites), the other dimensions as in CEMSolver."""

import time

import numpy as np
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
    (``model.prior_plan(info_dict, horizon) -> mean, std``), with the std scaled by ``std_scale``.

    Options:
      fixed_std: use this scalar std for every dimension instead of ``std_scale`` x the prior's std.
      mix: first iteration only, replace half the population by plain-CEM samples N(0, var_scale) so that CEM
        selection can fall back on the unit Gaussian when the proposal is wrong.
      hold_arm: z-scored per-step arm action of zero motion (6 values); adds two first-iteration candidates that
        hold still with the gripper open / closed (``hold_grip``, z-scored).
      stop_log: after each solve, print per env the encoded current cost to the goal (c0), the predicted cost of the
        best hold candidate and of the chosen plan, and the env's state_success flag if present ("STOPLOG ...").
      stop_threshold: if c0 < threshold, execute the best hold candidate instead of CEM's plan (needs hold_arm).
    """

    def __init__(self, *args, std_scale: float = 2.0, fixed_std: float | None = None, mix: bool = False,
                 hold_arm=None, hold_grip=(-0.723, 1.404), stop_log: bool = False,
                 stop_threshold: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.std_scale = float(std_scale)
        self.fixed_std = None if fixed_std is None else float(fixed_std)
        self.mix = bool(mix)
        self.hold_arm = None if hold_arm is None else [float(x) for x in hold_arm]
        self.hold_grip = [float(x) for x in hold_grip]
        self.stop_log = bool(stop_log)
        self.stop_threshold = None if stop_threshold is None else float(stop_threshold)

    def _modify_candidates(self, candidates, step):
        if step != 0:
            return candidates
        n = candidates.shape[1]
        if self.mix:
            noise = torch.randn(candidates[:, n // 2:].shape, generator=self.torch_gen, device=candidates.device,
                                dtype=candidates.dtype)
            candidates[:, n // 2:] = self.var_scale * noise
        if self.hold_arm is not None:
            A = candidates.shape[-1]
            for i, g in enumerate(self.hold_grip):
                step_a = torch.tensor(self.hold_arm + [g], device=candidates.device, dtype=candidates.dtype)
                candidates[:, n - 1 - i] = step_a.repeat(A // len(step_a))
        return candidates

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        mu, sd = self.model.prior_plan(info_dict, self.horizon)
        print(f'prior_plan mean std {sd.float().mean().item():.4f}')
        sd = torch.full_like(sd, self.fixed_std) if self.fixed_std is not None else self.std_scale * sd
        mu, sd = mu.to(self.dtype).cpu(), sd.to(self.dtype).cpu()
        self.init_action_distrib = lambda n, actions=None: (mu, sd)
        try:
            out = super().solve(info_dict, None)
        finally:
            del self.init_action_distrib
        if (self.stop_log or self.stop_threshold is not None) and self.hold_arm is not None:
            self._stop_rule(info_dict, out)
        return out

    def _stop_rule(self, info_dict, out):
        """Score [hold open, hold closed, chosen plan] with the model; c0 = encoded current latent's cost to the goal."""
        plan = out['actions'].to(self.device, self.dtype)  # (B, H, A)
        B, Hh, A = plan.shape
        holds = []
        for g in self.hold_grip:
            step_a = torch.tensor(self.hold_arm + [g], device=self.device, dtype=self.dtype)
            holds.append(step_a.repeat(A // len(step_a)).expand(B, Hh, A))
        cands = torch.stack(holds + [plan], 1)  # (B, 3, H, A)
        n = cands.shape[1]
        ex = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                v = v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else None)
                ex[k] = v.unsqueeze(1).expand(B, n, *v.shape[1:])
            elif isinstance(v, np.ndarray):
                ex[k] = np.repeat(v[:, None, ...], n, axis=1)
        cost = self.model.get_cost(ex, cands).float()  # (B, 3)
        pe, ge = ex['predicted_emb'], ex['goal_emb']
        c0 = (pe[:, 0, 0].float() - ge[:, 0, -1].float()).pow(2).flatten(1).sum(-1)  # (B,)
        succ = info_dict.get('state_success')
        if succ is not None:
            succ = np.asarray(succ.cpu() if torch.is_tensor(succ) else succ).reshape(B, -1)[:, -1]
        hold_best = cost[:, :2].argmin(1)
        for i in range(B):
            if self.stop_log:
                s_i = 'NA' if succ is None else int(bool(succ[i]))
                print(f'STOPLOG c0 {c0[i].item():.4f} hold {cost[i, hold_best[i]].item():.4f} '
                      f'plan {cost[i, 2].item():.4f} succ {s_i}')
            if self.stop_threshold is not None and c0[i].item() < self.stop_threshold:
                out['actions'][i] = cands[i, hold_best[i]].to(out['actions'].dtype).cpu()
