#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flow_matching.rate_matrix_continuous import ContinuousRateMatrixDesigner  # noqa: E402
import utils  # noqa: E402


def _rand_simplex(num_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Full-support prior: strictly positive and normalized.
    probs = torch.rand(num_classes, device=device, dtype=dtype) + 0.1
    return probs / probs.sum()


def _ensure_full_support(p0: torch.Tensor, eps: float) -> tuple[torch.Tensor, bool, int]:
    if p0.dim() != 1:
        raise ValueError(f"p0 must be 1D (S,), got shape {tuple(p0.shape)}")
    if (p0 < 0).any():
        raise ValueError("p0 must be non-negative.")
    if eps <= 0:
        return p0, False, 0

    zero_mask = p0 <= 0
    num_zeros = int(zero_mask.sum().item())
    if num_zeros == 0:
        return p0, False, 0

    p0 = p0.clone()
    p0[zero_mask] = float(eps)
    p0 = p0 / p0.sum()
    return p0, True, num_zeros


def _explicit_rstar_all_conditions(
    *,
    p0: torch.Tensor,  # (S,)
    x_t_label: torch.Tensor,  # (..., 1) long
    t: torch.Tensor,  # (B, 1) float
) -> torch.Tensor:
    """
    Explicit construction of the conditional ideal rate R_t^*(x_t -> k | x1=c) for all (k, c):
        R_t^* = ReLU( dt p_t(k|c) - dt p_t(x_t|c) ) / ( Z_t(c) * p_t(x_t|c) ).

    With full-support p0 and t<1 for the linear path, Z_t(c)=S for all c.
    """
    if p0.dim() != 1:
        raise ValueError(f"p0 must be 1D (S,), got shape {tuple(p0.shape)}")
    if x_t_label.dtype != torch.long:
        raise ValueError(f"x_t_label must be torch.long, got {x_t_label.dtype}")
    if x_t_label.shape[-1] != 1:
        raise ValueError(f"x_t_label must have last dim 1, got shape {tuple(x_t_label.shape)}")

    s = p0.shape[0]
    prefix_dims = x_t_label.shape[:-1]
    device = x_t_label.device
    dtype = p0.dtype

    # I_kc[k,c] = 1[k=c], expanded to (..., S_target, S_cond)
    eye = torch.eye(s, device=device, dtype=dtype)
    i_kc = eye.view(*([1] * len(prefix_dims)), s, s).expand(*prefix_dims, s, s)

    # I_xtc[c] = 1[x_t=c], expanded to (..., S_target, S_cond)
    xt_onehot = F.one_hot(x_t_label.squeeze(-1), num_classes=s).to(dtype)  # (..., S_cond)
    i_xtc = xt_onehot.unsqueeze(-2).expand(*prefix_dims, s, s)  # (..., S_target, S_cond)

    # p0(k) and p0(x_t)
    p0_k = p0.view(*([1] * len(prefix_dims)), s, 1).expand(*prefix_dims, s, s)
    p0_xt = p0.gather(0, x_t_label.squeeze(-1).reshape(-1)).reshape(*prefix_dims)  # (...,)
    p0_xt_full = p0_xt.view(*prefix_dims, 1, 1).expand(*prefix_dims, s, s)

    # delta = (1[k=c] - 1[x_t=c]) + (p0(x_t) - p0(k))
    delta = i_kc - i_xtc + (p0_xt_full - p0_k)
    numer = F.relu(delta)

    # p_t(x_t|c) = t*1[x_t=c] + (1-t)*p0(x_t)
    t_b = t.to(dtype=dtype)
    for _ in range(len(prefix_dims) - 1):
        t_b = t_b.unsqueeze(1)
    p0_xt_vec = p0_xt.view(*prefix_dims, 1).expand_as(xt_onehot)
    pt_xtc = t_b * xt_onehot + (1.0 - t_b) * p0_xt_vec  # (..., S_cond)

    denom = (float(s) * pt_xtc).clamp_min(1e-30)  # (..., S_cond)
    rates = numer / denom.unsqueeze(-2)  # (..., S_target, S_cond)
    return rates


def _explicit_rtheta_closed_form_rates(
    *,
    p0: torch.Tensor,  # (S,)
    hat_p: torch.Tensor,  # (..., S)
    x_t_label: torch.Tensor,  # (..., 1) long
    t: torch.Tensor,  # (B, 1) float, where B is the leading batch dim of hat_p
) -> torch.Tensor:
    """
    Implements Eq.(closed_form_rate) (i.e., the expected rate R^theta) for off-diagonal transitions (k != x_t).

    Notes:
    - This closed form assumes t<1 and p0 has full support (p0>0).
    - For k==x_t, the transition rate is defined as 0 (no self-transition).
    """
    if p0.dim() != 1:
        raise ValueError(f"p0 must be 1D (S,), got shape {tuple(p0.shape)}")
    if x_t_label.dtype != torch.long:
        raise ValueError(f"x_t_label must be torch.long, got {x_t_label.dtype}")
    if x_t_label.shape[-1] != 1:
        raise ValueError(f"x_t_label must have last dim 1, got shape {tuple(x_t_label.shape)}")

    s = p0.shape[0]
    if hat_p.shape[-1] != s:
        raise ValueError(f"hat_p last dim must be S={s}, got shape {tuple(hat_p.shape)}")

    # Broadcast p0 to hat_p shape.
    p0_b = p0.view(*([1] * (hat_p.dim() - 1)), s)
    p0_xt = p0_b.gather(-1, x_t_label)  # (..., 1)

    hat_xt = hat_p.gather(-1, x_t_label)  # (..., 1)
    hat_k = hat_p  # (..., S)

    relu_term = F.relu(p0_xt - p0_b)  # (..., S)
    numer = hat_k * (1.0 + p0_xt - p0_b) + (1.0 - hat_xt - hat_k) * relu_term  # (..., S)

    # Z_t == S when p0 has full support and t<1 for this linear path.
    # t is (B,1); broadcast to hat_p.
    while t.dim() < hat_p.dim():
        t = t.unsqueeze(-1)
    denom = (float(s) * (1.0 - t) * p0_xt).clamp_min(1e-30)  # (..., 1)

    rates = numer / denom  # (..., S)
    rates = rates.scatter(-1, x_t_label, 0.0)
    return rates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify equivalence between code and explicit formulas.\n"
            "- mode=rstar: compare conditional ideal rate R* matrices.\n"
            "- mode=rtheta: compare expected rate R^theta to Eq.(closed_form_rate).\n"
        )
    )
    parser.add_argument("--mode", type=str, default="rstar", choices=["rstar", "rtheta"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--dx", type=int, default=7)
    parser.add_argument("--de", type=int, default=5)
    parser.add_argument("--t", type=float, default=0.35, help="Must be in (0,1).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--p0-eps",
        type=float,
        default=1e-12,
        help="If p0 has zeros, replace zeros with this epsilon then renormalize.",
    )
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--assert-close", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.t < 1.0):
        raise ValueError("--t must be in (0,1) for this equivalence check.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Priors (p0); if they contain zeros, optionally perturb them to get full support.
    p0_x_raw = _rand_simplex(args.dx, device=device, dtype=dtype)
    p0_e_raw = _rand_simplex(args.de, device=device, dtype=dtype)
    p0_x, x_adjusted, x_zeros = _ensure_full_support(p0_x_raw, eps=args.p0_eps)
    p0_e, e_adjusted, e_zeros = _ensure_full_support(p0_e_raw, eps=args.p0_eps)
    if x_adjusted or e_adjusted:
        print(f"Adjusted p0 for full support: X_zeros={x_zeros} E_zeros={e_zeros} eps={args.p0_eps:g}")
    limit_dist = utils.PlaceHolder(
        X=p0_x,
        E=p0_e,
        y=torch.ones(1, device=device, dtype=dtype),
    )

    # Random current discrete state z_t as one-hot.
    x_t_label = torch.randint(args.dx, (args.bs, args.n, 1), device=device)
    x_t = F.one_hot(x_t_label.squeeze(-1), num_classes=args.dx).to(dtype)

    e_t_label = torch.randint(args.de, (args.bs, args.n, args.n, 1), device=device)
    # Make symmetric to match typical graph edge representation.
    e_t_label = torch.triu(e_t_label.squeeze(-1), diagonal=0)
    e_t_label = e_t_label + e_t_label.transpose(1, 2) - torch.diag_embed(torch.diagonal(e_t_label, dim1=1, dim2=2))
    e_t_label = e_t_label.unsqueeze(-1)
    e_t = F.one_hot(e_t_label.squeeze(-1), num_classes=args.de).to(dtype)

    x_1_pred = None
    e_1_pred = None
    if args.mode == "rtheta":
        # Random posterior predictions hat{p}_t(c) = p_theta(z1=c | z_t).
        x_1_logits = torch.randn(args.bs, args.n, args.dx, device=device, dtype=dtype)
        x_1_pred = F.softmax(x_1_logits, dim=-1)

        e_1_logits = torch.randn(args.bs, args.n, args.n, args.de, device=device, dtype=dtype)
        e_1_logits = 0.5 * (e_1_logits + e_1_logits.transpose(1, 2))
        e_1_pred = F.softmax(e_1_logits, dim=-1)

    # Time tensor.
    t = torch.full((args.bs, 1), float(args.t), device=device, dtype=dtype)

    designer = ContinuousRateMatrixDesigner(
        rdb="general",
        rdb_crit="x_1",
        eta=0.0,
        omega=0.0,
        limit_dist=limit_dist,
    )

    if args.mode == "rstar":
        dfm_variables = designer.compute_dfm_variables_all_states(t, x_t_label, e_t_label)
        r_code_x, r_code_e = designer.compute_Rstar_all_states(dfm_variables)
        r_exp_x = _explicit_rstar_all_conditions(p0=p0_x, x_t_label=x_t_label, t=t)
        r_exp_e = _explicit_rstar_all_conditions(p0=p0_e, x_t_label=e_t_label, t=t)
        label = "R* (code) vs R* (explicit from definition)"
    else:
        assert x_1_pred is not None and e_1_pred is not None
        r_code_x, r_code_e = designer.compute_graph_rate_matrix(
            t=t,
            node_mask=None,
            G_t=(x_t, e_t),
            G_1_pred=(x_1_pred, e_1_pred),
        )
        r_exp_x = _explicit_rtheta_closed_form_rates(p0=p0_x, hat_p=x_1_pred, x_t_label=x_t_label, t=t)
        r_exp_e = _explicit_rtheta_closed_form_rates(p0=p0_e, hat_p=e_1_pred, x_t_label=e_t_label, t=t)
        label = "R^theta (code, eta=omega=0) vs Eq.(closed_form_rate)"

    abs_diff_x = (r_code_x - r_exp_x).abs()
    abs_diff_e = (r_code_e - r_exp_e).abs()

    max_abs_x = abs_diff_x.max().item()
    max_abs_e = abs_diff_e.max().item()

    rel_diff_x = abs_diff_x / (r_exp_x.abs().clamp_min(1e-30))
    rel_diff_e = abs_diff_e / (r_exp_e.abs().clamp_min(1e-30))
    max_rel_x = rel_diff_x.max().item()
    max_rel_e = rel_diff_e.max().item()

    print(label)
    print(f"X: max_abs={max_abs_x:.3e}  max_rel={max_rel_x:.3e}  shape={tuple(r_code_x.shape)}")
    print(f"E: max_abs={max_abs_e:.3e}  max_rel={max_rel_e:.3e}  shape={tuple(r_code_e.shape)}")

    if args.assert_close:
        x_ok = torch.allclose(r_code_x, r_exp_x, atol=args.atol, rtol=args.rtol)
        e_ok = torch.allclose(r_code_e, r_exp_e, atol=args.atol, rtol=args.rtol)
        if not x_ok or not e_ok:
            raise AssertionError(
                f"Not close: X={x_ok} E={e_ok} (atol={args.atol} rtol={args.rtol}). "
                f"max_abs_x={max_abs_x:.3e} max_abs_e={max_abs_e:.3e}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
