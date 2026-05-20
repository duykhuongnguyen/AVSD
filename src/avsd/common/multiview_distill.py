import math

import torch
import torch.nn.functional as F


def build_candidate_union(topk_indices_by_view: list[torch.Tensor], sampled_token_ids: torch.Tensor):
    """
    Inputs:
      topk_indices_by_view: list of [B, T, K]
      sampled_token_ids:    [B, T]
    Returns:
      candidate_indices: [B, T, Kc_max]
      candidate_mask:    [B, T, Kc_max] bool
    """

    if not topk_indices_by_view:
        raise ValueError("topk_indices_by_view must contain at least one view.")

    batch_size, seq_len = sampled_token_ids.shape
    candidate_size = sum(indices.shape[-1] for indices in topk_indices_by_view) + 1

    candidate_indices = sampled_token_ids.unsqueeze(-1).expand(batch_size, seq_len, candidate_size).clone()
    candidate_mask = torch.zeros(
        batch_size,
        seq_len,
        candidate_size,
        dtype=torch.bool,
        device=sampled_token_ids.device,
    )

    flattened_sampled = sampled_token_ids.reshape(-1)
    flattened_candidates = candidate_indices.reshape(-1, candidate_size)
    flattened_mask = candidate_mask.reshape(-1, candidate_size)
    flattened_views = [indices.reshape(-1, indices.shape[-1]) for indices in topk_indices_by_view]

    for flat_idx in range(flattened_candidates.shape[0]):
        seen = set()
        ordered = []
        for view_indices in flattened_views:
            for token_id in view_indices[flat_idx].tolist():
                if token_id not in seen:
                    seen.add(token_id)
                    ordered.append(token_id)

        sampled_token = int(flattened_sampled[flat_idx].item())
        if sampled_token not in seen:
            ordered.append(sampled_token)
            seen.add(sampled_token)

        valid_count = len(ordered)
        flattened_candidates[flat_idx, :valid_count] = torch.tensor(
            ordered,
            dtype=flattened_candidates.dtype,
            device=flattened_candidates.device,
        )
        flattened_mask[flat_idx, :valid_count] = True

    return candidate_indices, candidate_mask


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    mask = mask.to(dtype=torch.bool)
    very_negative = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.full_like(logits, very_negative))
    log_probs = F.log_softmax(masked_logits, dim=dim)
    return torch.where(mask, log_probs, torch.full_like(log_probs, float("-inf")))


def pairwise_jsd_per_token(logp_a: torch.Tensor, logp_b: torch.Tensor, mask: torch.Tensor):
    """Returns [B, T]"""

    mask = mask.to(dtype=torch.bool)
    log_half = math.log(0.5)
    mix_log_probs = torch.logsumexp(
        torch.stack([logp_a + log_half, logp_b + log_half], dim=0),
        dim=0,
    )

    probs_a = torch.where(mask, logp_a.exp(), torch.zeros_like(logp_a))
    probs_b = torch.where(mask, logp_b.exp(), torch.zeros_like(logp_b))
    safe_logp_a = torch.where(mask, logp_a, torch.zeros_like(logp_a))
    safe_logp_b = torch.where(mask, logp_b, torch.zeros_like(logp_b))
    safe_mix_log = torch.where(mask, mix_log_probs, torch.zeros_like(mix_log_probs))

    kl_a = torch.sum(probs_a * (safe_logp_a - safe_mix_log), dim=-1)
    kl_b = torch.sum(probs_b * (safe_logp_b - safe_mix_log), dim=-1)
    return 0.5 * (kl_a + kl_b)


def compute_view_weights(
    view_log_probs: torch.Tensor,
    candidate_mask: torch.Tensor,
    mode: str = "uniform",
    agreement_eta: float = 5.0,
):
    """
    Inputs:
      view_log_probs: [B, M, T, Kc]
    Returns:
      weights: [B, T, M]
      aux: dict of stats
    """

    batch_size, num_views, seq_len, _ = view_log_probs.shape
    device = view_log_probs.device
    dtype = view_log_probs.dtype

    if mode == "uniform":
        weights = torch.full(
            (batch_size, seq_len, num_views),
            1.0 / num_views,
            dtype=dtype,
            device=device,
        )
        centrality = torch.zeros(batch_size, seq_len, num_views, dtype=dtype, device=device)
    elif mode == "agreement_centrality":
        if num_views == 1:
            weights = torch.ones(batch_size, seq_len, 1, dtype=dtype, device=device)
            centrality = torch.zeros(batch_size, seq_len, 1, dtype=dtype, device=device)
        else:
            centrality = torch.zeros(batch_size, seq_len, num_views, dtype=dtype, device=device)
            for view_idx in range(num_views):
                distance_sum = torch.zeros(batch_size, seq_len, dtype=dtype, device=device)
                for other_idx in range(num_views):
                    if view_idx == other_idx:
                        continue
                    distance_sum = distance_sum + pairwise_jsd_per_token(
                        view_log_probs[:, view_idx],
                        view_log_probs[:, other_idx],
                        candidate_mask,
                    )
                centrality[:, :, view_idx] = distance_sum / (num_views - 1)
            weights = torch.softmax(-agreement_eta * centrality, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        raise ValueError(f"Unsupported view weighting mode: {mode}")

    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
    return weights, {"centrality": centrality, "entropy": entropy}


def build_consensus_target(view_log_probs: torch.Tensor, weights: torch.Tensor, candidate_mask: torch.Tensor):
    """
    Returns log target probs [B, T, Kc]
    """

    weights_bmt = weights.permute(0, 2, 1).unsqueeze(-1)
    weighted_log_probs = (weights_bmt * view_log_probs).sum(dim=1)
    return masked_log_softmax(weighted_log_probs, candidate_mask, dim=-1)


def build_arithmetic_target(view_log_probs: torch.Tensor, candidate_mask: torch.Tensor):
    """
    Uniform arithmetic teacher over candidate-normalized view distributions.

    Returns log target probs [B, T, Kc].
    """

    if view_log_probs.ndim != 4:
        raise ValueError("view_log_probs must have shape [batch, views, seq_len, candidates].")

    valid_mask = candidate_mask.unsqueeze(1)
    view_probs = torch.where(valid_mask, view_log_probs.exp(), torch.zeros_like(view_log_probs))
    arithmetic_probs = view_probs.mean(dim=1)
    arithmetic_probs = torch.where(candidate_mask, arithmetic_probs, torch.zeros_like(arithmetic_probs))
    arithmetic_probs = arithmetic_probs / arithmetic_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    arithmetic_log_probs = torch.log(arithmetic_probs.clamp_min(1e-8))
    return torch.where(candidate_mask, arithmetic_log_probs, torch.full_like(arithmetic_log_probs, float("-inf")))


def build_avsd_target(
    student_log_probs: torch.Tensor,
    view_log_probs: torch.Tensor,
    weights: torch.Tensor,
    candidate_mask: torch.Tensor,
    gate_alpha: float,
    gate_var_coef: float,
    gate_gap_coef: float,
    sign_threshold: float,
    gate_mode: str = "sigmoid",
    consistency_exp_variance_mode: str = "on",
    consistency_exp_var_coef: float = 1.0,
):
    """
    Returns:
      target_log_probs: [B, T, Kc]
      aux: {gate, jensen_gap, ...} with mode-specific diagnostics
    """
    eps = 1e-8

    weights_bmt = weights.permute(0, 2, 1).unsqueeze(-1)
    view_probs = view_log_probs.exp()
    q_a = (weights_bmt * view_probs).sum(dim=1)
    q_a = q_a / q_a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_q_a = torch.log(q_a.clamp_min(1e-8))

    log_q_g = build_consensus_target(view_log_probs, weights, candidate_mask)

    valid_mask = candidate_mask.unsqueeze(1)
    safe_view_log_probs = torch.where(valid_mask, view_log_probs, torch.zeros_like(view_log_probs))
    safe_student_log_probs = torch.where(
        candidate_mask,
        student_log_probs,
        torch.zeros_like(student_log_probs),
    )

    delta = safe_view_log_probs - safe_student_log_probs.unsqueeze(1)
    delta_mean = (weights_bmt * delta).sum(dim=1)
    delta_abs_mean = (weights_bmt * delta.abs()).sum(dim=1)
    # C_t(v): view agreement under the weighted advantage family.
    consistency = delta_mean.abs() / (delta_abs_mean + eps)
    # V_t(v): variance of view advantages around the weighted mean.
    var_adv = (weights_bmt * (delta - delta_mean.unsqueeze(1)).pow(2)).sum(dim=1)
    # Legacy sign-based statistic used by the original sigmoid AVSD gate.
    sign_consistency = (weights_bmt * delta.sign()).sum(dim=1).abs()

    raw_jensen_gap = log_q_a - log_q_g
    # J_t(v): positive Jensen residual between the arithmetic and geometric teachers.
    jensen_gap = torch.where(candidate_mask, raw_jensen_gap.clamp_min(0.0), torch.zeros_like(raw_jensen_gap))

    if gate_mode == "sigmoid":
        variance_factor = torch.ones_like(var_adv)
        gate = torch.sigmoid(
            gate_alpha * (sign_consistency - sign_threshold)
            - gate_var_coef * var_adv
            - gate_gap_coef * jensen_gap
        )
    elif gate_mode == "consistency_exp":
        if consistency_exp_variance_mode == "on":
            variance_factor = torch.exp(-consistency_exp_var_coef * var_adv)
        elif consistency_exp_variance_mode == "off":
            variance_factor = torch.ones_like(var_adv)
        else:
            raise ValueError(
                "consistency_exp_variance_mode must be one of: on, off"
            )
        gate = consistency * variance_factor
    elif gate_mode == "avsd":
        num_views = view_log_probs.shape[1]
        expected_uniform = torch.full_like(weights, 1.0 / num_views)
        if not torch.allclose(weights, expected_uniform, atol=1e-6, rtol=1e-6):
            raise ValueError("avsd requires uniform view weights.")
        consensus_adv = delta_mean
        avsd = consensus_adv.abs() / (delta_abs_mean + eps)
        gate = avsd * consensus_adv.abs() / (consensus_adv.abs() + jensen_gap + eps)
    else:
        raise ValueError(f"Unsupported AVSD gate mode: {gate_mode}")

    gate = torch.where(candidate_mask, gate, torch.zeros_like(gate))

    target_log_probs = masked_log_softmax(log_q_g + gate * jensen_gap, candidate_mask, dim=-1)
    aux = {
        "gate": gate,
        "jensen_gap": jensen_gap,
        "log_q_a": log_q_a,
        "log_q_g": log_q_g,
    }
    if gate_mode == "avsd":
        aux.update(
            {
                "consensus_adv": torch.where(candidate_mask, consensus_adv, torch.zeros_like(consensus_adv)),
                "avsd": torch.where(
                    candidate_mask,
                    avsd,
                    torch.zeros_like(avsd),
                ),
            }
        )
    else:
        aux.update(
            {
                "variance_factor": torch.where(candidate_mask, variance_factor, torch.zeros_like(variance_factor)),
                "consistency": torch.where(candidate_mask, consistency, torch.zeros_like(consistency)),
                "sign_consistency": sign_consistency,
                "var_adv": var_adv,
            }
        )
    return target_log_probs, aux


def build_uniform_sampled_tinker_target(
    student_sample_log_probs: torch.Tensor,
    view_sample_log_probs: torch.Tensor,
    mode: str,
    gate_alpha: float,
    gate_var_coef: float,
    gate_gap_coef: float,
    sign_threshold: float,
    gate_mode: str = "sigmoid",
    consistency_exp_variance_mode: str = "on",
    consistency_exp_var_coef: float = 1.0,
):
    """
    Build a sampled-token-only multi-view target for the Tinker loss.

    Inputs:
      student_sample_log_probs: [B, T]
      view_sample_log_probs:    [B, M, T]

    Returns:
      target_sample_log_probs: [B, T]
      aux: diagnostics with sampled-token AVSD quantities

    The consensus baseline is the uniform geometric teacher. The arithmetic
    baseline is the uniform arithmetic teacher. AVSD starts from consensus and
    adds a gated Jensen residual toward arithmetic, all evaluated only at the
    sampled token.
    """
    eps = 1e-8
    if view_sample_log_probs.ndim != 3:
        raise ValueError("view_sample_log_probs must have shape [batch, views, seq_len].")
    if student_sample_log_probs.shape != view_sample_log_probs.shape[::2]:
        raise ValueError("student_sample_log_probs must have shape [batch, seq_len].")
    if mode not in {"consensus", "arithmetic", "avsd"}:
        raise ValueError("mode must be one of: consensus, arithmetic, avsd")

    num_views = view_sample_log_probs.shape[1]
    log_q_g = view_sample_log_probs.mean(dim=1)
    log_q_a = torch.logsumexp(view_sample_log_probs, dim=1) - math.log(num_views)
    jensen_gap = (log_q_a - log_q_g).clamp_min(0.0)

    aux = {
        "jensen_gap": jensen_gap,
        "log_q_a": log_q_a,
        "log_q_g": log_q_g,
    }

    if mode == "consensus":
        return log_q_g, aux
    if mode == "arithmetic":
        return log_q_a, aux

    delta = view_sample_log_probs - student_sample_log_probs.unsqueeze(1)
    delta_mean = delta.mean(dim=1)
    delta_abs_mean = delta.abs().mean(dim=1)
    consistency = delta_mean.abs() / (delta_abs_mean + eps)
    var_adv = (delta - delta_mean.unsqueeze(1)).pow(2).mean(dim=1)
    sign_consistency = delta.sign().mean(dim=1).abs()

    if gate_mode == "sigmoid":
        variance_factor = torch.ones_like(var_adv)
        gate = torch.sigmoid(
            gate_alpha * (sign_consistency - sign_threshold)
            - gate_var_coef * var_adv
            - gate_gap_coef * jensen_gap
        )
    elif gate_mode == "consistency_exp":
        if consistency_exp_variance_mode == "on":
            variance_factor = torch.exp(-consistency_exp_var_coef * var_adv)
        elif consistency_exp_variance_mode == "off":
            variance_factor = torch.ones_like(var_adv)
        else:
            raise ValueError("consistency_exp_variance_mode must be one of: on, off")
        gate = consistency * variance_factor
    elif gate_mode == "avsd":
        consensus_adv = delta_mean
        avsd = consensus_adv.abs() / (delta_abs_mean + eps)
        gate = avsd * consensus_adv.abs() / (consensus_adv.abs() + jensen_gap + eps)
    else:
        raise ValueError(f"Unsupported AVSD gate mode: {gate_mode}")

    target_sample_log_probs = log_q_g + gate * jensen_gap
    aux["gate"] = gate
    if gate_mode == "avsd":
        aux.update(
            {
                "consensus_adv": consensus_adv,
                "avsd": avsd,
            }
        )
    else:
        aux.update(
            {
                "variance_factor": variance_factor,
                "consistency": consistency,
                "sign_consistency": sign_consistency,
                "var_adv": var_adv,
            }
        )
    return target_sample_log_probs, aux


def sampled_token_tinker_loss_from_log_probs(
    student_sample_log_probs: torch.Tensor,
    target_sample_log_probs: torch.Tensor,
    labels: torch.Tensor,
):
    label_mask = labels != -100
    finite_mask = torch.isfinite(student_sample_log_probs) & torch.isfinite(target_sample_log_probs)
    mask = label_mask & finite_mask
    if not torch.any(mask):
        return student_sample_log_probs.new_zeros(())

    advantage = (target_sample_log_probs - student_sample_log_probs).detach()
    return -(advantage[mask] * student_sample_log_probs[mask]).mean()


def sampled_epistemic_preservation_loss(
    student_sample_log_probs: torch.Tensor,
    gate: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
):
    label_mask = labels != -100
    finite_mask = torch.isfinite(student_sample_log_probs) & torch.isfinite(gate)
    mask = label_mask & finite_mask
    if not torch.any(mask):
        return student_sample_log_probs.new_zeros(())

    sampled_entropy_estimate = -student_sample_log_probs
    per_token = -tau * (1.0 - gate) * sampled_entropy_estimate
    return per_token[mask].mean()


def generalized_jsd_from_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.5,
    token_clip: float | None = None,
):
    valid_mask = torch.isfinite(student_log_probs) & torch.isfinite(teacher_log_probs)

    safe_student_log = torch.where(valid_mask, student_log_probs, torch.zeros_like(student_log_probs))
    safe_teacher_log = torch.where(valid_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
    student_probs = torch.where(valid_mask, student_log_probs.exp(), torch.zeros_like(student_log_probs))
    teacher_probs = torch.where(valid_mask, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs))

    if beta == 0:
        jsd = torch.sum(teacher_probs * (safe_teacher_log - safe_student_log), dim=-1)
    elif beta == 1:
        jsd = torch.sum(student_probs * (safe_student_log - safe_teacher_log), dim=-1)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mix_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ],
                dim=0,
            ),
            dim=0,
        )
        safe_mix_log = torch.where(valid_mask, mix_log_probs, torch.zeros_like(mix_log_probs))
        kl_teacher = torch.sum(teacher_probs * (safe_teacher_log - safe_mix_log), dim=-1)
        kl_student = torch.sum(student_probs * (safe_student_log - safe_mix_log), dim=-1)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    label_mask = labels != -100
    if not torch.any(label_mask):
        return jsd.new_zeros(())
    return jsd[label_mask].mean()


def epistemic_preservation_loss(student_log_probs: torch.Tensor, gate: torch.Tensor, labels: torch.Tensor, tau: float):
    valid_mask = torch.isfinite(student_log_probs)
    student_probs = torch.where(valid_mask, student_log_probs.exp(), torch.zeros_like(student_log_probs))
    safe_student_log = torch.where(valid_mask, student_log_probs, torch.zeros_like(student_log_probs))

    entropy = -(student_probs * safe_student_log).sum(dim=-1)
    lambda_bar = (student_probs * gate).sum(dim=-1)
    per_token = -tau * (1.0 - lambda_bar) * entropy

    label_mask = labels != -100
    if not torch.any(label_mask):
        return per_token.new_zeros(())
    return per_token[label_mask].mean()
