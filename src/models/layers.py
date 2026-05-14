import torch
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, node_mask):
        """
        X: bs, n, dx
        node_mask: bs, n
        """
        # (bs, n, 1)
        mask = node_mask.unsqueeze(-1).type_as(X)

        # Zero out padding node features to counteract non-zero bias from LayerNorm
        X = X * mask

        # Number of valid nodes, avoid division by zero
        N = mask.sum(dim=1).clamp(min=1.0)

        # Compute statistics only over valid nodes
        m = X.sum(dim=1) / N

        # Build masked tensors for min / max
        X_for_max = X.clone()
        X_for_max[mask.expand_as(X) == 0] = float("-inf")
        ma = X_for_max.max(dim=1)[0]

        X_for_min = X.clone()
        X_for_min[mask.expand_as(X) == 0] = float("inf")
        mi = X_for_min.min(dim=1)[0]

        # Compute variance / standard deviation with mask
        variance = ((X - m.unsqueeze(1)) ** 2 * mask).sum(dim=1) / N
        std = torch.sqrt(variance + 1e-6)

        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, node_mask):
        """
        E: bs, n, n, de
        node_mask: bs, n
        """
        # Build edge mask (bs, n, n, 1)
        mask_n = node_mask.unsqueeze(-1)
        mask_e = (mask_n.unsqueeze(2) * mask_n.unsqueeze(1)).type_as(E)

        # Zero out padding edges
        E = E * mask_e

        # Number of valid edges
        N_sq = mask_e.sum(dim=(1, 2)).clamp(min=1.0)

        # Compute statistics only over valid edges
        m = E.sum(dim=(1, 2)) / N_sq

        E_for_max = E.clone()
        E_for_max[mask_e.expand_as(E) == 0] = float("-inf")
        ma = E_for_max.max(dim=2)[0].max(dim=1)[0]

        E_for_min = E.clone()
        E_for_min[mask_e.expand_as(E) == 0] = float("inf")
        mi = E_for_min.min(dim=2)[0].min(dim=1)[0]

        variance = ((E - m.unsqueeze(1).unsqueeze(1)) ** 2 * mask_e).sum(
            dim=(1, 2)
        ) / N_sq
        std = torch.sqrt(variance + 1e-6)

        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
