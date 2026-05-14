import torch
import torch.nn.functional as F

class ContinuousRateMatrixDesigner:
    def __init__(self, limit_dist):
        self.limit_dist = limit_dist
        self.num_classes_X = len(self.limit_dist.X)
        self.num_classes_E = len(self.limit_dist.E)
        
        # p0 distributions
        self.p0_X = self.limit_dist.X  # (dx,)
        self.p0_E = self.limit_dist.E  # (de,)

    def update_limit_dist(self, limit_dist):
        """Update the internal limit distribution."""
        self.limit_dist = limit_dist
        self.p0_X = self.limit_dist.X
        self.p0_E = self.limit_dist.E


    def compute_graph_rate_matrix(self, t, node_mask, G_t, G_1_pred):
        """
        Compute explicit transition rates based on the user's provided derivation:
        R_t(z_t, k) = [ p_t(k)(1 + p0(z_t) - p0(k)) + (1 - p_t(z_t) - p_t(k)) * ReLU(p0(z_t) - p0(k)) ]
                      / [ S * (1-t) * p0(z_t) ]
        """

        X_t, E_t = G_t
        X_1_pred, E_1_pred = G_1_pred  # These are \hat{p}_t(k)

        # Get indices of current states
        X_t_idx = X_t.argmax(dim=-1)      # (bs, n)
        E_t_idx = E_t.argmax(dim=-1)      # (bs, n, n)
        
        # t is likely (bs, 1). Expand for broadcasting
        t_X = t.view(-1, 1, 1)            # (bs, 1, 1)
        t_E = t.view(-1, 1, 1, 1)         # (bs, 1, 1, 1)

        # --- Compute Rates for Nodes (X) ---
        bs, n = X_t.shape[:2]
        S_X = self.num_classes_X
        
        # Prepare p0 values
        p0_X = self.p0_X.to(X_t.device)   # (S_X,)
        p0_X_zt = p0_X[X_t_idx]           # (bs, n)
        p0_X_k = p0_X.view(1, 1, S_X)     # (1, 1, S_X)
        
        # Prepare \hat{p}_t values
        pt_X_k = X_1_pred                 # (bs, n, S_X)
        pt_X_zt = pt_X_k.gather(-1, X_t_idx.unsqueeze(-1)).squeeze(-1) # (bs, n)
        # Gather the probability of the current state zt from pt_X_k (expand dim to gather, then squeeze)
        
        # Denominator: S * (1-t) * p0(z_t)
        # Add epsilon to (1-t) for stability near t=1
        denom_X = S_X * (1 - t_X + 1e-6) * p0_X_zt.unsqueeze(-1) # (bs, n, 1)
        
        # Term 1: p_t(k) * (1 + p0(z_t) - p0(k))
        term1_X = pt_X_k * (1 + p0_X_zt.unsqueeze(-1) - p0_X_k) # (bs, n, S_X)
        
        # Term 2: (1 - p_t(z_t) - p_t(k)) * ReLU(p0(z_t) - p0(k))
        relu_diff_X = F.relu(p0_X_zt.unsqueeze(-1) - p0_X_k) # (bs, n, S_X)
        factor_X = 1 - pt_X_zt.unsqueeze(-1) - pt_X_k      # (bs, n, S_X)
        term2_X = factor_X * relu_diff_X
        
        R_t_X = (term1_X + term2_X) / denom_X
        
        # Zero out diagonal (self-loops)
        mask_diag_X = torch.eye(S_X, device=X_t.device).view(1, 1, S_X, S_X)
        # However, R_t_X is (bs, n, S_X) where S_X is destination class k.
        # We need to zero out R_t_X[..., zt].   
        # Create a mask where index == zt
        mask_self_X = F.one_hot(X_t_idx, num_classes=S_X).bool()
        R_t_X.masked_fill_(mask_self_X, 0.0)

        # --- Compute Rates for Edges (E) ---
        S_E = self.num_classes_E
        
        # Prepare p0 values
        p0_E = self.p0_E.to(E_t.device)
        p0_E_zt = p0_E[E_t_idx]           # (bs, n, n)
        p0_E_k = p0_E.view(1, 1, 1, S_E)  # (1, 1, 1, S_E)
        
        # Prepare \hat{p}_t values
        pt_E_k = E_1_pred                 # (bs, n, n, S_E)
        pt_E_zt = pt_E_k.gather(-1, E_t_idx.unsqueeze(-1)).squeeze(-1) # (bs, n, n)
        
        # Denominator
        denom_E = S_E * (1 - t_E + 1e-6) * p0_E_zt.unsqueeze(-1)
        
        # Term 1
        term1_E = pt_E_k * (1 + p0_E_zt.unsqueeze(-1) - p0_E_k)
        
        # Term 2
        relu_diff_E = F.relu(p0_E_zt.unsqueeze(-1) - p0_E_k)
        factor_E = 1 - pt_E_zt.unsqueeze(-1) - pt_E_k
        term2_E = factor_E * relu_diff_E
        
        R_t_E = (term1_E + term2_E) / denom_E
        
        # Zero out diagonal
        mask_self_E = F.one_hot(E_t_idx, num_classes=S_E).bool()
        R_t_E.masked_fill_(mask_self_E, 0.0)

        # --- Masking ---
        if node_mask is not None:
             # node_mask is (bs, n)
             # X: (bs, n, S_X) -> valid nodes only
             x_mask = node_mask.unsqueeze(-1)
             R_t_X = R_t_X * x_mask
             
             # E: (bs, n, n, S_E) -> valid edges only
             # Edge is valid if both source and dest nodes are valid
             e_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # (bs, n, n)
             R_t_E = R_t_E * e_mask.unsqueeze(-1)

        return R_t_X, R_t_E
