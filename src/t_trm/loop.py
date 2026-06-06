import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cmr.model import InputTypes

class DTLGNGate(nn.Module):
    def __init__(self):
        super().__init__()
        # 9 grid points in {-1, 0, 1}^2
        # For each point, we learn logits over the 3 possible outputs {-1, 0, 1}
        self.logits = nn.Parameter(torch.zeros(9, 3)) # Initialize uniformly

    def get_grid_outputs(self, hard=False):
        if hard:
            # Deterministic selection at inference time
            max_indices = torch.argmax(self.logits, dim=-1)
            # Map indices 0, 1, 2 to values -1.0, 0.0, 1.0
            o = max_indices.float() - 1.0
            return o
        else:
            # Soft continuous representation for surrogate training
            p = F.softmax(self.logits, dim=-1) # (9, 3)
            o = p[:, 2] - p[:, 0] # Expected value in [-1, 1]
            return o

    def forward(self, x1, x2, hard=False):
        """
        Bivariate polynomial surrogate mapping.
        Inputs x1, x2 are bounded/clamped to [-1, 1].
        """
        x1 = torch.clamp(x1, -1.0, 1.0)
        x2 = torch.clamp(x2, -1.0, 1.0)
        
        o = self.get_grid_outputs(hard=hard)
        
        # Lagrange basis polynomials for the grid {-1, 0, 1}
        l1_neg1 = 0.5 * x1 * (x1 - 1.0)
        l1_0 = 1.0 - x1 * x1
        l1_pos1 = 0.5 * x1 * (x1 + 1.0)
        
        l2_neg1 = 0.5 * x2 * (x2 - 1.0)
        l2_0 = 1.0 - x2 * x2
        l2_pos1 = 0.5 * x2 * (x2 + 1.0)
        
        # Grid index mapping: (i+1)*3 + (j+1)
        # o[0]: (-1, -1), o[1]: (-1, 0), o[2]: (-1, 1)
        # o[3]: ( 0, -1), o[4]: ( 0, 0), o[5]: ( 0, 1)
        # o[6]: ( 1, -1), o[7]: ( 1, 0), o[8]: ( 1, 1)
        y = (
            o[0] * l1_neg1 * l2_neg1 + o[1] * l1_neg1 * l2_0 + o[2] * l1_neg1 * l2_pos1 +
            o[3] * l1_0 * l2_neg1    + o[4] * l1_0 * l2_0    + o[5] * l1_0 * l2_pos1 +
            o[6] * l1_pos1 * l2_neg1 + o[7] * l1_pos1 * l2_0 + o[8] * l1_pos1 * l2_pos1
        )
        return y

    def compute_losses(self):
        """
        Computes the commitment loss and monotone violations.
        """
        p = F.softmax(self.logits, dim=-1)
        
        # 1. Commitment Loss (forces p to be one-hot)
        commit_loss = torch.sum((p**2 - p)**2)
        
        # 2. Expected output values for monotone checks
        o = self.get_grid_outputs(hard=False)
        
        # Numerical Monotonicity (NM) Loss (non-decreasing along rows and columns)
        nm_loss = (
            torch.relu(o[0] - o[1]) + torch.relu(o[1] - o[2]) +
            torch.relu(o[3] - o[4]) + torch.relu(o[4] - o[5]) +
            torch.relu(o[6] - o[7]) + torch.relu(o[7] - o[8]) +
            torch.relu(o[0] - o[3]) + torch.relu(o[3] - o[6]) +
            torch.relu(o[1] - o[4]) + torch.relu(o[4] - o[7]) +
            torch.relu(o[2] - o[5]) + torch.relu(o[5] - o[8])
        )
        
        # Information Monotonicity (IM) Loss (principled degradation)
        def im_penalty(val_u, val_v):
            return torch.where(val_u >= 0, torch.relu(val_u - val_v), torch.relu(val_v - val_u))
            
        # Relations (u, v) representing u <=_inf v
        im_loss = (
            im_penalty(o[4], o[0]) + im_penalty(o[4], o[1]) + im_penalty(o[4], o[2]) +
            im_penalty(o[4], o[3]) + im_penalty(o[4], o[5]) +
            im_penalty(o[4], o[6]) + im_penalty(o[4], o[7]) + im_penalty(o[4], o[8])
        )
        # Bivariate refinements
        im_loss += im_penalty(o[1], o[0]) + im_penalty(o[1], o[2])
        im_loss += im_penalty(o[3], o[0]) + im_penalty(o[3], o[6])
        im_loss += im_penalty(o[5], o[2]) + im_penalty(o[5], o[8])
        im_loss += im_penalty(o[7], o[6]) + im_penalty(o[7], o[8])
        
        return commit_loss, nm_loss, im_loss


class DTLGNNeuron(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gate = DTLGNGate()
        # Learnable routing weights for selecting inputs from incoming signals
        self.routing1 = nn.Parameter(torch.randn(in_features) * 0.02)
        self.routing2 = nn.Parameter(torch.randn(in_features) * 0.02)

    def forward(self, inputs, hard=False):
        if hard:
            idx1 = torch.argmax(self.routing1)
            idx2 = torch.argmax(self.routing2)
            val1 = inputs[:, idx1]
            val2 = inputs[:, idx2]
        else:
            p1 = F.softmax(self.routing1, dim=-1)
            p2 = F.softmax(self.routing2, dim=-1)
            val1 = torch.matmul(inputs, p1)
            val2 = torch.matmul(inputs, p2)
            
        return self.gate(val1, val2, hard=hard)

    def compute_losses(self):
        commit, nm, im = self.gate.compute_losses()
        
        # Routing commitment encourages crisp selection
        p1 = F.softmax(self.routing1, dim=-1)
        p2 = F.softmax(self.routing2, dim=-1)
        route_commit = torch.sum((p1**2 - p1)**2) + torch.sum((p2**2 - p2)**2)
        
        return commit, nm, im, route_commit


class DTLGNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.neurons = nn.ModuleList([DTLGNNeuron(in_features) for _ in range(out_features)])

    def forward(self, inputs, hard=False):
        outputs = [neuron(inputs, hard=hard) for neuron in self.neurons]
        return torch.stack(outputs, dim=-1)

    def compute_losses(self):
        losses = [neuron.compute_losses() for neuron in self.neurons]
        commit = sum(x[0] for x in losses)
        nm = sum(x[1] for x in losses)
        im = sum(x[2] for x in losses)
        route_commit = sum(x[3] for x in losses)
        return commit, nm, im, route_commit


class TTRMLoop(nn.Module):
    def __init__(self, d_model, n_concepts, n_tasks, n_rules, n_latent=10, n_layers=2):
        """
        Ternary Tiny Recursive Model (T-TRM) Loop module.
        Updates input context x, steering vector y, and latent state z.
        """
        super().__init__()
        self.d_model = d_model
        self.n_concepts = n_concepts
        self.n_latent = n_latent
        self.n_tasks = n_tasks
        
        # Projections to map continuous activations to concept-like states in [-1, 1]
        self.proj_x = nn.Sequential(
            nn.Linear(d_model, n_concepts),
            nn.Tanh()
        )
        self.proj_y = nn.Sequential(
            nn.Linear(d_model, n_concepts),
            nn.Tanh()
        )
        
        # DTLGN receives concatenated [z_x; z_y; z_prev]
        in_features = 2 * n_concepts + n_latent
        
        self.dtlgn_layers = nn.ModuleList()
        curr_dim = in_features
        for _ in range(n_layers - 1):
            self.dtlgn_layers.append(DTLGNLayer(curr_dim, curr_dim))
        self.dtlgn_layers.append(DTLGNLayer(curr_dim, n_latent))
        
        # Project refined latent state z to CMR-compatible concept probabilities in [0, 1]
        self.proj_cmr_concepts = nn.Sequential(
            nn.Linear(n_latent, n_concepts),
            nn.Sigmoid()
        )
        
        # Project CMR task predictions back to the d_model steering space
        self.proj_steering = nn.Linear(n_tasks, d_model)

    def forward_step(self, z_x, z_y, z_prev, hard=False):
        curr = torch.cat([z_x, z_y, z_prev], dim=-1)
        for layer in self.dtlgn_layers[:-1]:
            curr = layer(curr, hard=hard)
        new_z = self.dtlgn_layers[-1](curr, hard=hard)
        return new_z

    def forward(self, x, cmr_model, T_loops=3, n_steps=2, hard=False):
        # Ensure model parameters are on the same device as input x
        if self.proj_x[0].weight.device != x.device:
            self.to(device=x.device)
            
        # Ensure cmr_model parameters are on the same device and dtype as this model
        model_dtype = self.proj_x[0].weight.dtype
        cmr_params = list(cmr_model.parameters())
        if len(cmr_params) > 0:
            cmr_device = cmr_params[0].device
            cmr_dtype = cmr_params[0].dtype
            if cmr_device != x.device or cmr_dtype != model_dtype:
                cmr_model.to(device=x.device, dtype=model_dtype)
                
        # Handle dtype mismatch by casting input x to match the model's parameters' dtype
        orig_dtype = x.dtype
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)
            
        batch_size = x.size(0)
        device = x.device
        
        z_x = self.proj_x(x)
        
        # Initialize T-TRM states using the correct dtype and device
        z = torch.zeros(batch_size, self.n_latent, dtype=model_dtype, device=device) # Kleene unknown
        y = torch.zeros(batch_size, self.d_model, dtype=model_dtype, device=device)
        
        for t in range(T_loops):
            z_y = self.proj_y(y)
            
            # n-step latent DTLGN recursion
            for _ in range(n_steps):
                z = self.forward_step(z_x, z_y, z, hard=hard)
                
            # Map z to [0, 1] for CMR
            c_pred = self.proj_cmr_concepts(z)
            
            # Retrieve CMR rule polarities & selections
            r = cmr_model.get_all_rule_vars()
            pospolarity = r[:, :, :, 0]
            irrelevance = r[:, :, :, 2]
            relevance = 1 - irrelevance
            
            if cmr_model.selector_input == InputTypes.concepts:
                sel_in = c_pred
            else:
                # Create zeros with matching dtype
                zeros_emb = torch.zeros(batch_size, cmr_model.embedding_size, dtype=model_dtype, device=device)
                c_embs, _ = cmr_model.concept_embedder(
                    zeros_emb, 
                    c=c_pred, 
                    train=False
                )
                sel_in = cmr_model.c_emb_combiner(c_embs.view(batch_size, -1))
                
            logits_s = cmr_model.neural_rule_selector(sel_in).view(-1, cmr_model.n_tasks, cmr_model.effective_n_rules)
            p_s = torch.softmax(logits_s, dim=-1)
            
            batch_c_dup = c_pred.unsqueeze(1).unsqueeze(1).repeat(1, cmr_model.n_tasks, cmr_model.effective_n_rules, 1)
            _pospolarity = pospolarity.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            _relevance = relevance.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            # Symbolic logic evaluation inside CMR
            y_per_rule = cmr_model.rule_module.calc_y(batch_c_dup, _pospolarity, _relevance)
            y_pred = torch.einsum("btr,btr->bt", p_s, y_per_rule)
            
            # Generate refined continuous steering vector
            y = self.proj_steering(y_pred)
            
        # Cast outputs back to original input dtype if needed
        if orig_dtype != model_dtype:
            y = y.to(dtype=orig_dtype)
            z = z.to(dtype=orig_dtype)
            c_pred = c_pred.to(dtype=orig_dtype)
            y_pred = y_pred.to(dtype=orig_dtype)
            
        return y, z, c_pred, y_pred

    def compute_losses(self):
        losses = [layer.compute_losses() for layer in self.dtlgn_layers]
        commit = sum(x[0] for x in losses)
        nm = sum(x[1] for x in losses)
        im = sum(x[2] for x in losses)
        route_commit = sum(x[3] for x in losses)
        return commit, nm, im, route_commit
