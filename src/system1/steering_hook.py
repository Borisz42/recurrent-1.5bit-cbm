import torch

class ActivationSteeringHook:
    def __init__(self, model, layer_path, t_trm, cmr_model, alpha=1.0):
        """
        Manages registration of forward hooks to patch activations using T-TRM steering.
        
        Args:
            model (torch.nn.Module): The PyTorch model to hook.
            layer_path (str): The dot-separated and bracket-indexed path to the layer.
            t_trm (torch.nn.Module): The T-TRM loop model.
            cmr_model (torch.nn.Module): The CMR logic model.
            alpha (float): Steering coefficient (how strongly to apply y).
        """
        self.model = model
        self.layer_path = layer_path
        self.t_trm = t_trm
        self.cmr_model = cmr_model
        self.alpha = alpha
        self.hook_handle = None
        self._target_module = self._resolve_layer_path(model, layer_path)
        
        self.history = []
        self.context_ema = None
        self.ema_decay = 0.9

    def _resolve_layer_path(self, model, path):
        curr = model
        try:
            parts = path.split(".")
            for part in parts:
                if "[" in part and part.endswith("]"):
                    name, index_str = part.split("[")
                    index = int(index_str[:-1])
                    curr = getattr(curr, name)[index]
                else:
                    curr = getattr(curr, part)
            return curr
        except (AttributeError, IndexError, ValueError) as e:
            raise ValueError(f"Could not resolve layer path '{path}' in the model. Error: {e}")

    def hook_fn(self, module, input, output):
        # Outputs of transformer layers are tuples (hidden_states, attention_weights, etc.)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # hidden_states shape: [batch, seq_len, emb_dim]
        seq_len = hidden_states.shape[1]
        
        # 1. Compute Context Representation x
        if seq_len > 1:
            # Prefill phase
            x = hidden_states.mean(dim=1)
            # Initialize EMA context
            self.context_ema = x.detach()
        else:
            # Generation phase
            x_token = hidden_states[:, -1]
            if self.context_ema is not None:
                # Update moving average
                self.context_ema = self.ema_decay * self.context_ema + (1.0 - self.ema_decay) * x_token
                x = self.context_ema
            else:
                x = x_token

        # 2. System 2 Reasoning
        with torch.no_grad():
            y, z, c_pred, y_pred = self.t_trm(x.float(), self.cmr_model, T_loops=3, n_steps=2, hard=True)
            
            # Retrieve CMR rule polarities & selections for visualization
            c_probs = c_pred[0].cpu().numpy()
            y_probs = y_pred[0].cpu().numpy()
            
            self.history.append({
                "c_probs": c_probs,
                "y_probs": y_probs
            })

        # 3. Apply Steering
        # Ensure y matches hidden_states dtype
        y = y.to(dtype=hidden_states.dtype)
        # Broadcast y to [batch, seq_len, emb_dim] and add
        hidden_states = hidden_states + self.alpha * y.unsqueeze(1)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def register(self):
        if self.hook_handle is None:
            self.hook_handle = self._target_module.register_forward_hook(self.hook_fn)
            print(f"Registered steering hook on target module: {self.layer_path}")

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("Removed steering hook.")

    def clear(self):
        self.history.clear()
        self.context_ema = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
