import os
import torch
from safetensors.torch import save_file

class ActivationHookExtractor:
    def __init__(self, model, layer_path="model.layers[14]"):
        """
        Manages registration of forward hooks to capture and cache activations from a model layer.
        
        Args:
            model (torch.nn.Module): The PyTorch model to hook.
            layer_path (str): The dot-separated and bracket-indexed path to the layer 
                              (e.g., "model.layers[14]" or "model.model.layers[14]").
        """
        self.model = model
        self.layer_path = layer_path
        self.extracted_activations = []
        self.hook_handle = None
        self._target_module = self._resolve_layer_path(model, layer_path)

    def _resolve_layer_path(self, model, path):
        """Helper to resolve dot-notation and index-notation module paths."""
        curr = model
        try:
            # Split by dots first
            parts = path.split(".")
            for part in parts:
                # Handle bracket notation like layers[14]
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
        """Hook function to capture hidden activations."""
        # Outputs of transformer layers are often tuples (hidden_states, attention_weights, etc.)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Append detached copy of states on CPU to conserve GPU memory
        self.extracted_activations.append(hidden_states.detach().cpu())

    def register(self):
        """Registers the forward hook on the target layer."""
        if self.hook_handle is None:
            self.hook_handle = self._target_module.register_forward_hook(self.hook_fn)
            print(f"Registered forward hook on target module: {self.layer_path}")

    def remove(self):
        """Removes the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("Removed forward hook.")

    def clear(self):
        """Clears previously captured activations from memory."""
        self.extracted_activations.clear()

    def get_concatenated_activations(self, dim=0):
        """Concatenates and returns all captured activations."""
        if not self.extracted_activations:
            raise ValueError("No activations have been captured yet.")
        return torch.cat(self.extracted_activations, dim=dim)

    def save_cache(self, filepath, dim=0):
        """
        Saves all captured activations to a flat .safetensors file.
        
        Args:
            filepath (str): Destination path for the safetensors file.
            dim (int): Dimension along which to concatenate activations (default 0).
        """
        activations = self.get_concatenated_activations(dim=dim)
        
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save using safetensors
        tensors = {"activations": activations}
        save_file(tensors, filepath)
        print(f"Successfully cached {activations.shape} activations to: {filepath}")
        
    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
