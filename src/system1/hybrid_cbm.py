import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONCEPTS = [
    "code generation",
    "mathematical reasoning",
    "information extraction",
    "text summarization",
    "creative narrative writing",
    "linguistic translation",
    "factual recall qa",
    "roleplay persona simulation",
    "linguistic classification"
]

class HybridCBM(nn.Module):
    def __init__(self, n_dynamic=10, emb_dim=None, clip_dim=512, concepts=None, clip_embeddings=None):
        """
        Hybrid Concept Bottleneck Model (HybridCBM) layer.
        
        Args:
            n_dynamic (int): Number of dynamic (learnable) concepts to extract.
            emb_dim (int, optional): Dimension of input activations x. If None, it will be
                                     inferred dynamically on the first forward pass.
            clip_dim (int): Dimensionality of the CLIP embedding space.
            concepts (list of str, optional): List of static concept labels. If None, uses default list.
            clip_embeddings (torch.Tensor, optional): Precomputed CLIP text embeddings of shape (n_static, clip_dim).
                                                      If None, random embeddings are initialized.
        """
        super().__init__()
        self.n_dynamic = n_dynamic
        self.clip_dim = clip_dim
        
        # Determine static concepts
        self.concepts = concepts if concepts is not None else DEFAULT_CONCEPTS
        self.n_static = len(self.concepts)
        
        # Check or initialize CLIP embeddings
        if clip_embeddings is not None:
            assert clip_embeddings.shape == (self.n_static, clip_dim), \
                f"clip_embeddings shape {clip_embeddings.shape} must match (n_static={self.n_static}, clip_dim={clip_dim})"
            self.register_buffer("static_embeddings", clip_embeddings)
        else:
            # Fallback to normalized random embeddings if no pre-computed CLIP embeddings are provided
            print("No precomputed CLIP embeddings provided. Initializing random normalized vectors for static concepts.")
            random_embeds = torch.randn(self.n_static, clip_dim)
            random_embeds = F.normalize(random_embeds, p=2, dim=1)
            self.register_buffer("static_embeddings", random_embeds)
            
        self.initialized = False
        self.emb_dim = emb_dim
        
        if emb_dim is not None:
            self._init_layers(emb_dim)

    def _init_layers(self, emb_dim):
        """Initializes projection and reconstruction layers."""
        self.emb_dim = emb_dim
        
        # Stable LayerNorm to prevent scale issues with large activations
        self.input_norm = nn.LayerNorm(emb_dim)
        
        # Project base activation x to CLIP space to compute static concept activations
        self.proj_clip = nn.Linear(emb_dim, self.clip_dim, bias=False)
        
        # Learnable projection from base activation x to dynamic concept bottleneck space
        self.proj_dynamic = nn.Linear(emb_dim, self.n_dynamic, bias=False)
        
        # Reconstruction decoder mapping bottleneck z back to original activation space
        self.decoder = nn.Linear(self.n_static + self.n_dynamic, emb_dim, bias=False)
        
        self.initialized = True

    def forward(self, x):
        """
        Forward pass projecting input activations to the hybrid concept bottleneck.
        
        Args:
            x (torch.Tensor): Model activations of shape (batch_size, sequence_length, emb_dim) 
                               or (batch_size, emb_dim).
                               
        Returns:
            z (torch.Tensor): Hybrid concept bottleneck values of shape (..., n_static + n_dynamic).
            x_rec (torch.Tensor): Reconstructed activation tensor of shape (..., emb_dim).
            rec_loss (torch.Tensor): Reconstruction loss (MSE) between x and x_rec.
        """
        # Dynamic shape inference if not pre-initialized
        if not self.initialized:
            input_dim = x.shape[-1]
            print(f"Dynamically inferring input activation dimension: {input_dim}")
            self._init_layers(input_dim)
            # Make sure we move the lazily created modules to the appropriate device
            self.to(device=x.device)

        # Ensure correct device of parameters
        if self.proj_clip.weight.device != x.device:
            self.to(device=x.device)

        # Handle dtype mismatch by casting input x to match the model's parameters' dtype
        orig_dtype = x.dtype
        model_dtype = self.proj_clip.weight.dtype
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)

        # Normalize the inputs using LayerNorm to prevent overflow/saturation
        x_norm = self.input_norm(x)

        # 1. Project activations x_norm to CLIP space and compute static concepts
        x_clip = self.proj_clip(x_norm)  # (..., clip_dim)
        
        # Compute cosine similarity with the concept embeddings
        # static_embeddings: (n_static, clip_dim)
        x_clip_normalized = F.normalize(x_clip, p=2, dim=-1)
        # static_embeddings is already normalized, but let's double check
        static_embeds_normalized = F.normalize(self.static_embeddings.to(device=x.device, dtype=x.dtype), p=2, dim=-1)
        
        # z_static: (..., n_static)
        z_static = torch.matmul(x_clip_normalized, static_embeds_normalized.T)
        
        # 2. Compute dynamic concepts
        z_dynamic = torch.tanh(self.proj_dynamic(x_norm))  # (..., n_dynamic)
        
        # 3. Combine to form the full concept bottleneck
        z = torch.cat([z_static, z_dynamic], dim=-1)  # (..., n_static + n_dynamic)
        
        # 4. Reconstruct original activations for representation decomposition loss
        x_rec = self.decoder(z)  # (..., emb_dim)
        
        # 5. Compute MSE reconstruction loss against the normalized input
        rec_loss = F.mse_loss(x_norm, x_rec)
        
        # Cast outputs back to original input dtype if needed
        if orig_dtype != model_dtype:
            z = z.to(dtype=orig_dtype)
            x_rec = x_rec.to(dtype=orig_dtype)
            
        return z, x_rec, rec_loss

    def compute_clip_embeddings(self, tokenizer=None, text_model=None):
        """
        Helper method to compute CLIP text embeddings from static concept strings.
        If a tokenizer and text model (e.g. CLIPTextModel) are provided, it populates
        the buffer.
        """
        if tokenizer is None or text_model is None:
            raise ValueError("Both tokenizer and text_model must be provided to compute embeddings.")
            
        text_model.eval()
        device = next(text_model.parameters()).device
        with torch.no_grad():
            inputs = tokenizer(self.concepts, padding=True, return_tensors="pt").to(device)
            outputs = text_model(**inputs)
            # Use pooler_output or projection of the last hidden state
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs[0][:, 0, :]
                
            embeddings = F.normalize(embeddings, p=2, dim=1)
            self.static_embeddings.copy_(embeddings.to(self.static_embeddings.device))
        print("Updated static concept embeddings using the provided text model.")

    def translate_dynamic_concepts(self, candidate_embeddings, candidate_labels):
        """
        Translates learned dynamic concept vectors back into human-understandable labels by projecting
        them to the CLIP space and finding the closest candidate text concept via cosine similarity.
        
        Args:
            candidate_embeddings (torch.Tensor): Precomputed CLIP embeddings of candidate concepts,
                                                 shape (n_candidates, clip_dim).
            candidate_labels (list of str): Human-understandable names for the candidate concepts.
            
        Returns:
            translated_labels (list of str): Assigned labels for each dynamic concept.
            similarities (torch.Tensor): Cosine similarity scores for the chosen labels, shape (n_dynamic,).
        """
        device = self.decoder.weight.device
        candidate_embeddings = candidate_embeddings.to(device=device, dtype=self.decoder.weight.dtype)
        
        # Extract dynamic concept vectors from the decoder weight matrix
        # self.decoder.weight shape is (emb_dim, n_static + n_dynamic)
        # Column j represents concept j
        dynamic_vectors = self.decoder.weight[:, self.n_static:].T  # (n_dynamic, emb_dim)
        
        # Project dynamic concept vectors to CLIP space
        projected_clip = self.proj_clip(dynamic_vectors)  # (n_dynamic, clip_dim)
        
        # Normalize vectors for cosine similarity
        projected_clip_norm = F.normalize(projected_clip, p=2, dim=-1)
        candidate_embeds_norm = F.normalize(candidate_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity
        # sim_matrix: (n_dynamic, n_candidates)
        sim_matrix = torch.matmul(projected_clip_norm, candidate_embeds_norm.T)
        
        # Find closest match for each dynamic concept
        max_sims, max_indices = torch.max(sim_matrix, dim=-1)
        
        translated_labels = [candidate_labels[idx.item()] for idx in max_indices]
        return translated_labels, max_sims

