import unittest
import torch
import argparse
from unittest.mock import MagicMock
from src.eval.chatbot_app import SteeredChatbot
from src.system1.steering_hook import ActivationSteeringHook

class TestActiveSteering(unittest.TestCase):
    def setUp(self):
        # Mock arguments to prevent loading massive LLMs during unit testing
        self.args = argparse.Namespace(
            model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
            adapter_dir="",
            hybrid_cbm_path="./hybrid_cbm.pt",
            model_dir="./t_trm_outputs",
            n_dynamic=5,
            clip_dim=512,
            n_latent=8,
            n_rules=10,
            layer_index=14,
            max_tokens=10,
            load_in_4bit=True,
            concepts_type="general",
            steering_alpha=1.0
        )
        
    def test_hook_initialization(self):
        # Create dummy models
        model = MagicMock()
        t_trm = MagicMock()
        cmr_model = MagicMock()
        
        # Test hook creation
        hook = ActivationSteeringHook(model, "layer_path", t_trm, cmr_model, alpha=0.5)
        self.assertEqual(hook.alpha, 0.5)
        self.assertIsNone(hook.context_ema)
        self.assertEqual(len(hook.history), 0)
        
    def test_hook_patching_logic(self):
        # Mock inputs
        dummy_hidden_states = torch.ones(1, 1, 128) # seq_len = 1 (generation step)
        
        # Create mock T-TRM that returns a known steering vector
        mock_y = torch.full((1, 128), 2.0)
        mock_t_trm = MagicMock(return_value=(
            mock_y,                           # y
            torch.zeros(1, 8),                # z
            torch.ones(1, 1, 5),              # c_pred
            torch.ones(1, 1, 2)               # y_pred
        ))
        
        hook = ActivationSteeringHook(MagicMock(), "layer", mock_t_trm, MagicMock(), alpha=1.0)
        
        # Call hook_fn manually
        patched_output = hook.hook_fn(MagicMock(), None, dummy_hidden_states)
        
        # Verify the steering vector was applied: hidden_states (1.0) + alpha (1.0) * y (2.0) = 3.0
        self.assertTrue(torch.allclose(patched_output, torch.full((1, 1, 128), 3.0)))
        
        # Verify history was recorded
        self.assertEqual(len(hook.history), 1)

if __name__ == '__main__':
    unittest.main()
