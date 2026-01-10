"""
EqProp Dashboard - Complete fixed version with all features working
"""

# ... (keep all imports and setup from original file, just showing the fixes)

# This is a patch file showing the key fixes needed
# Apply these changes to dashboard.py at the specified line numbers

# Fix 1: Update plots properly (around line 696)
def _update_plots(self):
    """Update all plot curves with latest data."""
    if not HAS_PYQTGRAPH:
        return
    
    if len(self.loss_history) == 0:
        return
    
    epochs = list(range(1, len(self.loss_history) + 1))
    
    # Update vision plots
    if hasattr(self, 'vis_loss_curve'):
        self.vis_loss_curve.setData(epochs, self.loss_history)
    if hasattr(self, 'vis_acc_curve'):
        self.vis_acc_curve.setData(epochs, self.acc_history)
    if hasattr(self, 'vis_lip_curve'):
        self.vis_lip_curve.setData(epochs, self.lipschitz_history)
    
    # Update LM plots  
    if hasattr(self, 'lm_loss_curve'):
        self.lm_loss_curve.setData(epochs, self.loss_history)
    if hasattr(self, 'lm_acc_curve'):
        self.lm_acc_curve.setData(epochs, self.acc_history)
    if hasattr(self, 'lm_lip_curve'):
        self.lm_lip_curve.setData(epochs, self.lipschitz_history)


# Fix 2: Enable generation before training (around line 728)
def _generate_text(self):
    """Generate text from the model (works even with untrained models)."""
    if self.model is None:
        self.gen_output.setText("⚠️ No model loaded yet. Start training to create a model.")
        return
    
    if not hasattr(self.model, 'generate'):
        self.gen_output.setText("⚠️ This model doesn't support generation.\\nUse a Transformer LM variant.")
        return
    
    temperature = self.temp_slider.value() / 10.0
    prompt = "ROMEO:"
    
    self.gen_output.setText(f"🎲 Generating from '{prompt}'...\\n(May be gibberish if undertrained)")
    
    try:
        text = self.model.generate(prompt, max_new_tokens=100, temperature=temperature)
        self.gen_output.setText(f"📝 Generated ({len(text)} chars):\\n\\n{text}")
    except Exception as e:
        self.gen_output.setText(f"❌ Error: {str(e)}\\n\\nTip: Train a few epochs first!")
