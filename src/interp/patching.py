"""
Activation patching: the core causal-intervention tool of mech interp.

Idea: run the model on a "clean" input and a "corrupted" input, cache
activations from one run, then splice a specific cached activation
(e.g. one attention head's output, one MLP neuron layer) into the other
run's forward pass. If swapping in that one activation flips the output
back towards the clean answer, that component is causally implicated in
the computation -- this is how you find *which* head/neuron matters,
not just which one merely correlates with the answer.
"""
from contextlib import contextmanager
import torch


class PatchableTransformer:
    """Thin wrapper that lets you run a forward pass with a specific
    internal activation overridden (patched) by a value captured from
    another run. Works by monkey-patching the relevant submodule's
    forward method for the duration of one call.
    """

    def __init__(self, model):
        self.model = model

    def run_with_cache(self, tokens):
        logits, cache = self.model(tokens, return_cache=True)
        return logits, cache

    @contextmanager
    def _patch_mlp_neuron(self, layer: int, neuron: int, value: torch.Tensor):
        block = self.model.blocks[layer]
        orig_forward = block.mlp.forward

        def patched_forward(x):
            pre = x @ block.mlp.W_in + block.mlp.b_in
            post = torch.relu(pre)
            post[..., neuron] = value  # override this neuron's activation
            block.mlp.pre_act = pre.detach()
            block.mlp.post_act = post.detach()
            return post @ block.mlp.W_out + block.mlp.b_out

        block.mlp.forward = patched_forward
        try:
            yield
        finally:
            block.mlp.forward = orig_forward

    def patch_mlp_neuron_and_run(self, tokens, layer, neuron, patch_value):
        """Runs `tokens` through the model with MLP neuron `neuron` in
        `layer` forcibly set to `patch_value` (a tensor broadcastable to
        [batch, seq]). Returns the resulting logits."""
        with self._patch_mlp_neuron(layer, neuron, patch_value):
            logits = self.model(tokens)
        return logits


def logit_diff(logits, correct_labels, incorrect_labels):
    """A standard patching metric: logit(correct) - logit(incorrect),
    at the final sequence position. Larger = model more confidently
    correct."""
    final_logits = logits[:, -1, :]
    correct = final_logits.gather(1, correct_labels.unsqueeze(1)).squeeze(1)
    incorrect = final_logits.gather(1, incorrect_labels.unsqueeze(1)).squeeze(1)
    return (correct - incorrect).mean().item()


def neuron_ablation_effect(model, inputs, labels, layer: int, neuron: int, device="cpu"):
    """Zero-ablates one MLP neuron and measures the drop in accuracy.
    Cheap, simple causal test: if killing this neuron tanks accuracy,
    it's load-bearing for the circuit; if accuracy barely moves, it's
    not part of the core computation (or the circuit is redundant)."""
    patchable = PatchableTransformer(model)

    with torch.no_grad():
        base_logits = model(inputs)
        base_preds = base_logits[:, -1, :].argmax(dim=-1)
        base_acc = (base_preds == labels).float().mean().item()

        zero_val = torch.zeros(inputs.shape[0], inputs.shape[1], device=device)
        patched_logits = patchable.patch_mlp_neuron_and_run(inputs, layer, neuron, zero_val)
        patched_preds = patched_logits[:, -1, :].argmax(dim=-1)
        patched_acc = (patched_preds == labels).float().mean().item()

    return {"base_acc": base_acc, "ablated_acc": patched_acc, "acc_drop": base_acc - patched_acc}
