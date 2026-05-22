# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OPT backward service (opt-125m) — reference path.

Reconstructs the full logits the forward skipped (vLLM only materializes
last-token logits): logits = final_hidden @ lm_head.weight.T. The saved
``final_hidden`` is already post-final-layer-norm (OPTModel.forward applies
``final_layer_norm`` before returning), so the shared ``_logit_loss_and_grad``
helper consumes it directly. opt-125m ties word embeddings, so the LM-head weight
is the embedding weight shared in ``shared["base"]`` under ``meta["lm_head_key"]``.

Loss = next-token cross-entropy with ``concat_input_ids`` as labels, shifted by 1
within each sample (full-sequence, no prompt masking).
"""

from vllm.deltaserve.bwd_services.base import BackwardService


class OPTBackwardService(BackwardService):
    def compute_loss_and_grad(self, activations: dict, sample_lens: list[int],
                              n: int):
        return self._logit_loss_and_grad(
            activations["final_hidden"][:n],
            activations["concat_input_ids"][:n],
            sample_lens,
        )
