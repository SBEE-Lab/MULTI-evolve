"""Real IF1 integration regression; requires cached weights and FASTA/CIF inputs.

Run with IF1_FASTA, IF1_STRUCTURE, and IF1_EXPECT_DEVICE=cpu or cuda.
Use CUDA_VISIBLE_DEVICES="" for the CPU-only fallback/error checks.
"""

import inspect
import os
import unittest

import esm
import numpy as np
import torch
from Bio import SeqIO
from esm.inverse_folding.util import CoordBatchConverter
from scipy.stats import spearmanr
from torch.nn.modules.module import register_module_forward_pre_hook

from multievolve.utils.zeroshot_utils import zero_shot_esm_if, zero_shot_esm_if_dms

MODEL = "esm_if1_gvp4_t16_142M_UR50"


class IF1DeviceIntegration(unittest.TestCase):
    def test_real_model_devices_and_scores(self):
        torch.set_num_threads(4)
        sequence = str(SeqIO.read(os.environ["IF1_FASTA"], "fasta").seq)
        structure_path = os.environ["IF1_STRUCTURE"]
        expected = os.environ["IF1_EXPECT_DEVICE"]
        self.assertEqual(torch.cuda.is_available(), expected == "cuda")
        observed = set()

        def observe(module, inputs):
            if module.__class__.__name__ == "GVPEncoder":
                device = next(module.parameters()).device.type
                observed.add(device)
                for value in inputs:
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.device.type, device)
                self.assertTrue(torch.is_inference_mode_enabled())

        # Observe real encoder execution, without replacing model or its outputs.
        handle = register_module_forward_pre_hook(observe)
        try:
            auto = zero_shot_esm_if_dms(sequence, structure_path)
            self.assertEqual(observed, {expected})
            for function in (zero_shot_esm_if_dms, zero_shot_esm_if):
                self.assertEqual(
                    inspect.signature(function).parameters["esm_if_device"].default,
                    "auto",
                )
            observed.clear()
            cpu = zero_shot_esm_if_dms(sequence, structure_path, esm_if_device="cpu")
            self.assertEqual(observed, {"cpu"})
            mutations = [[m] for m in cpu.mutations]
            mutations += [[cpu.mutations.iloc[0], cpu.mutations.iloc[-1]]]
            observed.clear()
            auto_raw = zero_shot_esm_if(
                mutations, [MODEL], sequence, structure_path, "A"
            )
            self.assertEqual(observed, {expected})
            observed.clear()
            cpu_raw = zero_shot_esm_if(
                mutations, [MODEL], sequence, structure_path, "A", esm_if_device="cpu"
            )
            self.assertEqual(observed, {"cpu"})
            if expected == "cuda":
                observed.clear()
                explicit = zero_shot_esm_if_dms(
                    sequence, structure_path, esm_if_device="cuda"
                )
                explicit_raw = zero_shot_esm_if(
                    mutations,
                    [MODEL],
                    sequence,
                    structure_path,
                    "A",
                    esm_if_device="cuda",
                )
                self.assertEqual(observed, {"cuda"})
                np.testing.assert_allclose(
                    explicit.logratio, auto.logratio, atol=2e-4, rtol=2e-4
                )
                np.testing.assert_allclose(explicit_raw, auto_raw, atol=2e-4, rtol=2e-4)
        finally:
            handle.remove()

        # Independent CPU forward reproduces the original raw-logit semantics.
        model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL)
        model.eval()
        structure = esm.inverse_folding.util.load_structure(structure_path, "A")
        coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
        coords, confidence, _, tokens, padding = CoordBatchConverter(alphabet)(
            [(coords, None, sequence)], device="cpu"
        )
        with torch.no_grad():
            logits, _ = model.forward(coords, padding, confidence, tokens[:, :-1])
        scores = logits.numpy()[0]

        def raw(mutation):
            return scores[alphabet.tok_to_idx[mutation[-1]], int(mutation[1:-1]) - 1]

        reference_dms = np.array([raw(m) - raw(m[:-1] + m[0]) for m in cpu.mutations])
        reference_raw = np.array([np.mean([raw(m) for m in ms]) for ms in mutations])
        np.testing.assert_allclose(cpu.logratio, reference_dms, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cpu_raw, reference_raw, atol=1e-5, rtol=1e-5)
        self.assertEqual(auto.mutations.tolist(), cpu.mutations.tolist())
        self.assertEqual(len(auto), 19 * len(sequence))
        for label, left, right in (
            ("dms", auto.logratio.to_numpy(), cpu.logratio.to_numpy()),
            ("raw", auto_raw, cpu_raw),
        ):
            # Backend reductions need an absolute float32 logit tolerance;
            # rank correlation and exact top-24 membership guard nomination use.
            tolerance = 5e-3 if expected == "cuda" else 1e-5
            np.testing.assert_allclose(left, right, atol=tolerance, rtol=0)
            self.assertLess(float(np.mean(np.abs(left - right))), 5e-4)
            rho = spearmanr(left, right).statistic
            self.assertGreater(rho, 0.99999)
            self.assertEqual(set(np.argsort(left)[-24:]), set(np.argsort(right)[-24:]))
            print(
                f"{label}: max_abs_error={np.max(np.abs(left - right)):.8g}, rho={rho:.12g}"
            )
        for function, args in (
            (zero_shot_esm_if_dms, (sequence, structure_path)),
            (zero_shot_esm_if, (mutations, [MODEL], sequence, structure_path, "A")),
        ):
            with self.assertRaisesRegex(ValueError, "esm_if_device"):
                function(*args, esm_if_device="invalid")
            if expected == "cpu":
                with self.assertRaisesRegex(RuntimeError, "CUDA.*not available"):
                    function(*args, esm_if_device="cuda")


if __name__ == "__main__":
    unittest.main()
