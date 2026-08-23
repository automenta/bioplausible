"""Sprint 1.4 — Deterministic GPU seeding + ``biopl-repro-check`` CLI.

Covers the new :func:`seed_everything` / :func:`capture_environment` /
:func:`deps_hash` gate helpers and the ``biopl-repro-check`` console script.
Any regression in seeding (an unseeded RNG source leaking in) that would make a
same-seed run non-bitwise-identical is caught here.

These are CPU-gated unit tests; the full bitwise reproducibility sweep itself
lives behind the CLI (heavy, per-family, and already exercised in CI).
"""

import json

import pytest
import torch

from bioplausible.cli.repro import main as repro_main
from bioplausible.utils import capture_environment, deps_hash, seed_everything


class TestSeedEverything:
    def test_returns_environment_fingerprint(self):
        env = seed_everything(7, device="cpu")
        assert "git_commit" in env
        assert "torch_version" in env
        assert "cuda_version" in env
        assert "python_version" in env

    def test_same_seed_same_parameters(self):
        seed_everything(42, device="cpu")
        w1 = torch.randn(5, 5)
        seed_everything(42, device="cpu")
        w2 = torch.randn(5, 5)
        assert torch.equal(w1, w2)

    def test_seeds_all_rng_sources(self):
        import random as pyrandom

        import numpy as np

        seed_everything(99, device="cpu")
        py_1 = pyrandom.random()
        np_1 = np.random.rand()
        th_1 = torch.rand(1).item()
        seed_everything(99, device="cpu")
        py_2 = pyrandom.random()
        np_2 = np.random.rand()
        th_2 = torch.rand(1).item()
        assert py_1 == py_2
        assert np_1 == np_2
        assert th_1 == th_2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_device_seeds_cudnn(self):
        env = seed_everything(5, device="cuda")
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert env["cuda_version"] != "n/a"

    def test_cuda_request_without_gpu_raises(self, monkeypatch):
        if torch.cuda.is_available():
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError):
            seed_everything(1, device="cuda")


class TestCaptureEnvironment:
    def test_deterministic_and_complete(self):
        e1 = capture_environment()
        e2 = capture_environment()
        assert e1 == e2
        assert set(e1) == {
            "git_commit",
            "torch_version",
            "cuda_version",
            "python_version",
        }

    def test_deps_hash_stable(self):
        assert deps_hash() == deps_hash()
        assert len(deps_hash()) == 12


class TestReproCheckCLI:
    def test_empty_models_exits_2(self):
        assert repro_main(["--models", " "]) == 2

    def test_json_report_all_pass(self, capsys):
        code = repro_main(["--seed", "7", "--device", "cpu", "--json"])
        assert code == 0
        out = capsys.readouterr().out
        report = json.loads(out)
        assert report["seed"] == 7
        assert report["device"] == "cpu"
        assert all(report["results"].values())
        assert report["exit_code"] == 0
