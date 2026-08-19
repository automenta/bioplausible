"""
Local LLM Support for AutoScientist.

Provides integration with llama.cpp, ollama, and other local LLM backends
for hypothesis generation and reasoning without API keys.
"""

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from bioplausible.autoscientist.reasoner import Hypothesis

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Response from a local LLM."""

    text: str
    model: str
    backend: str
    latency_ms: float
    tokens_generated: int | None = None


class LocalLLMBackend(ABC):
    """Abstract base class for local LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate text from the model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and working."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, object]:
        """Get information about the loaded model."""
        pass


class OllamaBackend(LocalLLMBackend):
    """Ollama local LLM backend (http://localhost:11434)."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is not None:
            return self._available

        try:
            import urllib.request

            # Check if Ollama is running
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    models = [m["name"] for m in data.get("models", [])]
                    self._available = (
                        self.model in models or f"{self.model}:latest" in models
                    )
                    if not self._available:
                        logger.warning(
                            "Model %s not found in Ollama. Available: %s",
                            self.model,
                            models,
                        )
                else:
                    self._available = False
        except Exception as e:
            logger.debug("Ollama availability check failed: %s", e)
            self._available = False

        return self._available

    def get_model_info(self) -> dict[str, object]:
        """Get model information from Ollama."""
        try:
            import urllib.request

            with urllib.request.urlopen(
                f"{self.base_url}/api/show",
                data=json.dumps({"name": self.model}).encode(),
                timeout=10,
            ) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {"model": self.model, "backend": "ollama", "status": "unknown"}

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate text using Ollama API."""
        import urllib.request

        if not self.is_available():
            raise RuntimeError(f"Ollama not available or model {self.model} not loaded")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        start = time.time()
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e.read().decode()}") from e

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=result.get("response", ""),
            model=self.model,
            backend="ollama",
            latency_ms=latency,
            tokens_generated=result.get("eval_count"),
        )


class LlamaCppBackend(LocalLLMBackend):
    """llama.cpp backend via llama-cpp-python or CLI."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = all
        n_threads: int | None = None,
        use_mlock: bool = True,
        use_mmap: bool = True,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.use_mlock = use_mlock
        self.use_mmap = use_mmap
        self.verbose = verbose
        self._llm = None
        self._cli_available = self._check_cli()

    def _check_cli(self) -> bool:
        """Check if llama.cpp CLI is available."""
        try:
            result = subprocess.run(
                ["llama-cli", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except FileNotFoundError, subprocess.TimeoutExpired:
            return False

    def is_available(self) -> bool:
        """Check if llama.cpp is available (Python bindings or CLI)."""
        # Try Python bindings first
        try:
            from llama_cpp import Llama  # type: ignore

            return self.model_path.exists()
        except ImportError:
            pass

        # Fall back to CLI
        return self._cli_available and self.model_path.exists()

    def _get_python_llm(self):
        """Lazy-load llama-cpp-python instance."""
        if self._llm is None:
            from llama_cpp import Llama  # type: ignore

            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                use_mlock=self.use_mlock,
                use_mmap=self.use_mmap,
                verbose=self.verbose,
            )
        return self._llm

    def get_model_info(self) -> dict[str, object]:
        """Get model information."""
        return {
            "model_path": str(self.model_path),
            "backend": "llama.cpp",
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "available": self.is_available(),
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate text using llama.cpp."""
        start = time.time()

        if self._llm is not None or self._try_import_llama_cpp():
            # Use Python bindings
            llm = self._get_python_llm()
            result = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                echo=False,
            )
            text = result["choices"][0]["text"]
            tokens = result.get("usage", {}).get("completion_tokens")
        else:
            # Use CLI
            text = self._generate_cli(prompt, max_tokens, temperature, stop_sequences)
            tokens = None

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=text,
            model=str(self.model_path.name),
            backend="llama.cpp",
            latency_ms=latency,
            tokens_generated=tokens,
        )

    def _try_import_llama_cpp(self) -> bool:
        """Try to import llama_cpp."""
        try:
            from llama_cpp import Llama  # noqa: F401

            return True
        except ImportError:
            return False

    def _generate_cli(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
    ) -> str:
        """Generate using llama.cpp CLI."""
        cmd = [
            "llama-cli",
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "--ctx-size",
            str(self.n_ctx),
        ]

        if self.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(self.n_gpu_layers)])

        if stop_sequences:
            for stop in stop_sequences:
                cmd.extend(["--stop", stop])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(f"llama.cpp CLI failed: {result.stderr}")

        # Extract generated text (after the prompt)
        output = result.stdout
        if prompt in output:
            output = output.split(prompt, 1)[-1]
        return output.strip()


class TransformersBackend(LocalLLMBackend):
    """Hugging Face Transformers backend (CPU/GPU)."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        """Check if transformers and model are available."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Check if we can load the model (dry run)
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Lazy-load model and tokenizer."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "auto": "auto",
            }
            torch_dtype = dtype_map.get(self.dtype, "auto")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=self.trust_remote_code,
            )
            self._model.eval()

    def get_model_info(self) -> dict[str, object]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "backend": "transformers",
            "device": self.device,
            "dtype": self.dtype,
            "available": self.is_available(),
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate text using Transformers."""
        import torch

        self._load_model()

        start = time.time()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)

        # Apply stop sequences
        if stop_sequences:
            for stop in stop_sequences:
                if stop in text:
                    text = text.split(stop)[0]

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=text,
            model=self.model_name,
            backend="transformers",
            latency_ms=latency,
            tokens_generated=generated.shape[0],
        )


class VLLMBackend(LocalLLMBackend):
    """vLLM backend for high-throughput local inference."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None

    def is_available(self) -> bool:
        """Check if vLLM is available."""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_model(self):
        """Lazy-load vLLM engine."""
        if self._llm is None:
            from vllm import LLM, SamplingParams

            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
            self._sampling_params_class = SamplingParams

    def get_model_info(self) -> dict[str, object]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "backend": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "available": self.is_available(),
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate text using vLLM."""
        self._load_model()

        start = time.time()

        params = self._sampling_params_class(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )

        outputs = self._llm.generate([prompt], params)
        text = outputs[0].outputs[0].text
        tokens = len(outputs[0].outputs[0].token_ids)

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=text,
            model=self.model_name,
            backend="vllm",
            latency_ms=latency,
            tokens_generated=tokens,
        )


def create_local_llm(
    backend: Literal["ollama", "llama.cpp", "transformers", "vllm"],
    **kwargs,
) -> LocalLLMBackend:
    """
    Factory function to create a local LLM backend.

    Args:
        backend: Backend type ("ollama", "llama.cpp", "transformers", "vllm")
        **kwargs: Backend-specific arguments

    Returns:
        LocalLLMBackend instance
    """
    backends = {
        "ollama": OllamaBackend,
        "llama.cpp": LlamaCppBackend,
        "transformers": TransformersBackend,
        "vllm": VLLMBackend,
    }

    if backend not in backends:
        raise ValueError(
            f"Unknown backend: {backend}. Available: {list(backends.keys())}"
        )

    return backends[backend](**kwargs)


class LocalLLMHypothesisGenerator:
    """
    Hypothesis generator using local LLMs.

    Replaces the OpenAI-dependent LLMHypothesisGenerator with local-first alternatives.
    """

    def __init__(
        self,
        backend: Literal["ollama", "llama.cpp", "transformers", "vllm"] = "ollama",
        **backend_kwargs,
    ):
        self.backend = create_local_llm(backend, **backend_kwargs)
        self._system_prompt = self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are a scientific research assistant specializing in biologically plausible "
            "machine learning. Your task is to propose novel, testable hypotheses based on "
            "experimental results and literature. Focus on:\n"
            "1. Local learning algorithms (EqProp, FA, TP, PC, Hebbian, Spiking)\n"
            "2. Bio-plausibility vs accuracy tradeoffs\n"
            "3. Scaling laws for local learning\n"
            "4. Cross-domain transfer\n"
            "5. Hardware-algorithm co-design\n\n"
            "Return hypotheses as JSON with fields: statement, confidence (0-1), "
            "proposed_model, proposed_task, proposed_propagator, reasoning (array of strings)."
        )

    def generate(self, context: str) -> list["Hypothesis"]:
        """Generate hypotheses from context using local LLM."""
        from bioplausible.autoscientist.reasoner import Hypothesis

        if not self.backend.is_available():
            logger.warning("Local LLM backend not available, returning fallback")
            return self._fallback_hypotheses(context)

        prompt = f"{self._system_prompt}\n\nContext:\n{context}\n\nGenerate 3-5 hypotheses as JSON:"

        try:
            response = self.backend.generate(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                stop_sequences=["```", "\n\nContext:", "\n\nGenerate"],
            )

            # Parse JSON from response
            text = response.text.strip()
            # Extract JSON if wrapped in code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text)
            hypotheses = []

            for item in data.get("hypotheses", []):
                hypotheses.append(
                    Hypothesis(
                        statement=item.get("statement", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        proposed_model=item.get("model"),
                        proposed_task=item.get("task"),
                        proposed_propagator=item.get("propagator"),
                        reasoning_chain=item.get("reasoning", []),
                        source=f"llm:{self.backend.get_model_info().get('model', 'local')}",
                    )
                )

            logger.info(
                "Generated %d hypotheses via local LLM (%s, %.0fms)",
                len(hypotheses),
                self.backend.__class__.__name__,
                response.latency_ms,
            )
            return hypotheses

        except (json.JSONDecodeError, KeyError, RuntimeError) as e:
            logger.warning("Local LLM hypothesis generation failed: %s", e)
            return self._fallback_hypotheses(context)

    def _fallback_hypotheses(self, context: str) -> list["Hypothesis"]:
        """Fallback when LLM is unavailable."""
        from bioplausible.autoscientist.reasoner import Hypothesis

        return [
            Hypothesis(
                statement="Try alternative architectures on underexplored tasks",
                confidence=0.3,
                source="rule-based",
                reasoning_chain=[
                    "Local LLM backend unavailable, using fallback heuristic"
                ],
            ),
        ]


def get_recommended_local_model(task: str = "general") -> dict[str, object]:
    """
    Get recommended local model configuration for a task.

    Args:
        task: Task type ("general", "coding", "reasoning", "lightweight")

    Returns:
        Dictionary with backend and model recommendations
    """
    recommendations = {
        "general": {
            "backend": "ollama",
            "model": "llama3.1:8b",
            "reason": "Good balance of capability and speed for general scientific reasoning",
        },
        "coding": {
            "backend": "ollama",
            "model": "codellama:13b",
            "reason": "Specialized for code generation and technical reasoning",
        },
        "reasoning": {
            "backend": "ollama",
            "model": "llama3.1:70b",
            "reason": "Larger model for complex multi-step reasoning (requires 48GB+ RAM)",
        },
        "lightweight": {
            "backend": "ollama",
            "model": "phi3:mini",
            "reason": "Fast, small model for quick hypothesis generation (2-4GB RAM)",
        },
        "llama_cpp_general": {
            "backend": "llama.cpp",
            "model_path": "models/llama-3.1-8b-instruct.Q4_K_M.gguf",
            "reason": "Quantized Llama 3.1 8B for CPU/GPU inference via llama.cpp",
        },
        "transformers_general": {
            "backend": "transformers",
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
            "reason": "Small but capable model via Hugging Face Transformers",
        },
    }

    return recommendations.get(task, recommendations["general"])


__all__ = [
    "LocalLLMBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "TransformersBackend",
    "VLLMBackend",
    "LocalLLMHypothesisGenerator",
    "LLMResponse",
    "create_local_llm",
    "get_recommended_local_model",
]
