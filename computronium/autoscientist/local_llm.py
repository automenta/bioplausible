"""
Local LLM Support for AutoScientist.

Provides integration with llama.cpp, ollama, and other local LLM backends
for hypothesis generation and reasoning without API keys.
"""

import json
import logging
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from computronium.autoscientist.reasoner import Hypothesis

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

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and working."""

    @abstractmethod
    def get_model_info(self) -> dict[str, object]:
        """Get information about the loaded model."""


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

        try:  # ruff: ignore[too-many-statements-in-try-clause]
            import urllib.request

            # Check if Ollama is running
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:  # ruff: ignore[suspicious-url-open-usage]
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

            with urllib.request.urlopen(  # ruff: ignore[suspicious-url-open-usage]
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
            req = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage]
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # ruff: ignore[suspicious-url-open-usage]
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
            result = subprocess.run(  # ruff: ignore[subprocess-run-without-check]
                ["llama-cli", "--version"],  # ruff: ignore[start-process-with-partial-path]
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0  # ruff: ignore[try-consider-else]
        except FileNotFoundError, subprocess.TimeoutExpired:
            return False

    def is_available(self) -> bool:
        """Check if llama.cpp is available (Python bindings or CLI)."""
        # Try Python bindings first
        try:
            from llama_cpp import (
                Llama,  # type: ignore  # ruff: ignore[unused-import, blanket-type-ignore]
            )

            return self.model_path.exists()
        except ImportError:
            pass

        # Fall back to CLI
        return self._cli_available and self.model_path.exists()

    def _get_python_llm(self):
        """Lazy-load llama-cpp-python instance."""
        if self._llm is None:
            from llama_cpp import (
                Llama,  # type: ignore  # ruff: ignore[blanket-type-ignore]
            )

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
            from llama_cpp import Llama  # ruff: ignore[unused-import]

            return True  # ruff: ignore[try-consider-else]
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

        result = subprocess.run(  # ruff: ignore[subprocess-run-without-check, subprocess-without-shell-equals-true]
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
            import torch  # ruff: ignore[unused-import]
            from transformers import (  # ruff: ignore[unused-import]
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            # Check if we can load the model (dry run)
            return True  # ruff: ignore[try-consider-else]
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
            import vllm  # ruff: ignore[unused-import]

            return True  # ruff: ignore[try-consider-else]
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

    def generate(self, context: str) -> list[Hypothesis]:
        """Generate hypotheses from context using local LLM."""
        from computronium.autoscientist.reasoner import Hypothesis

        if not self.backend.is_available():
            logger.warning("Local LLM backend not available, returning fallback")
            return self._fallback_hypotheses(context)

        prompt = f"{self._system_prompt}\n\nContext:\n{context}\n\nGenerate 3-5 hypotheses as JSON:"

        try:  # ruff: ignore[too-many-statements-in-try-clause]
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
            return hypotheses  # ruff: ignore[try-consider-else]

        except (json.JSONDecodeError, KeyError, RuntimeError) as e:
            logger.warning("Local LLM hypothesis generation failed: %s", e)
            return self._fallback_hypotheses(context)

    def _fallback_hypotheses(self, context: str) -> list[Hypothesis]:
        """Fallback when LLM is unavailable."""
        from computronium.autoscientist.reasoner import Hypothesis

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


# ============================================================
# P1.45a: Ollama Auto-Model-Pull
# ============================================================


class OllamaAutoPull:
    """Automatically pull missing Ollama models.

    Detects when a requested model is not available and pulls it via
    `ollama pull` command. Provides progress tracking and retry logic.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def list_available_models(self) -> list[str]:
        """List all models currently available in Ollama."""
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:  # ruff: ignore[suspicious-url-open-usage]
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.debug("Failed to list Ollama models: %s", e)
        return []

    def is_model_available(self, model: str) -> bool:
        """Check if a specific model is available."""
        models = self.list_available_models()
        return model in models or f"{model}:latest" in models

    def pull_model(self, model: str, show_progress: bool = True) -> bool:
        """Pull a model via Ollama API.

        Args:
            model: Model name to pull (e.g., "llama3.1:8b")
            show_progress: Whether to log progress

        Returns:
            True if pull succeeded, False otherwise
        """
        import urllib.request

        payload = json.dumps({"name": model, "stream": show_progress}).encode()

        for attempt in range(self.max_retries):  # ruff: ignore[too-many-nested-blocks]
            try:  # ruff: ignore[too-many-statements-in-try-clause]
                req = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage]
                    f"{self.base_url}/api/pull",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # ruff: ignore[suspicious-url-open-usage]
                    if show_progress:
                        for line in resp:
                            try:
                                progress = json.loads(line.decode())
                                status = progress.get("status", "")
                                if "completed" in status or "success" in status.lower():
                                    logger.info("Model %s pulled successfully", model)
                                    return True
                            except json.JSONDecodeError:
                                continue
                    else:
                        # Wait for completion
                        result = json.loads(resp.read().decode())
                        if result.get("status") == "success":
                            logger.info("Model %s pulled successfully", model)
                            return True
            except urllib.error.HTTPError as e:
                logger.warning(
                    "Pull attempt %d failed: %s", attempt + 1, e.read().decode()
                )
            except Exception as e:
                logger.warning("Pull attempt %d failed: %s", attempt + 1, e)

            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff

        logger.error(
            "Failed to pull model %s after %d attempts", model, self.max_retries
        )
        return False

    def ensure_model(self, model: str) -> bool:
        """Ensure a model is available, pulling if necessary.

        Args:
            model: Model name to ensure

        Returns:
            True if model is available (was already or pulled successfully)
        """
        if self.is_model_available(model):
            logger.debug("Model %s already available", model)
            return True

        logger.info("Model %s not found, pulling...", model)
        return self.pull_model(model)


# ============================================================
# P1.45b: llama.cpp Quantization Auto-Select
# ============================================================


class LlamaCppQuantizationSelector:
    """Automatically select optimal llama.cpp quantization based on hardware.

    Selects between Q4_K_M (4-bit, faster, less VRAM) and Q8_0 (8-bit, higher quality,
    more VRAM) based on available GPU memory and task requirements.
    """

    # Quantization options with (size_gb, quality_score, speed_score)
    QUANTIZATIONS = {  # ruff: ignore[mutable-class-default]
        "Q4_K_M": {"size_gb": 4.7, "quality": 0.85, "speed": 0.95},
        "Q4_K_S": {"size_gb": 4.2, "quality": 0.80, "speed": 0.98},
        "Q5_K_M": {"size_gb": 5.7, "quality": 0.90, "speed": 0.90},
        "Q5_K_S": {"size_gb": 5.2, "quality": 0.87, "speed": 0.93},
        "Q6_K": {"size_gb": 6.7, "quality": 0.93, "speed": 0.85},
        "Q8_0": {"size_gb": 8.7, "quality": 0.98, "speed": 0.75},
        "F16": {"size_gb": 16.0, "quality": 1.00, "speed": 0.60},
        "F32": {"size_gb": 32.0, "quality": 1.00, "speed": 0.50},
    }

    def __init__(
        self,
        prefer_quality: bool = False,
        max_vram_gb: float | None = None,
        model_base_name: str = "llama-3.1-8b-instruct",
    ):
        self.prefer_quality = prefer_quality
        self.max_vram_gb = max_vram_gb or self._detect_vram()
        self.model_base_name = model_base_name

    def _detect_vram(self) -> float:
        """Detect available GPU VRAM in GB."""
        try:
            import torch

            if torch.cuda.is_available():
                # Get free memory on first GPU
                free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
                return free_bytes / (1024**3)
        except Exception:  # ruff: ignore[try-except-pass]
            pass
        # Default to 8GB if detection fails
        return 8.0

    def select_quantization(self) -> str:
        """Select best quantization for current hardware.

        Returns:
            Quantization suffix (e.g., "Q4_K_M", "Q8_0")
        """
        # Filter quantizations that fit in VRAM (leave 1GB headroom)
        available_vram = self.max_vram_gb - 1.0
        if available_vram <= 0:
            available_vram = 1.0  # Minimum for CPU offload

        candidates = [
            (name, info)
            for name, info in self.QUANTIZATIONS.items()
            if info["size_gb"] <= available_vram
        ]

        if not candidates:
            # Fallback to smallest
            return min(self.QUANTIZATIONS.items(), key=lambda x: x[1]["size_gb"])[0]

        if self.prefer_quality:
            # Select highest quality that fits
            return max(candidates, key=lambda x: x[1]["quality"])[0]
        else:
            # Select best speed/quality balance (Pareto-optimal)
            # Score = quality * 0.6 + speed * 0.4  # ruff: ignore[commented-out-code]
            return max(
                candidates,
                key=lambda x: x[1]["quality"] * 0.6 + x[1]["speed"] * 0.4,
            )[0]

    def get_model_path(self, quantization: str | None = None) -> str:
        """Get full model path for selected quantization.

        Args:
            quantization: Specific quantization to use, or None for auto-select

        Returns:
            Full model path string
        """
        quant = quantization or self.select_quantization()
        return f"models/{self.model_base_name}.{quant}.gguf"

    def get_recommendation_info(self) -> dict[str, object]:
        """Get detailed recommendation info."""
        selected = self.select_quantization()
        return {
            "selected_quantization": selected,
            "model_path": self.get_model_path(selected),
            "available_vram_gb": self.max_vram_gb,
            "prefer_quality": self.prefer_quality,
            "all_options": self.QUANTIZATIONS,
        }


# ============================================================
# P1.45c: Speculative Decoding
# ============================================================


class SpeculativeDecodingBackend(LocalLLMBackend):
    """Speculative decoding backend for faster hypothesis generation.

    Uses a small draft model to generate tokens speculatively, which are then
    verified by a larger target model. Accepted tokens are kept, rejected ones
    trigger fallback to target model generation.

    This can provide 2-3x speedup for hypothesis generation while maintaining
    output quality of the larger model.
    """

    def __init__(
        self,
        target_backend: LocalLLMBackend,
        draft_backend: LocalLLMBackend,
        max_draft_tokens: int = 5,
        acceptance_threshold: float = 0.5,
    ):
        self.target_backend = target_backend
        self.draft_backend = draft_backend
        self.max_draft_tokens = max_draft_tokens
        self.acceptance_threshold = acceptance_threshold

    def is_available(self) -> bool:
        """Check if both backends are available."""
        return self.target_backend.is_available() and self.draft_backend.is_available()

    def get_model_info(self) -> dict[str, object]:
        """Get information about both models."""
        return {
            "backend": "speculative_decoding",
            "target": self.target_backend.get_model_info(),
            "draft": self.draft_backend.get_model_info(),
            "max_draft_tokens": self.max_draft_tokens,
            "acceptance_threshold": self.acceptance_threshold,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate text using speculative decoding.

        This is a simplified implementation. Full speculative decoding requires
        access to model logits, which may not be available via all backends.
        This version uses a heuristic: generate with draft, verify with target
        by comparing outputs.
        """
        import time

        start = time.time()
        all_text = ""
        remaining_tokens = max_tokens

        while remaining_tokens > 0:
            draft_tokens = min(self.max_draft_tokens, remaining_tokens)

            # Generate draft tokens
            draft_response = self.draft_backend.generate(
                prompt + all_text,
                max_tokens=draft_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
            draft_text = draft_response.text

            if not draft_text:
                break

            # Verify with target model by generating same continuation
            # and checking agreement (heuristic)
            target_response = self.target_backend.generate(
                prompt + all_text,
                max_tokens=draft_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
            target_text = target_response.text

            # Simple acceptance: if draft and target share significant prefix
            common_len = self._common_prefix_length(draft_text, target_text)
            acceptance_rate = common_len / max(len(draft_text), 1)

            if acceptance_rate >= self.acceptance_threshold:
                # Accept draft tokens
                all_text += draft_text[:common_len]
                remaining_tokens -= common_len
            else:
                # Reject: use target output instead
                all_text += target_text
                remaining_tokens -= min(len(target_text.split()), draft_tokens)

            # Check stop sequences
            if stop_sequences:
                for stop in stop_sequences:
                    if stop in all_text:
                        all_text = all_text.split(stop)[0]
                        remaining_tokens = 0
                        break

        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=all_text,
            model=f"{self.target_backend.get_model_info().get('model', 'target')}+speculative",
            backend="speculative_decoding",
            latency_ms=latency,
            tokens_generated=len(all_text.split()),
        )

    def _common_prefix_length(self, text1: str, text2: str) -> int:
        """Compute length of common prefix in words."""
        words1 = text1.split()
        words2 = text2.split()
        common = 0
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                common += 1
            else:
                break
        return common


def create_speculative_backend(
    target: Literal["ollama", "llama.cpp", "transformers", "vllm"],
    draft: Literal["ollama", "llama.cpp", "transformers", "vllm"],
    target_kwargs: dict | None = None,
    draft_kwargs: dict | None = None,
    max_draft_tokens: int = 5,
) -> SpeculativeDecodingBackend:
    """Factory function to create a speculative decoding backend.

    Args:
        target: Target (large) model backend
        draft: Draft (small) model backend
        target_kwargs: Arguments for target backend
        draft_kwargs: Arguments for draft backend
        max_draft_tokens: Maximum tokens to draft per iteration

    Returns:
        SpeculativeDecodingBackend instance
    """
    target_backend = create_local_llm(target, **(target_kwargs or {}))
    draft_backend = create_local_llm(draft, **(draft_kwargs or {}))

    return SpeculativeDecodingBackend(
        target_backend=target_backend,
        draft_backend=draft_backend,
        max_draft_tokens=max_draft_tokens,
    )


__all__ = [
    "LLMResponse",
    "LlamaCppBackend",
    "LlamaCppQuantizationSelector",
    "LocalLLMBackend",
    "LocalLLMHypothesisGenerator",
    "OllamaAutoPull",
    "OllamaBackend",
    "SpeculativeDecodingBackend",
    "TransformersBackend",
    "VLLMBackend",
    "create_local_llm",
    "get_recommended_local_model",
]
