"""
Joint Kernel Cache for 6-D Architecture.

Persists compiled/cached kernels for:
- CoupledTransition.step
- Plasticity update
- Stability estimator
- Adapter projection

Cache key: coordinate hash + tensor shapes + dtype/precision + device + adapter stack
"""

from __future__ import annotations

import hashlib
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from computronium.core.joint.transition import CoupledTransition
    from computronium.core.plasticity import PlasticityPrimitive
    from computronium.state import CompositeState, SystemContext


@dataclass(frozen=True, slots=True)
class KernelCacheKey:
    """Composite key for kernel cache entries."""

    coordinate_hash: str  # Hash of 6-D coordinate (S/G/D/M/C/U)
    tensor_shapes: tuple[tuple[int, ...], ...]  # Input tensor shapes
    dtype: str  # e.g., "float32", "bfloat16"
    device: str  # e.g., "cuda:0", "cpu"
    adapter_stack_hash: str  # Hash of adapter composition
    kernel_type: str  # "transition", "plasticity", "stability", "adapter"

    def to_string(self) -> str:
        """Convert to string for storage."""
        parts = [
            self.coordinate_hash,
            "|".join(str(s) for s in self.tensor_shapes),
            self.dtype,
            self.device,
            self.adapter_stack_hash,
            self.kernel_type,
        ]
        return "::".join(parts)

    @classmethod
    def from_string(cls, s: str) -> KernelCacheKey:
        """Parse from string."""
        parts = s.split("::")
        if len(parts) != 6:
            raise ValueError(f"Invalid cache key format: {s}")
        return cls(
            coordinate_hash=parts[0],
            tensor_shapes=tuple(
                tuple(int(x) for x in s.split(",")) for s in parts[1].split("|") if s
            ),
            dtype=parts[2],
            device=parts[3],
            adapter_stack_hash=parts[4],
            kernel_type=parts[5],
        )


@dataclass(frozen=True, slots=True)
class KernelCacheEntry:
    """Cached kernel entry with metadata."""

    key: KernelCacheKey
    compiled_artifact: bytes  # Pickled compiled kernel (torch.compile, Triton, etc.)
    metadata: dict[str, Any]  # Compilation time, memory, etc.
    hit_count: int = 0
    created_at: str = ""  # ISO timestamp


class JointKernelCache:
    """
    Thread-safe cache for compiled joint architecture kernels.

    Supports:
    - In-memory LRU cache for fast access
    - Persistent disk cache for cross-process reuse
    - Cache key based on 6-D coordinate + shapes + precision + device + adapters
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_memory_entries: int = 128,
        max_disk_entries: int = 1024,
        enable_disk: bool = True,
    ):
        """
        Initialize the kernel cache.

        Args:
            cache_dir: Directory for persistent disk cache
            max_memory_entries: Maximum entries in memory LRU
            max_disk_entries: Maximum entries on disk
            enable_disk: Whether to enable disk persistence
        """
        self._max_memory = max_memory_entries
        self._max_disk = max_disk_entries
        self._enable_disk = enable_disk

        # In-memory LRU cache
        self._memory_cache: OrderedDict[str, KernelCacheEntry] = OrderedDict()
        self._memory_lock = threading.Lock()

        # Disk cache
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "computronium" / "kernel_cache"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_index_path = self._cache_dir / "index.json"
        self._disk_index: dict[str, dict] = {}
        self._disk_lock = threading.Lock()
        self._load_disk_index()

    def _load_disk_index(self) -> None:
        """Load disk cache index."""
        if self._disk_index_path.exists():
            try:
                import json

                with self._disk_index_path.open("r") as f:
                    self._disk_index = json.load(f)
            except Exception:
                self._disk_index = {}

    def _save_disk_index(self) -> None:
        """Save disk cache index."""
        try:
            import json

            with self._disk_index_path.open("w") as f:
                json.dump(self._disk_index, f)
        except Exception:
            pass

    @staticmethod
    def _hash_coordinate(coordinate: str) -> str:
        """Hash a 6-D coordinate string."""
        return hashlib.sha256(coordinate.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_adapter_stack(adapters: list[Any]) -> str:
        """Hash a stack of adapters."""
        adapter_infos = []
        for adapter in adapters:
            adapter_infos.append(f"{type(adapter).__name__}:{id(adapter)}")
        return hashlib.sha256("|".join(adapter_infos).encode()).hexdigest()[:16]

    @staticmethod
    def _get_tensor_shapes(
        tensors: dict[str, torch.Tensor] | list[torch.Tensor],
    ) -> tuple[tuple[int, ...], ...]:
        """Extract tensor shapes for cache key."""
        if isinstance(tensors, dict):
            shapes = tuple(tuple(t.shape) for t in tensors.values())
        else:
            shapes = tuple(tuple(t.shape) for t in tensors)
        return shapes

    def _make_key(
        self,
        coordinate: str,
        tensor_shapes: tuple[tuple[int, ...], ...],
        dtype: torch.dtype,
        device: str,
        adapter_stack: list[Any] | None,
        kernel_type: str,
    ) -> KernelCacheKey:
        """Create a cache key."""
        return KernelCacheKey(
            coordinate_hash=self._hash_coordinate(coordinate),
            tensor_shapes=tensor_shapes,
            dtype=str(dtype).replace("torch.", ""),
            device=device,
            adapter_stack_hash=self._hash_adapter_stack(adapter_stack or []),
            kernel_type=kernel_type,
        )

    def get(
        self,
        coordinate: str,
        tensor_shapes: tuple[tuple[int, ...], ...],
        dtype: torch.dtype,
        device: str,
        adapter_stack: list[Any] | None,
        kernel_type: str,
    ) -> Any | None:
        """
        Get a cached kernel.

        Returns:
            The compiled kernel artifact, or None if not found.
        """
        key = self._make_key(
            coordinate, tensor_shapes, dtype, device, adapter_stack, kernel_type
        )
        key_str = key.to_string()

        # Check memory cache first
        with self._memory_lock:
            entry = self._memory_cache.get(key_str)
            if entry is not None:
                # Update hit count and move to end (LRU)
                new_entry = KernelCacheEntry(
                    key=entry.key,
                    compiled_artifact=entry.compiled_artifact,
                    metadata=entry.metadata,
                    hit_count=entry.hit_count + 1,
                    created_at=entry.created_at,
                )
                self._memory_cache[key_str] = new_entry
                self._memory_cache.move_to_end(key_str)
                return pickle.loads(entry.compiled_artifact)

        # Check disk cache
        if self._enable_disk:
            with self._disk_lock:
                if key_str in self._disk_index:
                    entry_path = self._cache_dir / f"{key_str}.pkl"
                    if entry_path.exists():
                        try:
                            with entry_path.open("rb") as f:
                                artifact = pickle.load(f)
                            # Promote to memory cache
                            self.put(
                                key_str,
                                artifact,
                                KernelCacheEntry(
                                    key=key,
                                    compiled_artifact=pickle.dumps(artifact),
                                    metadata=self._disk_index[key_str].get(
                                        "metadata", {}
                                    ),
                                    hit_count=self._disk_index[key_str].get(
                                        "hit_count", 0
                                    )
                                    + 1,
                                    created_at=self._disk_index[key_str].get(
                                        "created_at", ""
                                    ),
                                ),
                            )
                            return artifact
                        except Exception:
                            pass

        return None

    def put(
        self,
        key_str: str,
        artifact: Any,
        entry: KernelCacheEntry | None = None,
    ) -> None:
        """Store a compiled kernel in the cache."""
        if entry is None:
            # Create entry from key_str
            key = KernelCacheKey.from_string(key_str)
            entry = KernelCacheEntry(
                key=key,
                compiled_artifact=pickle.dumps(artifact),
                metadata={},
                hit_count=0,
                created_at="",
            )

        # Add to memory cache
        with self._memory_lock:
            self._memory_cache[key_str] = entry
            self._memory_cache.move_to_end(key_str)
            while len(self._memory_cache) > self._max_memory:
                self._memory_cache.popitem(last=False)

        # Add to disk cache
        if self._enable_disk:
            with self._disk_lock:
                entry_path = self._cache_dir / f"{key_str}.pkl"
                try:
                    with entry_path.open("wb") as f:
                        pickle.dump(artifact, f)
                    self._disk_index[key_str] = {
                        "metadata": entry.metadata,
                        "hit_count": entry.hit_count,
                        "created_at": entry.created_at,
                    }
                    self._save_disk_index()

                    # Prune disk cache if needed
                    if len(self._disk_index) > self._max_disk:
                        # Remove oldest entries
                        sorted_keys = sorted(
                            self._disk_index.keys(),
                            key=lambda k: self._disk_index[k].get("created_at", ""),
                        )
                        for old_key in sorted_keys[
                            : len(self._disk_index) - self._max_disk
                        ]:
                            old_path = self._cache_dir / f"{old_key}.pkl"
                            old_path.unlink(missing_ok=True)
                            del self._disk_index[old_key]
                        self._save_disk_index()
                except Exception:
                    pass

    def cache_transition_step(
        self,
        coordinate: str,
        transition: CoupledTransition,
        context: SystemContext,
        sample_state: CompositeState,
        adapter_stack: list[Any] | None = None,
    ) -> Any:
        """
        Cache or retrieve a compiled CoupledTransition.step.

        Args:
            coordinate: 6-D coordinate string
            transition: The CoupledTransition instance
            context: SystemContext with immutable parameters
            sample_state: Sample CompositeState for shape inference
            adapter_stack: List of adapters in the pipeline

        Returns:
            Compiled step function
        """
        # Extract all tensors from CompositeState for shape inference
        all_tensors = {}
        all_tensors.update(sample_state.activity)
        all_tensors.update(sample_state.plastic)
        all_tensors.update(sample_state.substrate)
        tensor_shapes = self._get_tensor_shapes(all_tensors)
        dtype = next(iter(all_tensors.values())).dtype if all_tensors else torch.float32
        device = str(next(iter(all_tensors.values())).device) if all_tensors else "cpu"

        # Try to get from cache
        cached = self.get(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "transition"
        )
        if cached is not None:
            return cached

        # Compile the transition step
        compiled_step = torch.compile(
            transition.step, mode="reduce-overhead", fullgraph=True
        )

        # Warm up with sample input
        with torch.no_grad():
            _ = compiled_step(sample_state, context)

        # Cache it
        key = self._make_key(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "transition"
        )
        key_str = key.to_string()
        self.put(key_str, compiled_step)

        return compiled_step

    def cache_plasticity_update(
        self,
        coordinate: str,
        plasticity: PlasticityPrimitive,
        context: SystemContext,
        sample_psi: dict[str, torch.Tensor],
        sample_state: CompositeState,
        adapter_stack: list[Any] | None = None,
    ) -> Any:
        """Cache or retrieve a compiled plasticity update step."""
        # Extract all tensors for shape inference
        all_tensors = {}
        all_tensors.update(sample_psi)
        all_tensors.update(sample_state.activity)
        all_tensors.update(sample_state.plastic)
        all_tensors.update(sample_state.substrate)
        tensor_shapes = self._get_tensor_shapes(all_tensors)
        dtype = next(iter(all_tensors.values())).dtype if all_tensors else torch.float32
        device = str(next(iter(all_tensors.values())).device) if all_tensors else "cpu"

        cached = self.get(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "plasticity"
        )
        if cached is not None:
            return cached

        compiled_update = torch.compile(
            plasticity.step, mode="reduce-overhead", fullgraph=True
        )

        with torch.no_grad():
            _ = compiled_update(sample_psi, sample_state, context)

        key = self._make_key(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "plasticity"
        )
        key_str = key.to_string()
        self.put(key_str, compiled_update)

        return compiled_update

    def cache_stability_estimator(
        self,
        coordinate: str,
        estimator: Any,  # StabilityMonitor (protocol not yet defined)
        sample_trajectory: list[dict[str, torch.Tensor]],
        adapter_stack: list[Any] | None = None,
    ) -> Any:
        """Cache or retrieve a compiled stability estimator."""
        tensor_shapes = self._get_tensor_shapes(
            sample_trajectory[0] if sample_trajectory else {}
        )
        dtype = (
            next(iter(sample_trajectory[0].values())).dtype
            if sample_trajectory
            else torch.float32
        )
        device = (
            str(next(iter(sample_trajectory[0].values())).device)
            if sample_trajectory
            else "cpu"
        )

        cached = self.get(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "stability"
        )
        if cached is not None:
            return cached

        compiled_estimator = torch.compile(
            estimator.estimate, mode="reduce-overhead", fullgraph=True
        )

        if sample_trajectory:
            with torch.no_grad():
                _ = compiled_estimator(sample_trajectory)

        key = self._make_key(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "stability"
        )
        key_str = key.to_string()
        self.put(key_str, compiled_estimator)

        return compiled_estimator

    def cache_adapter_projection(
        self,
        coordinate: str,
        adapter: Any,
        sample_input: dict[str, torch.Tensor],
        adapter_stack: list[Any] | None = None,
    ) -> Any:
        """Cache or retrieve a compiled adapter projection."""
        tensor_shapes = self._get_tensor_shapes(sample_input)
        dtype = next(iter(sample_input.values())).dtype
        device = str(next(iter(sample_input.values())).device)

        cached = self.get(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "adapter"
        )
        if cached is not None:
            return cached

        compiled_adapter = torch.compile(
            adapter.project, mode="reduce-overhead", fullgraph=True
        )

        with torch.no_grad():
            _ = compiled_adapter(sample_input)

        key = self._make_key(
            coordinate, tensor_shapes, dtype, device, adapter_stack, "adapter"
        )
        key_str = key.to_string()
        self.put(key_str, compiled_adapter)

        return compiled_adapter

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._memory_lock:
            mem_entries = len(self._memory_cache)
            mem_hits = sum(e.hit_count for e in self._memory_cache.values())

        with self._disk_lock:
            disk_entries = len(self._disk_index)
            disk_hits = sum(v.get("hit_count", 0) for v in self._disk_index.values())

        return {
            "memory_entries": mem_entries,
            "memory_hits": mem_hits,
            "disk_entries": disk_entries,
            "disk_hits": disk_hits,
            "cache_dir": str(self._cache_dir),
        }

    def clear(self) -> None:
        """Clear all caches."""
        with self._memory_lock:
            self._memory_cache.clear()

        if self._enable_disk:
            with self._disk_lock:
                for key in list(self._disk_index.keys()):
                    (self._cache_dir / f"{key}.pkl").unlink(missing_ok=True)
                self._disk_index.clear()
                self._save_disk_index()


# Global instance for convenience
_global_cache: JointKernelCache | None = None


def get_kernel_cache() -> JointKernelCache:
    """Get or create the global kernel cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = JointKernelCache()
    return _global_cache


def set_kernel_cache(cache: JointKernelCache) -> None:
    """Set the global kernel cache instance."""
    global _global_cache
    _global_cache = cache
