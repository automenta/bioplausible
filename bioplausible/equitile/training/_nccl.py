"""
NCCL Communication Primitives
=============================

Thin wrappers around ``torch.distributed`` collectives for inter-GPU
communication.  Extracted from ``multigpu.py`` to be shared by both the
unified distributed module and any other code that needs NCCL primitives.
"""

import logging
import os

import torch
import torch.distributed as dist

from bioplausible.equitile.core.config import NCCLConfig

logger = logging.getLogger(__name__)


class NCCLCommunicator:
    """NCCL-based inter-GPU communication.

    Provides:
    - All-reduce for gradient synchronization
    - All-gather for activity exchange
    - Broadcast for weight synchronization
    - Send/recv for tile boundary communication

    Parameters
    ----------
    config : NCCLConfig, optional
        NCCL configuration
    """

    def __init__(self, config: NCCLConfig | None = None) -> None:

        self.config = config or NCCLConfig()
        self.initialized = False
        self.device: torch.device | None = None

    def init_process_group(
        self,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        """Initialize NCCL process group.

        Parameters
        ----------
        rank : int, optional
            Process rank
        world_size : int, optional
            Total processes
        """
        if rank is not None:
            self.config.rank = rank
        if world_size is not None:
            self.config.world_size = world_size

        for key, value in self.config.to_env().items():
            os.environ.setdefault(key, value)

        try:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
            )
            self.device = torch.device(f"cuda:{self.config.rank}")
            torch.cuda.set_device(self.device)
            self.initialized = True

            logger.info(
                "NCCL initialized: rank %d/%d, device %s",
                self.config.rank,
                self.config.world_size,
                self.device,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("NCCL initialization failed: %s", e)
            self.device = torch.device("cpu")

    def destroy(self) -> None:
        """Destroy process group."""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "avg",
    ) -> torch.Tensor:
        """All-reduce a tensor across devices.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to reduce
        op : str
            Reduction operation: 'avg', 'sum', 'min', 'max'

        Returns
        -------
        torch.Tensor
            Reduced tensor
        """
        if not self.initialized or self.config.world_size == 1:
            return tensor

        op_map = {
            "avg": dist.ReduceOp.AVG,
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }
        reduce_op = op_map.get(op, dist.ReduceOp.AVG)
        dist.all_reduce(tensor, op=reduce_op)

        return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Gather tensors from all devices.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to gather

        Returns
        -------
        list of torch.Tensor
            Gathered tensors
        """
        if not self.initialized or self.config.world_size == 1:
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(gathered, tensor)

        return gathered

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast tensor from source device.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to broadcast
        src : int
            Source rank

        Returns
        -------
        torch.Tensor
            Broadcasted tensor
        """
        if not self.initialized or self.config.world_size == 1:
            return tensor

        dist.broadcast(tensor, src=src)
        return tensor

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
        tag: int = 0,
    ) -> None:
        """Send tensor to destination device.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to send
        dst : int
            Destination rank
        tag : int
            Message tag
        """
        if not self.initialized:
            return

        dist.send(tensor, dst=dst, tag=tag)

    def recv(
        self,
        tensor: torch.Tensor,
        src: int = -1,
        tag: int = 0,
    ) -> int:
        """Receive tensor from source device.

        Parameters
        ----------
        tensor : torch.Tensor
            Buffer to receive into
        src : int
            Source rank (-1 for any)
        tag : int
            Message tag

        Returns
        -------
        int
            Source rank
        """
        if not self.initialized:
            return -1

        return dist.recv(tensor, src=src, tag=tag)

    def barrier(self) -> None:
        """Synchronization barrier."""
        if self.initialized:
            dist.barrier()

    @property
    def rank(self) -> int:
        """Get this process's rank."""
        return self.config.rank

    @property
    def world_size(self) -> int:
        """Get total number of processes."""
        return self.config.world_size

    @property
    def is_distributed(self) -> bool:
        """Check if distributed mode is active."""
        return self.initialized and self.config.world_size > 1
