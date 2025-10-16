from dataclasses import dataclass
from typing import Protocol, Literal, Optional, Dict
import torch

Domain = Literal["latent", "image"]

class GuidanceFn(Protocol):
    def __call__(self, pred, **kwargs) -> torch.Tensor:
        """
        pred: predicted x0 at current step. latent or image per `domain`.
        returns: scalar loss (higher = worse), we descend it in the inner loop
        """

@dataclass
class GuidanceConfig:
    enabled: bool = False
    domain: Domain = "image"     # "latent" for PSLD-in-latent; "image" for image-space
    num_steps: int = 5
    step_wt: float = 5.0         # like UGD's --optim_forward_guidance_wt
    k_recur: int = 1             # Number of self-recurrence iterations per timestep
    normalize_grad: bool = True  # Whether to normalize gradients during optimization
    decode_kwargs: Optional[Dict] = None  # e.g., clamp, range, VAE fp

