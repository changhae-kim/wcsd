import yaml

from dacite import from_dict
from dataclasses import dataclass, field

@dataclass
class Config:

    device: str = "cuda"

    # Model Parameters
    io_channels: list[int] = None
    mid_channels: int = 8
    Nt: int = 3
    Ns_list: list[int] = field(default_factory=lambda: [3, 5, 7])
    num_slayers: int = 1

    # Trainer Parameters
    optimizer: str = "Adam"
    lr: float = 1e-3
    epochs: int = 300

    # XAS Normalizer Parameters
    spec_type: str = None
    denoise_channels: list[str] = None
    labels: str = None
    pre_edge_kws: dict = None
