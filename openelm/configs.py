from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseConfig:
    output_dir: str = "logs/"


@dataclass
class ModelConfig(BaseConfig):
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    deterministic: bool = False
    top_p: float = 0.95
    temp: float = 1.0
    gen_max_len: int = 1024
    max_retries: int = 20
    batch_size: int = 10
    model_type: str = MISSING       # Can be "hf", "openai", "gemini", "claude"
    model_path: str = MISSING       # Can be HF or API-calling model name 
    api_key: str = MISSING          # if model_type is not "hf"
    do_sample: bool = True
    num_return_sequences: int = 1
    device: str = "cuda"

@dataclass
class MutationModelConfig(ModelConfig):
    model_type: str = "openrouter"
    model_path: str = "google/gemma-4-31b-it"
    api_key: str = ""    # openrouter API

@dataclass
class TargetModelConfig(ModelConfig):

    model_type: str = "hf"
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    api_key: str = ""


@dataclass
class QDConfig(BaseConfig):
    init_steps: int = 10
    total_steps: int = 1000
    history_length: int = 1
    save_history: bool = True
    save_snapshot_interval: int = 10
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = True
    crossover_parents: int = 2
    no_variation: bool = False
    current_niche_only: bool = False 

    pool_size = 10

@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (10,))


@dataclass
class CVTMAPElitesConfig(QDConfig):
    qd_name: str = "cvtmapelites"
    n_niches: int = 100
    cvt_samples: int = 10000


@dataclass
class EnvConfig(BaseConfig):
    batch_size: int = 1  # Batch size of MAP-Elites
    env_name: str = MISSING
    debug: bool = False
    seed: Optional[int] = 42

@dataclass
class BiasConfig(EnvConfig):
    env_name: str = "bias"
    behavior: list[int] = field(
        default_factory=lambda: [0, 1]
    )
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 10],
            [0, 10],
        ]
    )


defaults_elm = [
    {"generator": "generator"},
    {"target": "target"},
    {"qd": "mapelites"},
    {"env": "bias"},
    "_self_",
]


@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/nothing/${now:%y-%m-%d_%H:%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_elm)
    generator: Any = MISSING
    target: Any = MISSING
    qd: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None




def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="bias", node=BiasConfig)

    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="qd", name="ga", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtga", node=CVTMAPElitesConfig)
    cs.store(group="generator", name="generator", node=MutationModelConfig)
    cs.store(group="target", name="target", node=TargetModelConfig)
    cs.store(name="elm", node=ELMConfig)
    return cs


CONFIGSTORE = register_configstore()