import hydra
from omegaconf import OmegaConf

from openelm.elm import ELM


def set_seed(seed=None, deterministic=False) -> int:
    if seed is None:
        seed = np.random.default_rng().integers(2**32 - 1)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)
    return seed


@hydra.main(
    config_name="elm",
    version_base="1.2",
)
def main(config):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    set_seed(config.qd.seed)

    elm = ELM(config)
    best_indv = elm.run(init_steps=config.qd.init_steps, total_steps=config.qd.total_steps)

    print("Best Individual: ", best_indv)

if __name__ == "__main__":
    main()