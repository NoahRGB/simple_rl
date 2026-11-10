# NoahRGB 11/2026
#
# a small run script responsible for loading the Hydra
# config and calling start()
#
#

import hydra
from hydra.core.hydra_config import HydraConfig

from simple_rl.core.run import start

@hydra.main(config_path="simple_rl/configs", config_name="config", version_base=None)
def main(cfg):
    start(cfg, run_dir=HydraConfig.get().runtime.output_dir)

if __name__ == "__main__":
    main()