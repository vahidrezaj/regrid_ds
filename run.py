'''
Hydra entry point: read the composed config, instantiate the dataset's
preprocessing class, and run it.
'''

import os
from pathlib import Path
from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate


# Read base directory from .env
load_dotenv()
base_path = Path(os.getenv("LOCAL_DIR"))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    '''
    main
    '''
    preproc_factory = instantiate(cfg.preprocessing.preproc_cls)
    preproc = preproc_factory(cfg, base_path)

    if cfg.get("dry_run", False):
        preproc.report()
    else:
        preproc()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
