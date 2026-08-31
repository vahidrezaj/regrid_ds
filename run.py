'''
Hydra entry point: read the composed config and, depending on `mode`, either
run the dataset's preprocessing pipeline, log a dry-run summary, or check
already-saved output on disk.
'''

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from hbm_prep.validate import validate_output

MODES = ("run", "dry_run", "check")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    '''
    main
    '''
    mode = cfg.get("mode", "run")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if mode == "check":
        # read-only: doesn't need source data, so no LOCAL_DIR/base_path required
        ok = validate_output(cfg)
        sys.exit(0 if ok else 1)

    load_dotenv()
    base_path = Path(os.getenv("LOCAL_DIR"))
    preproc_factory = instantiate(cfg.preprocessing.preproc_cls)
    preproc = preproc_factory(cfg, base_path)

    if mode == "dry_run":
        preproc.report()
    else:
        preproc()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
