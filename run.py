'''
Hydra entry point: read the composed config and, depending on `mode`, either
run the dataset's preprocessing pipeline, log a dry-run summary, or check
already-saved output on disk.
'''

import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from validate import validate_output

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
        # read-only: doesn't need source data
        ok = validate_output(cfg)
        sys.exit(0 if ok else 1)

    preproc_factory = instantiate(cfg.domain.preproc_cls)
    preproc = preproc_factory(cfg)

    if mode == "dry_run":
        preproc.report()
    else:
        preproc.report()
        preproc()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
