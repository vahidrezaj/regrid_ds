'''
Grab data from the DMI FTP server using lftp.

Point it at one or more folders under REMOTE_DIR (comma-separated, e.g.
`-f abc,cdf,qpt`) and it mirrors each one down to LOCAL_DIR in turn,
printing a separator once a folder is done. Leave -f out to mirror the
whole REMOTE_DIR tree instead. Connection details and paths come from
.env (FTP_HOST, FTP_USERNAME, FTP_PASS, REMOTE_DIR, LOCAL_DIR).

Example usage:
python -m hbm_prep.ftp_downloder.lftp_downloader -f extract_2013-2025_currents,winds_extract

Modes:
    `download`: mirror --continue --parallel={args.parallel} --only-newer
    `mirror`:   mirror --continue --parallel={args.parallel} --delete --no-perms
    `dry_run`:  mirror --dry-run --verbose --continue --delete
'''

import os
import argparse
from pathlib import Path
import subprocess
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, default=None, help=\
                    ("Select one or more folders from FTP_REMOTE_DIR as a "
                     "comma-separated list (e.g. abc,cdf,qpt), or download "
                     "the entire directory if not specified"))
parser.add_argument("-p", "--parallel", type=int, default=8, help=\
                    "Number of parallel downloads")
parser.add_argument("-m", "--mode", type=str, default="download", help=\
                    ("Select mirror modes: {download, mirror, dry_run}.\n"
                     "download : Download/update only [Default] \n"
                     "mirror : Full mirror (delete local files removed from remote)"
                     "dry_run : Dry run"))
args = parser.parse_args()

# Load variables from .env
load_dotenv()

ftp_host = os.getenv("FTP_HOST")
ftp_username = os.getenv("FTP_USERNAME")
ftp_pass = os.getenv("FTP_PASS")
base_remote_dir = os.getenv("REMOTE_DIR")
base_local_dir = os.getenv("LOCAL_DIR")

folders = [f.strip() for f in args.folder.split(",")] if args.folder else [None]

if args.mode.lower() == "download":
    MIRROR = f"mirror --continue --parallel={args.parallel} --only-newer"
elif args.mode.lower() == "mirror":
    MIRROR = f"mirror --continue --parallel={args.parallel} --delete --no-perms"
elif args.mode.lower() == "dry_run":
    MIRROR = "mirror --dry-run --verbose --continue --delete"
else:
    raise ValueError('mode should be one of: download, mirror, dry_run')

for folder in folders:
    if folder is not None:
        remote_dir = str(Path(base_remote_dir, folder).as_posix()) # pylint: disable=C0103
        local_dir = Path(base_local_dir, folder)
    else:
        remote_dir = base_remote_dir
        local_dir = Path(base_local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    local_dir = str(local_dir.as_posix())

    lftp_exp = (
        f"set net:connection-limit {args.parallel}; "
        f"open {ftp_host}; "
        f"user {ftp_username} {ftp_pass}; "
        f"lcd {local_dir}; "
        f"{MIRROR} {remote_dir} .; "
        f"quit"
    )

    print('='*50)
    print(f' Start {args.mode}' + (f' ({folder})' if folder is not None else ''))
    print(f' Remote: {remote_dir}')
    print(f' Local: {local_dir}')
    print('='*50)

    result = subprocess.run(
        ["lftp", "-e", lftp_exp],
        check=False
    )

    if result.returncode != 0:
        print(f"lftp finished with errors (exit code {result.returncode})")

    print('-'*50, "Done")
