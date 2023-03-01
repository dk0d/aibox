from __future__ import print_function

import argparse
import atexit
import datetime
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from pprint import pprint

# SBATCH -e {log_dir}/{name}.%J.err

DEFAULT_TEMPLATE = """\
#!/bin/bash

{header}
#SBATCH -o {log_dir}/{name}.%J.log
#SBATCH -J {name}

{bash_setup}

__script__
"""

VALID_DEPENDENCY_TYPES = {
    "after",
    "afterany",
    "afterburstbuffer",
    "aftercorr",
    "afternotok",
    "afterok",
    "expand",
}


def tmp(suffix=".sh"):
    t = tempfile.mktemp(suffix=suffix)
    atexit.register(os.unlink, t)
    return t


class Slurm(object):
    def __init__(
        self,
        name,
        slurm_kwargs=None,
        tmpl=None,
        date_in_name=True,
        scripts_dir="slurm-scripts",
        log_dir="logs",
        bash_strict=True,
    ):
        if slurm_kwargs is None:
            slurm_kwargs = {}
        if tmpl is None:
            tmpl = DEFAULT_TEMPLATE
        self.log_dir = log_dir
        self.bash_strict = bash_strict

        header = []

        if "time" not in slurm_kwargs.keys():
            slurm_kwargs["time"] = "7-00:00:00"

        for k, v in slurm_kwargs.items():
            if len(k) > 1:
                k = f"--{k}="
            else:
                k = f"-{k} "
            header.append(f"#SBATCH {k}{v}")

        # add bash setup list to collect bash script config
        bash_setup = []
        if bash_strict:
            bash_setup.append("set -eo pipefail -o nounset")

        self.header = "\n".join(header)
        self.bash_setup = "\n".join(bash_setup)
        self.name = "".join(
            x for x in name.replace(" ", "-") if x.isalnum() or x == "-"
        )
        self.tmpl = tmpl
        self.slurm_kwargs = slurm_kwargs
        if scripts_dir is not None:
            self.scripts_dir = os.path.abspath(scripts_dir)
        else:
            self.scripts_dir = None
        self.date_in_name = bool(date_in_name)

    def __str__(self):
        return self.tmpl.format(
            name=self.name,
            header=self.header,
            log_dir=self.log_dir,
            bash_setup=self.bash_setup,
        )

    def _tmpfile(self):
        if self.scripts_dir is None:
            return tmp()
        else:
            for _dir in [self.scripts_dir, self.log_dir]:
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
            return "%s/%s.sh" % (self.scripts_dir, self.name)

    def run(
        self,
        command,
        name_addition=None,
        cmd_kwargs=None,
        _cmd="sbatch",
        tries=1,
        depends_on=None,
        depends_how="afterok",
        debug=False,
    ):
        """
        command: a bash command that you want to run
        name_addition: if not specified, the sha1 of the command to run
                       appended to job name. if it is "date", the yyyy-mlurm-dd
                       date will be added to the job name.
        cmd_kwargs: dict of extra arguments to fill in command
                   (so command itself can be a template).
        _cmd: submit command (change to "bash" for testing).
        tries: try to run a job either this many times or until the first
               success.
        depends_on: job ids that this depends on before it is run (users 'afterok')
        depends_how: ability to change how a job depends on others
        """
        if depends_how not in VALID_DEPENDENCY_TYPES:
            raise ValueError(f"depends_how must be in {VALID_DEPENDENCY_TYPES}")

        if name_addition is None:
            name_addition = hashlib.sha1(command.encode("utf-8")).hexdigest()[:4]

        if self.date_in_name:
            name_addition += "-" + datetime.strftime(
                datetime.now(), format="%y-%m-%d-%H-%M-%S"
            )

        name_addition = name_addition.strip(" -")

        if cmd_kwargs is None:
            cmd_kwargs = {}

        n = self.name
        self.name = self.name.strip(" -")
        self.name += "-" + name_addition.strip(" -")
        args = []
        for k, v in cmd_kwargs.items():
            args.append("export %s=%s" % (k, v))
        args = "\n".join(args)

        tmpl = str(self).replace("__script__", args + "\n###\n" + command)
        if depends_on is None or (len(depends_on) == 1 and depends_on[0] is None):
            depends_on = []

        if debug:
            print(tmpl)
            return

        with open(self._tmpfile(), "w") as sh:
            sh.write(tmpl)

        job_id = None
        for itry in range(1, tries + 1):
            args = [_cmd]
            # sbatch (https://slurm.schedmd.com/sbatch.html) job dependency has the following format:
            # -d, --dependency=<dependency_list>
            #       <dependency_list> is of the form <type:job_id[:job_id][,type:job_id[:job_id]]>
            # Create job dependency string
            dependency_string = "".join([f":{d}" for d in depends_on])
            if depends_on:
                dependency_string = f"{depends_how}{dependency_string}"
            if itry > 1:
                mid = f"afternotok:{job_id}"
                # Merge retry dependency to job dependencies
                if dependency_string:
                    dependency_string = f"{dependency_string},{mid}"
                else:
                    dependency_string = mid
            # Add dependency option to sbatch
            if dependency_string:
                args.extend([f"--dependency={dependency_string}"])
            args.append(sh.name)
            res = subprocess.check_output(args).strip()
            print(str(res, encoding="utf-8"), file=sys.stderr)
            self.name = n
            if not res.startswith(b"Submitted batch"):
                return None
            j_id = int(res.split()[-1])
            if itry == 1:
                job_id = j_id
        return job_id


class SlurmConfig:
    @property
    def params(self) -> dict:
        return {
            re.sub("_", "-", k).lower(): v
            for k, v in self.__dict__.items()
            if v is not None
        }

    def __init__(
        self,
        nodes=1,
        cpus_per_task=1,
        mem_per_cpu="5gb",
        mem=None,
        qos=None,
        partition=None,
        distribution=None,
        mail_user=None,
        mail_type=None,  # (NONE, BEGIN, END, FAIL, ALL)
        gpu=None,
        ngpu=None,
        time="4-00:00:00",
        **kwargs,  # ignore any other kwargs
    ) -> None:
        self.nodes = nodes

        # per pytorch lightning docs: https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html
        self.ntasks_per_node = ngpu

        self.cpus_per_task = cpus_per_task
        self.mem = mem  # total mem
        if self.mem is None:
            self.mem_per_cpu = mem_per_cpu
        assert (
            self.mem is not None or self.mem_per_cpu is not None
        ), f"One of `mem` or `mem_per_cpu` must be set"
        self.qos = qos
        self.mail_user = mail_user
        self.mail_type = mail_type
        self.partition = partition
        self.distribution = distribution
        self.gres = f"gpu:{gpu}:{ngpu}"
        self.time = time


def write_slurm_script(
    name,
    env_name,
    py_file_path,
    py_file_args: dict | list,
    scripts_dir: Path,
    log_dir: Path,
    cudaVersion: str,
    slurm_cfg: SlurmConfig,
    conda_envs_dir="~/.conda/envs",
    verbose=False,
    debug=False,
):


    s = Slurm(
        name,
        slurm_kwargs=slurm_cfg.params,
        date_in_name=False,
        scripts_dir=scripts_dir,
        log_dir=log_dir,
    )

    if verbose:
        pprint(slurm_cfg.params)
        pprint(py_file_args)

    _scriptArgs = [""]
    if isinstance(py_file_args, dict):
        for k, v in py_file_args.items():
            k = k = f"--{k}=" if len(k) > 1 else f"-{k} "
            if isinstance(v, bool):
                if v:
                    k = k[:-1]
                    _scriptArgs.append(f"{k}")
            else:
                _scriptArgs.append(f"{k}{v}")
    else:

        _scriptArgs.extend(py_file_args)

    envPath = Path(conda_envs_dir).expanduser().resolve() / env_name
    _scriptArgs = " ".join(_scriptArgs)
    s.run(
        f"""
echo "Date      : $(date)"
echo "host      : $(hostname -s)"
echo "Directory : $(pwd)"

module purge
module load cuda/{cudaVersion}
module load conda
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "$(nvidia-smi | grep Version)"
echo "Running $SLURM_JOB_NAME on $SLURM_CPUS_ON_NODE CPU cores"
export PATH=$PATH:{envPath / 'bin'}
source activate {envPath}
srun python {Path(py_file_path).expanduser().resolve()}{_scriptArgs}

echo "End Date    : $(date)"
""",
        debug=debug,
    )
