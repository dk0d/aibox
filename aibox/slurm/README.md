
send in job name and kwargs for slurm params:

```python
>>> s = Slurm("job-name", {"account": "ucgd-kp", "partition": "ucgd-kp"})
>>> print(str(s))
#!/bin/bash
<BLANKLINE>
#SBATCH -e logs/job-name.%J.err
#SBATCH -o logs/job-name.%J.out
#SBATCH -J job-name
<BLANKLINE>
#SBATCH --account=ucgd-kp
#SBATCH --partition=ucgd-kp
#SBATCH --time=84:00:00
<BLANKLINE>
set -eo pipefail -o nounset
<BLANKLINE>

__script__
```

```python
>>> s = Slurm("job-name", {"account": "ucgd-kp", "partition": "ucgd-kp"}, bash_strict=False)
>>> print(str(s))
#!/bin/bash
<BLANKLINE>
#SBATCH -e logs/job-name.%J.err
#SBATCH -o logs/job-name.%J.out
#SBATCH -J job-name
<BLANKLINE>
#SBATCH --account=ucgd-kp
#SBATCH --partition=ucgd-kp
#SBATCH --time=84:00:00
<BLANKLINE>
<BLANKLINE>
<BLANKLINE>

__script__
```

```python
>>> job_id = s.run("rm -f aaa; sleep 10; echo 213 > aaa", name_addition="", tries=1)
>>> job = s.run("cat aaa; rm aaa", name_addition="", tries=1, depends_on=[job_id])
```
