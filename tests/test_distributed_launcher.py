"""Distributed launcher tests."""


from ptk.config.defaults import scaffold_config
from ptk.config.schema import ComputeStrategy, DeviceType, TrainingMethod
from ptk.distributed.detect import ComputeEnvironment
from ptk.distributed.launcher import render_slurm_script, render_torchrun_script


def test_render_torchrun_script(tmp_path):
    config = scaffold_config(TrainingMethod.SFT, run_name="test-run")
    output = tmp_path / "torchrun.sh"
    render_torchrun_script(config, output)
    content = output.read_text(encoding="utf-8")
    assert "torchrun" in content
    assert "ptk.cli run" in content
    assert output.stat().st_mode & 0o111


def test_render_slurm_script(tmp_path):
    config = scaffold_config(TrainingMethod.SFT, run_name="slurm-run")
    env = ComputeEnvironment(
        device=DeviceType.CUDA,
        strategy=ComputeStrategy.MULTI_NODE,
        num_gpus=2,
        num_nodes=2,
    )
    output = tmp_path / "job.sbatch"
    render_slurm_script(config, env, output)
    content = output.read_text(encoding="utf-8")
    assert "ptk-slurm-run" in content
    assert "nodes=2" in content
    assert "NPROC_PER_NODE=2" in content
