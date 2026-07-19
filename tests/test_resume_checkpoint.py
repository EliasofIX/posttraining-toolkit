"""Resume checkpoint resolution tests."""


from ptk.training.base_trainer import resolve_resume_checkpoint


def test_resolve_resume_checkpoint_no_auto_scan(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint-10").mkdir()
    (ckpt_dir / "final").mkdir()

    assert resolve_resume_checkpoint(ckpt_dir) is None
    assert resolve_resume_checkpoint(ckpt_dir, auto_resume=True) == str(ckpt_dir / "checkpoint-10")


def test_resolve_resume_checkpoint_explicit_path(tmp_path):
    ckpt = tmp_path / "checkpoints" / "final"
    ckpt.mkdir(parents=True)
    assert resolve_resume_checkpoint(tmp_path / "checkpoints", ckpt) == str(ckpt)
