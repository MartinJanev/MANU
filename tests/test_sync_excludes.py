from tests.test_sync import REPO, fake_rsync  # noqa: F401  (fixture)

import subprocess


def test_sync_never_copies_worktrees_or_the_paper(fake_rsync):  # noqa: F811
    env, args = fake_rsync
    env.update(CLUSTER="u@h", REMOTE="/r/manu", RSYNC_DRY="1")
    subprocess.run(["bash", str(REPO / "tools/sync_to_cluster.sh")], env=env, check=True, cwd="/")
    a = args.read_text().splitlines()
    assert "/.claude/" in a and "/paper/" in a
