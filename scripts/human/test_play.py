"""Tests for scripts/human/play.sh (A36, file 4): the commands it would run (DRY_RUN=1), the skip of played kitchens, and the
table (TABLE_ONLY=1) over fake result folders. Needs bash (Git Bash); skipped otherwise. No HAZARD, no build.

Run from G:\\iw-thinking-fast:
  venv/Scripts/python.exe -X utf8 -m unittest scripts.human.test_play -v      (or with PYTHONPATH=scripts: human.test_play)
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PLAY = os.path.join(HERE, "play.sh")
BASH = shutil.which("bash")
FIVE = ["mm_kitchen_3a-1", "mm_kitchen_3a-10", "mm_kitchen_3a-11", "mm_kitchen_3a-12", "mm_kitchen_3a-13"]


def run(env_extra, *args):
    env = dict(os.environ, DRY_RUN="1", **env_extra)
    p = subprocess.run([BASH, PLAY, *args], env=env, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


@unittest.skipIf(BASH is None, "bash not on PATH")
class TestPlay(unittest.TestCase):
    def setUp(self):
        self.results = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.results, ignore_errors=True)

    def commands(self, env_extra, *args):
        rc, out, err = run(dict(env_extra, RESULTS=self.results), *args)
        self.assertEqual(rc, 0, err)
        return [l for l in out.splitlines() if "run_baseline.py" in l], out

    def test_default_sitting_is_the_five_kitchens_clock_off_camera_on(self):
        cmds, _ = self.commands({}, "cr_cam")
        self.assertEqual(len(cmds), 5)
        for c, k in zip(cmds, FIVE):
            self.assertIn("--agent human", c)
            self.assertIn("--clock none", c)
            self.assertIn("--episodes 1", c)
            self.assertIn(f"test_set\\\\{k}", c.replace("\\\\\\\\", "\\\\"))  # printf %q escapes the backslashes
            self.assertIn("human_cr_cam", c)  # --resume_dir <results>/human_<sitting>
            self.assertNotIn("--no_camera", c)
        self.assertNotIn("--no_browser", cmds[0])  # the first kitchen opens the tab
        for c in cmds[1:]:
            self.assertIn("--no_browser", c)

    def test_text_only_sitting_passes_no_camera_and_named_kitchens(self):
        cmds, _ = self.commands({"CAMERA": "off", "PORT": "8800", "HOLD": "5"}, "cr_text", "mm_kitchen_3a-2", "mm_kitchen_3a-3")
        self.assertEqual(len(cmds), 2)
        for c in cmds:
            self.assertIn("--no_camera", c)
            self.assertIn("--human_port 8800", c)
            self.assertIn("--human_hold 5", c)
        self.assertIn("mm_kitchen_3a-2", cmds[0])
        self.assertIn("mm_kitchen_3a-3", cmds[1])

    def test_played_kitchens_are_skipped(self):
        d = os.path.join(self.results, "human_cr_text", "mm_kitchen_3a-10")
        os.makedirs(d)
        with open(os.path.join(d, "eval_result.json"), "w") as f:
            json.dump({"avg_score": 1, "avg_max_score": 9}, f)
        cmds, out = self.commands({"CAMERA": "off"}, "cr_text")
        self.assertEqual(len(cmds), 4)
        self.assertIn("mm_kitchen_3a-10 already played, skipping", out)
        self.assertFalse(any("mm_kitchen_3a-10" in c for c in cmds))

    def test_table_reads_the_sitting_and_the_sweeps_on_the_same_kitchens(self):
        def put(path, obj):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(obj, f)
        sit = os.path.join(self.results, "human_cr_text")
        put(os.path.join(sit, "mm_kitchen_3a-1", "eval_result.json"), {"avg_score": 4.5, "avg_max_score": 9})
        put(os.path.join(sit, "mm_kitchen_3a-1", "summary.json"),
            {"human": {"score": 0.5, "decisions": 12, "asked": 10, "automatic": 2, "wall_s_mean": 4.2, "wall_s_max": 9.1},
             "frames_at_30fps_total": 1300, "final_frame_count": 1540, "wall_s_total": 600.0})
        put(os.path.join(self.results, "sweep_fire_random_test25", "mm_kitchen_3a-1", "eval_result.json"), {"avg_score": 6, "avg_max_score": 9})
        put(os.path.join(self.results, "sweep_fire_random_test25", "mm_kitchen_3a-10", "eval_result.json"), {"avg_score": 3, "avg_max_score": 9})
        rc, out, err = run({"TABLE_ONLY": "1", "RESULTS": self.results}, "cr_text")
        self.assertEqual(rc, 0, err)
        lines = out.splitlines()
        human = next(l for l in lines if l.startswith("human cr_text"))
        self.assertIn("0.500", human)
        self.assertIn("0.500 (1)", human)  # mean over the kitchens played
        rnd = next(l for l in lines if l.startswith("random"))
        self.assertIn("0.667", rnd)
        self.assertIn("0.333", rnd)
        self.assertIn("0.500 (2)", rnd)
        self.assertNotIn("rule", "\n".join(l.split()[0] for l in lines if l))  # no rule sweep in the temp results: no row
        self.assertIn("mm_kitchen_3a-1: score 0.5, decisions 12 (10 asked, 2 automatic), think 4.2 s mean / 9.1 s max", out)
        self.assertIn("1300", out)
        self.assertNotIn("run_baseline.py", out)  # table only, nothing launched


if __name__ == "__main__":
    unittest.main()
