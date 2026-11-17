"""Tests for the validation runner (scripts/compiled/validate.py): scene selection, command assembly, result collection,
the dry run. No simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest compiled.test_validate -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from compiled import validate as val

HERE = os.path.dirname(os.path.abspath(__file__))


class TestPieces(unittest.TestCase):
    def test_test_kitchens_are_refused(self):
        with self.assertRaises(ValueError):
            val.scene_dir("mm_kitchen_3a-1")
        self.assertTrue(val.scene_dir("mm_craftroom_2a-1").endswith(os.path.join("room_setup_fire", "mm_craftroom_2a-1")))
        for s in val.DEFAULT_SCENES:
            self.assertFalse(s.startswith("mm_kitchen_3a-"))

    def test_command(self):
        cmd = val.run_command("p.py", "mm_craftroom_2a-1", "out", clock="tcm", build="B.exe", launch="vglrun -d egl", extra=["--cached_assets"])
        self.assertEqual(cmd[cmd.index("--agent") + 1], "program")
        self.assertEqual(cmd[cmd.index("--program") + 1], "p.py")
        self.assertEqual(cmd[cmd.index("--clock") + 1], "tcm")
        self.assertIn("--cached_assets", cmd)
        self.assertEqual(cmd[cmd.index("--launch") + 1], "vglrun -d egl")

    def test_collect(self):
        out = tempfile.mkdtemp()
        for s, score, illegal in (("a", (2.0, 8.0), 1), ("b", (0.0, 5.0), 0)):
            os.makedirs(os.path.join(out, s))
            json.dump({"avg_score": score[0], "avg_max_score": score[1]}, open(os.path.join(out, s, "eval_result.json"), "w"))
            json.dump({"decisions": 20, "frames_charged_total": 40, "program": {"illegal_or_crashed": illegal, "wall_s_mean": 0.01,
                                                                                 "actions": {"walk_to": 10}}},
                      open(os.path.join(out, s, "summary.json"), "w"))
        res = val.collect(out, ["a", "b", "c"])
        self.assertEqual((res["n"], res["mean"], res["sd"], res["illegal_total"]), (2, 0.125, 0.177, 1))
        self.assertIsNone(res["scenes"]["c"])
        self.assertEqual(res["scenes"]["a"]["value_rate"], 0.25)
        self.assertEqual(res["scenes"]["a"]["frames_charged"], 40)

    def test_dry_run_command_line(self):
        out = os.path.join(tempfile.mkdtemp(), "v")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in [env.get("PYTHONPATH", ""), os.path.dirname(HERE)] if p)
        r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "validate.py"), "--program", "p.py", "--out", out,
                            "--scenes", "mm_craftroom_2a-1", "mm_kitchen_2a-1", "--dry_run"], capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        res = json.load(open(os.path.join(out, "validation.json")))
        self.assertEqual(res["n"], 0)
        self.assertEqual([l["scene"] for l in res["log"]], ["mm_craftroom_2a-1", "mm_kitchen_2a-1"])
        self.assertIn("--program", res["log"][0]["cmd"])


if __name__ == "__main__":
    unittest.main()
