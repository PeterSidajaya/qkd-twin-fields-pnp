"""
Lightweight consistency tests for shared PnP SDP reporting/observations.
Run: python -m unittest phase/test_pnp_output.py
Does not solve SDPs. Requires the scripts' numerical dependencies.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pnp_benchmark as bench
from pnp_results import COLUMNS, result_row, write_outputs


def load_script(filename,name):
    path=HERE/filename
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


class TestUnifiedPnPOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s5=load_script("pnp-stage-5a.py","test_pnp_stage5a")
        cls.pm=load_script("pnp-stage-5a-PM.py","test_pnp_stage5a_pm")

    def test_reference_normalizes(self):
        for M in (1,2,4):
            prob,p_pass,ebit=bench.coherent_reference_probabilities(M,0.01,20)
            self.assertTrue(np.allclose(prob[0]+prob[1]+prob[2],1.0,atol=1e-14))
            self.assertGreater(p_pass,0.0)
            self.assertLessEqual(ebit,0.5+1e-8)

    def test_lambda_one_limit(self):
        self.pm.check_pm_kernel_identities()
        settings=self.pm.make_pm_settings([1.0]*4)
        for L in (0.0,2.0,20.0,40.0):
            expected,pass1,_=self.s5.synthetic_announcement_probabilities(4,0.01,L)
            observed=self.pm.pm_synthetic_announcement_probabilities(settings,0.01,L)
            for z in self.s5.Z_ALL:
                self.assertTrue(np.allclose(observed[z],expected[z],atol=2e-15,rtol=0),(z,L))
            pass2=np.mean((observed[self.s5.Z_P]+observed[self.s5.Z_M])[:4])
            self.assertAlmostEqual(pass1,pass2,places=13)

    def test_schema_csv(self):
        rows=[
            result_row("stage1","baseline",2,M=4,n_bar=0.01,
                       e_ph_upper=.25,e_bit=.015,P_pass_cond=.003,
                       R_acc=.0005,status="optimal",
                       phase_bound_scope="legacy e_ph<=1/2",
                       parameters={"nmax":7}),
            result_row("stage4","upper tail",2,M=4,n_bar=0.01,
                       e_ph_upper=.4,e_bit=.015,P_pass_cond=.003,
                       R_acc=0,status="optimal",P_monitor_scope="unknown",
                       epsilon_tail=.01,beta=.001),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(rows,stage="test",base_dir=tmp,plot=False)
            with (Path(tmp)/"results.csv").open(newline="") as f:
                entries=list(csv.DictReader(f))
            self.assertEqual(list(entries[0].keys()),list(COLUMNS))
            self.assertEqual(entries[0]["P_monitor"],"1.0")
            self.assertEqual(entries[1]["R_original"],"")
            self.assertEqual(entries[1]["epsilon_tail"],"0.01")


if __name__=="__main__":
    unittest.main()
