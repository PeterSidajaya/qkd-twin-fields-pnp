"""Dependency-free tests for Stage 5A PM optimization orchestration."""
from concurrent.futures import ThreadPoolExecutor
import random
import unittest
from unittest.mock import patch

import pnp_pm_optimize as opt


class PMOptimizerTests(unittest.TestCase):
    def test_reproducible_stratified_candidates(self):
        lhs = opt.latin_hypercube(16, 3, 0.05, random.Random(123))
        self.assertEqual(lhs, opt.latin_hypercube(16, 3, 0.05, random.Random(123)))
        self.assertTrue(all(len(point) == 3 and all(0.05 <= x <= 1.0 for x in point)
                            for point in lhs))

    def test_fixed_and_handpicked_seeds(self):
        seeds = opt._seed_profiles(3, 0.05)
        self.assertEqual(seeds[0], (1.0, 1.0, 1.0))
        self.assertEqual(seeds[1], (0.8, 0.5, 0.2))

    def test_inaccurate_solve_cannot_win(self):
        accurate = {"status": "optimal", "R_acc": 0.0, "e_ph": 0.5}
        inaccurate = {"status": "optimal_inaccurate", "R_acc": 1.0, "e_ph": 0.0}
        self.assertLess(opt._rank(inaccurate), opt._rank(accurate))

    def test_parallel_batch_search_with_mock_sdps(self):
        def fake_sdp(task):
            distance, test, _ = task
            squared = sum((x - y) ** 2 for x, y in zip(test, (0.7, 0.4, 0.2)))
            return {"distance_km": distance, "lambda_by_basis": [1.0, *test],
                    "status": "optimal", "R_acc": max(0.0, 1.0 - squared),
                    "e_ph": 0.1 + 0.1 * squared}

        with patch.object(opt, "_evaluate", fake_sdp), ThreadPoolExecutor(max_workers=3) as pool:
            best, fixed, records = opt.optimize_distance(
                pool, 20.0, {"M": 4, "lambda_min": 0.05},
                random_samples=12, refine_rounds=5, rng=random.Random(123))
        self.assertEqual(fixed["lambda_by_basis"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(best["lambda_by_basis"][0], 1.0)
        self.assertGreater(best["R_acc"], fixed["R_acc"])
        self.assertGreater(len(records), 15)


if __name__ == "__main__":
    unittest.main()
