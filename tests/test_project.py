"""Lightweight integrity checks for the synthetic dataset and feature contract."""
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_features

class ShipmentDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data" / "shipments_synthetic.csv")

    def test_expected_record_count_and_unique_ids(self):
        self.assertEqual(len(self.data), 300)
        self.assertTrue(self.data["shipment_id"].is_unique)

    def test_targets_are_well_formed(self):
        self.assertTrue(self.data["final_delay_hours"].notna().all())
        expected = (self.data["final_delay_hours"] > 24).astype(int)
        self.assertTrue((expected == self.data["delayed_over_24h"]).all())

    def test_missing_intermediate_data_is_present(self):
        stage_cols = [
            "origin_departure_delay_hours",
            "port_processing_delay_hours",
            "customs_delay_hours",
        ]
        self.assertTrue(self.data[stage_cols].isna().any().any())

    def test_feature_contract_excludes_future_outcome(self):
        X = build_features(self.data.head(5))
        self.assertEqual(set(X.columns), set(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
        self.assertNotIn("final_delay_hours", X.columns)
        self.assertNotIn("actual_delivery", X.columns)
        self.assertEqual(len(X), 5)

if __name__ == "__main__":
    unittest.main(verbosity=2)
