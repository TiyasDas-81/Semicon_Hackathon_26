import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.app import app

class TestRestoreAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_health_endpoint(self):
        """Tests health check endpoint status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        
    def test_list_datasets_endpoint(self):
        """Tests that available datasets are listed correctly."""
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        datasets = response.json().get("datasets", [])
        self.assertTrue(len(datasets) > 0)
        dataset_ids = [d["id"] for d in datasets]
        self.assertIn("carinthia", dataset_ids)
        self.assertIn("miic", dataset_ids)
        self.assertIn("nist", dataset_ids)
        
    def test_list_samples_valid(self):
        """Tests sample listing for a valid dataset."""
        response = self.client.get("/api/datasets/nist/samples")
        # May be 200 if downloaded, or 400 if not downloaded, but should be a valid FastAPI response
        self.assertIn(response.status_code, [200, 400])
        
    def test_list_samples_invalid_dataset(self):
        """Tests sample listing with an invalid dataset ID returns a 400 error."""
        response = self.client.get("/api/datasets/invalid_dataset_id/samples")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid dataset ID", response.json()["detail"])
        
    def test_path_traversal_detection(self):
        """Tests that path traversal attempts are caught and return a 400 error."""
        # NIST traversal attempt in POST body
        response = self.client.post(
            "/api/restore",
            data={
                "dataset": "nist",
                "sample_id": "../etc/passwd",
                "model": "bicubic",
                "mode": "real"
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Access denied", response.json()["detail"])
        
    def test_restore_synthetic_mode(self):
        """Tests restoration in synthetic mode, verifying metrics calculation."""
        # Create a dummy 64x64 image
        dummy_img = np.zeros((64, 64), dtype=np.uint8)
        _, img_encoded = cv2.imencode(".png", dummy_img)
        img_bytes = img_encoded.tobytes()
        
        # Call restore
        response = self.client.post(
            "/api/restore",
            files={"image": ("dummy.png", img_bytes, "image/png")},
            data={"model": "bicubic", "mode": "synthetic"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restored_b64", data)
        self.assertIn("deviation_b64", data)
        self.assertIn("risk_b64", data)
        # PSNR/SSIM will be N/A/None since dummy has no ground truth file, which is correct
        self.assertIsNone(data["psnr"])
        
    def test_restore_real_mode_metrics_suppression(self):
        """Tests that real mode explicitly suppresses PSNR/SSIM metrics."""
        dummy_img = np.zeros((64, 64), dtype=np.uint8)
        _, img_encoded = cv2.imencode(".png", dummy_img)
        img_bytes = img_encoded.tobytes()
        
        response = self.client.post(
            "/api/restore",
            files={"image": ("dummy.png", img_bytes, "image/png")},
            data={"model": "bicubic", "mode": "real"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["psnr"])
        self.assertIsNone(data["ssim"])
        self.assertEqual(data["mode"], "real")
        
    def test_missing_checkpoint_handling(self):
        """Tests that requesting an unavailable checkpoint throws a clean HTTP 400 error."""
        dummy_img = np.zeros((64, 64), dtype=np.uint8)
        _, img_encoded = cv2.imencode(".png", dummy_img)
        img_bytes = img_encoded.tobytes()
        
        # If we query with an invalid model, it raises 400
        response = self.client.post(
            "/api/restore",
            files={"image": ("dummy.png", img_bytes, "image/png")},
            data={"model": "invalid_model_type", "mode": "real"}
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
