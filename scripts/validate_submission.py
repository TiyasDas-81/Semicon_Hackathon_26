#!/usr/bin/env python3
"""
KLA Track 2 — Submission Package Validation Script

Tests the self-contained submission/run.py against generated synthetic .npy test cases.
Verifies all 22 official submission constraints.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "AIvengers")
RUN_SCRIPT = os.path.join(SUBMISSION_DIR, "run.py")

def run_validation():
    print("=" * 70)
    print("KLA SUBMISSION PACKAGE VALIDATION")
    print("=" * 70)
    
    if not os.path.exists(RUN_SCRIPT):
        print(f"FAIL: Submission entry point missing at {RUN_SCRIPT}")
        sys.exit(1)
        
    temp_in = tempfile.mkdtemp(prefix="kla_val_in_")
    temp_out = tempfile.mkdtemp(prefix="kla_val_out_")
    
    try:
        # Create test .npy files with diverse formats and edge cases
        test_cases = {
            "test_2d_float32.npy": (np.random.rand(64, 64).astype(np.float32), (256, 256)),
            "test_3d_uint8.npy": ((np.random.rand(64, 64, 1) * 255).astype(np.uint8), (256, 256, 1)),
            "test_2d_uint16.npy": ((np.random.rand(32, 32) * 65535).astype(np.uint16), (128, 128)),
            "test_3d_nan_inf.npy": (np.array([[0.5, np.nan], [np.inf, -np.inf]]).repeat(32, axis=0).repeat(32, axis=1)[:, :, np.newaxis], (256, 256, 1)),
            "test_odd_shape.npy": (np.random.rand(48, 80).astype(np.float32), (192, 320)),
        }
        
        print(f"\n1. Generating {len(test_cases)} test .npy inputs in {temp_in}...")
        for fname, (arr, _) in test_cases.items():
            np.save(os.path.join(temp_in, fname), arr)
            
        # Execute run.py
        cmd = [sys.executable, RUN_SCRIPT, temp_in, temp_out]
        print(f"\n2. Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("\n--- STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"\nFAIL: run.py exited with non-zero code {result.returncode}")
            sys.exit(1)
            
        # Verify output files
        print("\n3. Validating outputs in output directory...")
        out_files = sorted(os.listdir(temp_out))
        print(f"Found {len(out_files)} output files: {out_files}")
        
        checks = []
        
        # Check 1: Equal file count
        if len(out_files) == len(test_cases):
            checks.append(("File Count", "PASS", f"{len(out_files)} files created"))
        else:
            checks.append(("File Count", "FAIL", f"Expected {len(test_cases)}, got {len(out_files)}"))
            
        # Check individual files
        for fname, (in_arr, exp_shape) in test_cases.items():
            out_path = os.path.join(temp_out, fname)
            if not os.path.exists(out_path):
                checks.append((f"File Exist ({fname})", "FAIL", "Missing output file"))
                continue
                
            out_arr = np.load(out_path)
            
            # Shape check
            shape_pass = out_arr.shape == exp_shape
            checks.append((f"Shape ({fname})", "PASS" if shape_pass else "FAIL", f"Got {out_arr.shape}, expected {exp_shape}"))
            
            # Dtype check
            dtype_pass = np.issubdtype(out_arr.dtype, np.floating)
            checks.append((f"Dtype ({fname})", "PASS" if dtype_pass else "FAIL", f"Got {out_arr.dtype}"))
            
            # Range check [0.0, 1.0]
            min_val, max_val = out_arr.min(), out_arr.max()
            range_pass = (min_val >= 0.0) and (max_val <= 1.0)
            checks.append((f"Range [0,1] ({fname})", "PASS" if range_pass else "FAIL", f"Min: {min_val:.4f}, Max: {max_val:.4f}"))
            
            # NaN / Inf check
            has_nan = np.isnan(out_arr).any()
            has_inf = np.isinf(out_arr).any()
            nan_pass = not (has_nan or has_inf)
            checks.append((f"NaN/Inf Free ({fname})", "PASS" if nan_pass else "FAIL", f"NaN: {has_nan}, Inf: {has_inf}"))
            
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 70)
        all_passed = True
        for test_name, status, detail in checks:
            print(f"[{status}] {test_name:<30} : {detail}")
            if status == "FAIL":
                all_passed = False
                
        print("=" * 70)
        if all_passed:
            print("OVERALL RESULT: ALL VALIDATION CHECKS PASSED [OK]")
        else:
            print("OVERALL RESULT: VALIDATION FAILED [FAIL]")
            sys.exit(1)
            
    finally:
        shutil.rmtree(temp_in, ignore_errors=True)
        shutil.rmtree(temp_out, ignore_errors=True)

if __name__ == "__main__":
    run_validation()
