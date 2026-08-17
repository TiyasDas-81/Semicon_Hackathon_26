#!/usr/bin/env python3
"""
KLA Track 2 — Submission Package Audit Script

Scans the submission directory for:
- Required files and directories
- Model weight presence
- Forbidden runtime dependencies (download URLs, API keys, hardcoded user paths)
"""

import os
import sys
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "AIvengers")

REQUIRED_ITEMS = [
    "run.py",
    "requirements.txt",
    "README.md",
    "models",
    os.path.join("models", "best_cnn.pth"),
    os.path.join("models", "model_definition.py"),
]

FORBIDDEN_PATTERNS = [
    (r"https?://", "HTTP/HTTPS download URL"),
    (r"huggingface", "Hugging Face download reference"),
    (r"torch\.hub", "Torch Hub download reference"),
    (r"pretrained\s*=\s*True", "Automatic pretrained model download"),
    (r"requests\.", "Requests HTTP module call"),
    (r"urllib\.", "Urllib HTTP module call"),
    (r"C:\\Users\\", "Hardcoded Windows user path"),
    (r"/home/[a-zA-Z0-9_-]+/", "Hardcoded Linux user path"),
    (r"api_key", "API key reference"),
    (r"kaggle", "Kaggle credential reference"),
]

def audit_submission():
    print("=" * 70)
    print("KLA SUBMISSION PACKAGE AUDIT")
    print("=" * 70)
    
    audit_passed = True
    
    # 1. Structure Verification
    print("\n1. Verifying Required Submission Files & Model Weights...")
    for item in REQUIRED_ITEMS:
        full_path = os.path.join(SUBMISSION_DIR, item)
        if os.path.exists(full_path):
            size_str = ""
            if os.path.isfile(full_path):
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                size_str = f"({size_mb:.2f} MB)"
            print(f"  [PASS] Found: {item} {size_str}")
        else:
            print(f"  [FAIL] Missing: {item}")
            audit_passed = False
            
    # 2. Scanning Code for Forbidden External / Path Dependencies
    print("\n2. Scanning Python Code for Forbidden Runtime Dependencies...")
    py_files = []
    for root, _, files in os.walk(SUBMISSION_DIR):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
                
    for py_file in py_files:
        rel_path = os.path.relpath(py_file, SUBMISSION_DIR)
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        file_clean = True
        for pattern, desc in FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"  [FAIL] {rel_path}: Found {desc} (pattern: '{pattern}')")
                file_clean = False
                audit_passed = False
                
        if file_clean:
            print(f"  [PASS] {rel_path}: No forbidden runtime dependencies found.")
            
    print("\n" + "=" * 70)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 70)
    if audit_passed:
        print("OVERALL AUDIT: SUBMISSION PACKAGE IS CLEAN & PORTABLE [OK]")
    else:
        print("OVERALL AUDIT: ISSUES DETECTED IN SUBMISSION PACKAGE ❌")
        sys.exit(1)

if __name__ == "__main__":
    audit_submission()
