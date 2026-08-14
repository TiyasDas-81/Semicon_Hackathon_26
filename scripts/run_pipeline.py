import subprocess
import argparse
import sys
import os

def run_cmd(cmd):
    print(f"\nExecuting: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
    
    # Read output line-by-line in real time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            sys.stdout.flush()
            
    rc = process.poll()
    if rc != 0:
        print(f"Command failed with exit code {rc}")
        sys.exit(rc)

def main():
    parser = argparse.ArgumentParser(description="Master script to run the Semicon Image Restoration pipeline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation")
    parser.add_argument("--skip-train-cnn", action="store_true", help="Skip training CNN model")
    parser.add_argument("--skip-train-trans", action="store_true", help="Skip training Transformer model")
    args = parser.parse_args()
    
    # 1. Dataset Generation
    if not args.skip_gen:
        print("\n=== STEP 1: GENERATING SEMICONDUCTOR DATASET ===")
        run_cmd([sys.executable, "scripts/generate_dataset.py", "--config", args.config])
    else:
        print("\nSkipping dataset generation.")

    # 2. Train CNN Baseline
    if not args.skip_train_cnn:
        print("\n=== STEP 2: TRAINING CNN BASELINE (EDSR) ===")
        run_cmd([sys.executable, "training/trainer.py", "--config", args.config, "--model", "cnn"])
    else:
        print("\nSkipping CNN training.")
        
    # 3. Train Transformer Model
    if not args.skip_train_trans:
        print("\n=== STEP 3: TRAINING MAIN TRANSFORMER (SWINIR-LIGHT) ===")
        run_cmd([sys.executable, "training/trainer.py", "--config", args.config, "--model", "transformer"])
    else:
        print("\nSkipping Transformer training.")
        
    # 4. Evaluation and Benchmarking
    print("\n=== STEP 4: EVALUATING AND GENERATING BENCHMARKS ===")
    run_cmd([sys.executable, "evaluation/evaluator.py", "--config", args.config])
    
    print("\n=== PIPELINE RUN COMPLETE! ===")

if __name__ == "__main__":
    main()
