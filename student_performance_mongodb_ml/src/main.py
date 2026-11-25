import argparse
from pathlib import Path
from mongo_store import import_csv
from train_models import main as train

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Student Performance ML + MongoDB project")
parser.add_argument("action", choices=["import", "train", "all"])
args = parser.parse_args()
if args.action in ("import", "all"):
    import_csv(ROOT / "data" / "student_performance.csv")
if args.action in ("train", "all"):
    train()
