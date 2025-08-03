# explainml.py
import argparse
import pandas as pd
from profiler.stats_report import analyze_dataset
from models.trainer import evaluate_models
from reports.report_generator import generate_pdf_report

def main():
    parser = argparse.ArgumentParser(description="ExplainML++ - Intelligent AutoML")
    parser.add_argument("data", help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--output", default="reports/report.pdf", help="Output report path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows")

    profile = analyze_dataset(df, args.target)
    print(f"Task: {profile['task_type']} | Imbalance: {profile.get('imbalance_ratio'):.2f}x")

    X, y = df.drop(columns=[args.target]), df[args.target]
    X = X.select_dtypes(include=[float, int]).fillna(0)

    results, best_model = evaluate_models(X, y)
    print(f"🏆 Best: {results.iloc[0]['model']} | F1: {results.iloc[0]['f1_mean']:.3f}")

    diag_data = {
        "dataset": args.data,
        "target": args.target,
        "best_model": results.iloc[0]["model"],
        "f1_score": results.iloc[0]["f1_mean"],
        "suggestions": [{"suggestion": "Consider SMOTE", "priority": "high"}]
    }
    generate_pdf_report(diag_data, args.output)

if __name__ == "__main__":
    main()