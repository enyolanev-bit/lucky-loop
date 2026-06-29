PROTOCOL = {"ablations": ["No feature scaling", "StandardScaler", "MinMaxScaler"], "baseline_models": ["logisticregression"], "candidate_models": ["gradientboostingclassifier", "logisticregression", "randomforestclassifier", "svc"], "claim_blocked_if": "No significant difference in balanced accuracy with and without feature scaling across multiple seeds.", "claim_enabled_if": "Statistically significant improvement in balanced accuracy with feature scaling across multiple seeds.", "dataset_id": "eeg_eye_state", "dataset_source": "openml", "feature_columns": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14"], "hypothesis": "Feature scaling improves logistic regression accuracy on breast cancer classification.", "primary_metric": "balanced_accuracy", "protocol_id": "generated_ml_research_protocol", "question": "Does feature scaling improve logistic regression accuracy on breast_cancer?", "risk_controls": ["Use repeated seeds to ensure robustness.", "Verify effect size against seed noise.", "Gate final claims through deterministic verification."], "secondary_metrics": ["accuracy", "f1_macro", "precision_macro", "recall_macro"], "seeds": [42, 52, 62, 72, 82], "split_strategy": "stratified_train_test_split", "target_column": "Class", "task_type": "classification"}
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

# Define a function to run the experiment

def run_experiment(args):
    # Load dataset
    df = pd.read_csv(args.dataset_csv)
    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]

    # Prepare output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / 'runs'
    runs_dir.mkdir(exist_ok=True)

    # Define models and scalers
    models = {
        'logisticregression': LogisticRegression(max_iter=1000),
        'gradientboostingclassifier': GradientBoostingClassifier(),
        'randomforestclassifier': RandomForestClassifier(),
        'svc': SVC()
    }

    scalers = {
        'No feature scaling': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }

    # Initialize results
    results = []
    start_time = time.time()

    # Dry run settings
    if args.dry_run:
        seeds = [42]
        model_keys = list(models.keys())[:2]
        df = df.sample(n=300, random_state=42)
    else:
        seeds = [42, 52, 62, 72, 82]
        model_keys = models.keys()

    # Iterate over seeds, models, and scalers
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        for model_key in model_keys:
            model = models[model_key]
            for scaler_name, scaler in scalers.items():
                if scaler:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled, X_test_scaled = X_train, X_test

                # Fit model
                model.fit(X_train_scaled, y_train)
                warnings.warn(f"progress: Seed {seed}, Model {model_key}, Scaler {scaler_name} fit complete.")

                # Predict and evaluate
                y_pred = model.predict(X_test_scaled)
                metrics = {
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_macro': f1_score(y_test, y_pred, average='macro'),
                    'precision_macro': precision_score(y_test, y_pred, average='macro'),
                    'recall_macro': recall_score(y_test, y_pred, average='macro')
                }

                # Record results
                results.append({
                    'seed': seed,
                    'model': model_key,
                    'scaler': scaler_name,
                    'metrics': metrics
                })

    # Calculate summary statistics
    runtime_seconds = time.time() - start_time
    summary = {
        'mean_balanced_accuracy': np.mean([r['metrics']['balanced_accuracy'] for r in results]),
        'std_balanced_accuracy': np.std([r['metrics']['balanced_accuracy'] for r in results])
    }

    # Determine best condition
    best_condition = max(results, key=lambda r: r['metrics']['balanced_accuracy'])

    # Prepare output JSON
    output = {
        'status': 'completed',
        'protocol_id': 'generated_ml_research_protocol',
        'dataset': 'eeg_eye_state',
        'dataset_source': 'openml',
        'lab_action': 'open-does-feature-scaling-improve-logistic-regression-accuracy-on-breast-canc',
        'primary_metric': 'balanced_accuracy',
        'runs': results,
        'summary': summary,
        'effect_size': summary['mean_balanced_accuracy'],
        'seed_noise': summary['std_balanced_accuracy'],
        'effect_to_noise_ratio': summary['mean_balanced_accuracy'] / summary['std_balanced_accuracy'],
        'best_condition': best_condition,
        'protocol_warnings': [],
        'artifacts': [],
        'runtime_seconds': runtime_seconds
    }

    # Write output JSON
    output_path = runs_dir / f'experiment_{args.step}.json'
    output_path.write_text(json.dumps(output, indent=2))

    # Print JSON to stdout
    print(json.dumps(output, indent=2))

# Main function to parse arguments and run the experiment

def main():
    parser = argparse.ArgumentParser(description='Run a supervised classification experiment.')
    parser.add_argument('--dataset-csv', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--target-column', type=str, required=True, help='Name of the target column.')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory for results.')
    parser.add_argument('--step', type=str, required=True, help='Experiment step identifier.')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run with limited data and models.')
    args = parser.parse_args()

    run_experiment(args)

if __name__ == '__main__':
    main()
