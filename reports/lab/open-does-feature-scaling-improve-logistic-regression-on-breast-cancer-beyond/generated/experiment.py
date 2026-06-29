PROTOCOL = {"ablations": ["No feature scaling", "StandardScaler", "MinMaxScaler"], "baseline_models": ["logisticregression", "svc", "decisiontreeclassifier"], "candidate_models": ["decisiontreeclassifier", "gradientboostingclassifier", "logisticregression", "mlpclassifier", "randomforestclassifier", "svc"], "claim_blocked_if": "No significant improvement in balanced accuracy and F1-score after feature scaling across multiple seeds.", "claim_enabled_if": "Statistically significant improvement in balanced accuracy and F1-score across multiple seeds with feature scaling.", "dataset_id": "eeg_eye_state", "dataset_source": "openml", "feature_columns": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14"], "hypothesis": "Feature scaling improves logistic regression performance on breast cancer detection beyond seed noise.", "primary_metric": "balanced_accuracy", "protocol_id": "generated_ml_research_protocol", "question": "Does feature scaling improve logistic regression on breast_cancer beyond seed noise?", "risk_controls": ["Use repeated seeds to ensure robustness against seed noise.", "Verify effect size against run-to-run noise.", "Gate final claims through deterministic verification."], "secondary_metrics": ["accuracy", "f1_macro", "precision_macro", "recall_macro"], "seeds": [42, 52, 62, 72, 82], "split_strategy": "stratified_train_test_split", "target_column": "Class", "task_type": "classification"}
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

# Define a function to run the experiment

def run_experiment(args):
    # Load dataset
    df = pd.read_csv(args.dataset_csv)
    if args.dry_run:
        df = df.sample(n=300, random_state=42)

    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]

    # Define models and scalers
    models = {
        'logisticregression': LogisticRegression(max_iter=1000),
        'svc': SVC(),
        'decisiontreeclassifier': DecisionTreeClassifier(),
        'gradientboostingclassifier': GradientBoostingClassifier(),
        'randomforestclassifier': RandomForestClassifier(),
        'mlpclassifier': MLPClassifier(max_iter=1000)
    }

    scalers = {
        'No feature scaling': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }

    # Prepare results container
    results = []
    start_time = time.time()

    # Iterate over seeds
    for seed in ([42] if args.dry_run else [42, 52, 62, 72, 82]):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

        # Iterate over scalers
        for scaler_name, scaler in scalers.items():
            if scaler:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled, X_test_scaled = X_train, X_test

            # Iterate over models
            for model_name, model in models.items():
                if args.dry_run and model_name not in ['logisticregression', 'svc']:
                    continue

                # Fit model
                model.fit(X_train_scaled, y_train)
                warnings.warn(f"progress: Seed {seed}, Scaler {scaler_name}, Model {model_name} fit complete.")

                # Predict and evaluate
                y_pred = model.predict(X_test_scaled)
                run_result = {
                    'seed': seed,
                    'scaler': scaler_name,
                    'model': model_name,
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_macro': f1_score(y_test, y_pred, average='macro'),
                    'precision_macro': precision_score(y_test, y_pred, average='macro'),
                    'recall_macro': recall_score(y_test, y_pred, average='macro')
                }
                results.append(run_result)

    # Calculate summary statistics
    runtime_seconds = time.time() - start_time
    summary = {
        'mean_balanced_accuracy': np.mean([r['balanced_accuracy'] for r in results]),
        'mean_accuracy': np.mean([r['accuracy'] for r in results]),
        'mean_f1_macro': np.mean([r['f1_macro'] for r in results]),
        'mean_precision_macro': np.mean([r['precision_macro'] for r in results]),
        'mean_recall_macro': np.mean([r['recall_macro'] for r in results])
    }

    # Determine best condition
    best_condition = max(results, key=lambda r: r['balanced_accuracy'])

    # Prepare output JSON
    output = {
        'status': 'completed',
        'protocol_id': 'generated_ml_research_protocol',
        'dataset': 'eeg_eye_state',
        'dataset_source': 'openml',
        'lab_action': 'open-does-feature-scaling-improve-logistic-regression-on-breast-cancer-beyond',
        'primary_metric': 'balanced_accuracy',
        'runs': results,
        'summary': summary,
        'effect_size': None,  # Placeholder
        'seed_noise': None,  # Placeholder
        'effect_to_noise_ratio': None,  # Placeholder
        'best_condition': best_condition,
        'protocol_warnings': [],
        'artifacts': [],
        'runtime_seconds': runtime_seconds
    }

    # Write output JSON
    output_path = Path(args.out_dir) / f'runs/experiment_{args.step}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    # Print JSON to stdout
    print(json.dumps(output, indent=2))

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run supervised classification experiment.')
    parser.add_argument('--dataset-csv', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--target-column', type=str, required=True, help='Name of the target column.')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory for results.')
    parser.add_argument('--step', type=str, required=True, help='Experiment step identifier.')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run with limited data and models.')
    args = parser.parse_args()

    run_experiment(args)
