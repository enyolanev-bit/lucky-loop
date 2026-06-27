#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time, warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC


def get_dataset(name: str):
    if name == "breast_cancer":
        d = load_breast_cancer()
    elif name == "wine":
        d = load_wine()
    elif name == "digits":
        d = load_digits()
    else:
        raise ValueError(f"unknown dataset: {name}")
    return d.data, d.target


def build_model(args):
    if args.model == "logistic_regression":
        clf = LogisticRegression(max_iter=args.max_iter, C=args.C, random_state=args.seed)
        return make_pipeline(StandardScaler(), clf) if args.scale else clf
    if args.model == "random_forest":
        return RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed, n_jobs=-1)
    if args.model == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth or 3, random_state=args.seed)
    if args.model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth, random_state=args.seed)
    if args.model == "svc":
        clf = SVC(C=args.C, kernel=args.kernel, gamma="scale", random_state=args.seed)
        return make_pipeline(StandardScaler(), clf) if args.scale else clf
    raise ValueError(f"unknown model: {args.model}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "wine", "digits"])
    p.add_argument("--model", required=True, choices=["logistic_regression", "random_forest", "gradient_boosting", "hist_gradient_boosting", "svc"])
    p.add_argument("--scale", action="store_true")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--kernel", default="rbf")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-noise", type=float, default=0.0, help="Fraction of training labels to flip for controlled perturbation demos.")
    args = p.parse_args()

    t0 = time.perf_counter()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X, y = get_dataset(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    if args.label_noise:
        rng = np.random.default_rng(args.seed)
        y_train = y_train.copy()
        classes = np.unique(y_train)
        n_flip = int(round(len(y_train) * args.label_noise))
        if n_flip > 0:
            idxs = rng.choice(len(y_train), size=n_flip, replace=False)
            for idx in idxs:
                alternatives = classes[classes != y_train[idx]]
                y_train[idx] = rng.choice(alternatives)
    clf = build_model(args)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    runtime = time.perf_counter() - t0
    out = {
        "status": "success",
        "dataset": args.dataset,
        "model": args.model,
        "params": vars(args),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="weighted")),
        "runtime_seconds": round(runtime, 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_label_noise_flips": int(round(len(y_train) * args.label_noise)) if args.label_noise else 0,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
