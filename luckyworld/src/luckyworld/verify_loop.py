"""verify_loop — phase de vérification live : 'explore (world-model) PUIS vérifie avant d'affirmer'.

Après l'exploration guidée par le world-model, on re-lance les méthodes trouvées sur
plusieurs seeds (labels propres) et on applique le Verifier (effet > bruit) : le "best"
single-seed du rapport est-il un vrai gagnant, ou dans le bruit ? Déterministe, un-foolable.
"""
from __future__ import annotations
import warnings
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from .verifier import verify, Verdict

warnings.filterwarnings("ignore", category=ConvergenceWarning)
_DATASETS = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits}


def _build(model: str, seed: int):
    if model == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    if model == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if model == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    if model == "svc":
        return make_pipeline(StandardScaler(), SVC(C=1.0, random_state=seed))
    return None


def verify_from_traces(traces, dataset: str = "breast_cancer", seeds=(0, 1, 2, 3)) -> Verdict | None:
    """Re-lance les méthodes apparues dans les traces sur N seeds, applique le Verifier."""
    models = []
    for t in traces:
        m = getattr(t.proposed_action, "model", None)
        if m and m not in models and _build(m, 0) is not None:
            models.append(m)
    if len(models) < 2:
        return None  # besoin d'≥2 méthodes pour comparer
    X, y = _DATASETS[dataset](return_X_y=True)
    per_method: dict[str, list[float]] = {}
    for m in models:
        accs = []
        for s in seeds:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=s, stratify=y)
            clf = _build(m, s); clf.fit(Xtr, ytr)
            accs.append(round(accuracy_score(yte, clf.predict(Xte)), 4))
        per_method[m] = accs
    return verify(per_method)
