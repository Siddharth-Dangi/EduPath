"""
data_engine.py
--------------
Core analytics engine for EduPulse. Handles synthetic data generation,
student performance clustering, and pass-probability forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class PerformanceAnalyzer:
    """
    Trains an unsupervised clustering model and a supervised classifier
    to categorise learners and forecast their academic outcomes.
    """

    PROFILE_LABELS = {0: "Struggling", 1: "Developing", 2: "Excelling"}

    def __init__(self, n_clusters: int = 3):
        self.scaler = MinMaxScaler()
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=7, n_init="auto")
        self.classifier = GradientBoostingClassifier(n_estimators=100, random_state=7)

        self.is_ready = False
        self.model_accuracy = 0.0
        self._cluster_profile_map: dict = {}

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def build_sample_dataset(self, n: int = 600) -> pd.DataFrame:
        """Creates a realistic synthetic dataset of student activity metrics."""
        rng = np.random.default_rng(seed=7)

        quiz_avg = rng.uniform(35, 100, n)
        study_hours = rng.uniform(2, 60, n)
        tasks_done = rng.integers(0, 12, n)
        attendance_pct = rng.uniform(50, 100, n)

        # Composite score drives pass/fail with added noise
        composite = (
            quiz_avg * 0.40
            + study_hours * 1.20
            + tasks_done * 4.5
            + attendance_pct * 0.30
        )
        composite = (composite - composite.min()) / (composite.max() - composite.min())
        noise = rng.normal(0, 0.05, n)
        passed = ((composite + noise) > np.median(composite)).astype(int)

        return pd.DataFrame(
            {
                "student_id": range(1, n + 1),
                "quiz_avg": quiz_avg,
                "study_hours": study_hours,
                "tasks_done": tasks_done,
                "attendance_pct": attendance_pct,
                "passed": passed,
            }
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the clustering and classification models on the dataset."""
        feature_cols = ["quiz_avg", "study_hours", "tasks_done", "attendance_pct"]
        X = df[feature_cols].values
        y = df["passed"].values

        X_scaled = self.scaler.fit_transform(X)

        # --- Unsupervised: cluster learners into performance tiers ---
        cluster_ids = self.clusterer.fit_predict(X_scaled)
        df = df.copy()
        df["cluster"] = cluster_ids

        # Derive a stable profile mapping based on mean quiz average per cluster
        means = df.groupby("cluster")["quiz_avg"].mean().sort_values()
        self._cluster_profile_map = {
            cluster: label
            for label, (cluster, _) in zip(
                ["Struggling", "Developing", "Excelling"], means.items()
            )
        }
        df["learner_tier"] = df["cluster"].map(self._cluster_profile_map)

        # --- Supervised: predict pass / fail ---
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.2, random_state=7
        )
        self.classifier.fit(X_tr, y_tr)
        self.model_accuracy = accuracy_score(y_te, self.classifier.predict(X_te))

        self.is_ready = True
        return df

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def evaluate_student(
        self,
        quiz_avg: float,
        study_hours: float,
        tasks_done: int,
        attendance_pct: float,
    ) -> dict:
        """Returns profile tier and pass-probability for a single student."""
        if not self.is_ready:
            raise RuntimeError("Models have not been trained yet. Call fit() first.")

        sample = np.array([[quiz_avg, study_hours, tasks_done, attendance_pct]])
        sample_scaled = self.scaler.transform(sample)

        cluster_id = int(self.clusterer.predict(sample_scaled)[0])
        pass_prob = float(self.classifier.predict_proba(sample_scaled)[0][1])
        pass_pred = bool(self.classifier.predict(sample_scaled)[0])

        tier = self._cluster_profile_map.get(cluster_id, "Developing")

        return {
            "tier": tier,
            "cluster_id": cluster_id,
            "pass_probability": pass_prob,
            "will_pass": pass_pred,
        }
