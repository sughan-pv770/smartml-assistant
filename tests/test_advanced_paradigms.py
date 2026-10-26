import pytest
import pandas as pd
import numpy as np
from smartml.core.semi_supervised import SemiSupervisedTrainer
from smartml.core.reinforcement import TabularBanditOptimizer
from smartml.core.self_supervised import TabularSelfSupervisedLearner
from smartml.assistant import Assistant


def test_semi_supervised_trainer():
    # Create dataset with 30% labeled, 70% unlabeled (-1)
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n) * 2,
        "feat3": np.random.randn(n) + 1
    })
    # true binary target based on feat1 > 0
    true_y = (X["feat1"] > 0).astype(int).values
    y = true_y.copy()
    # mask 70% as -1
    mask = np.random.rand(n) > 0.3
    y[mask] = -1

    trainer = SemiSupervisedTrainer(algorithm="self_training")
    res = trainer.fit(X, y)

    assert res.n_pseudo_labeled >= 0
    assert len(res.transduced_labels) == n
    assert res.n_unlabeled > 0
    assert "SemiSupervised" in res.narrative or "SelfTraining" in res.narrative


def test_bandit_optimizer():
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, size=n),
        "income": np.random.uniform(20, 100, size=n),
        "treatment": np.random.choice(["ad_A", "ad_B", "ad_C"], size=n),
        "clicks": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })

    bandit = TabularBanditOptimizer(strategy="linucb")
    res = bandit.optimize(df, action_col="treatment", reward_col="clicks", context_cols=["age", "income"])

    assert res.n_rounds == n
    assert res.n_arms == 3
    assert res.cumulative_reward >= 0
    assert res.best_arm in ["ad_A", "ad_B", "ad_C"]
    assert "bandit" in res.narrative.lower() or "linucb" in res.narrative.lower()


def test_self_supervised_learner():
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n) * 3,
        "cat1": np.random.choice(["low", "med", "high"], size=n)
    })

    learner = TabularSelfSupervisedLearner(embedding_dim=4, max_iter=20)
    res = learner.fit_transform(df)

    assert res.embeddings_df.shape == (n, 4)
    assert res.reconstruction_mse >= 0
    assert res.n_features_in > 0
    assert "embedding" in res.narrative.lower() or "autoencoder" in res.narrative.lower()


def test_assistant_advanced_paradigms():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "target": [0, 1] * 10 + [-1] * 40,
        "action": np.random.choice(["A", "B"], size=n),
        "reward": np.random.choice([0, 1], size=n)
    })

    ast = Assistant(df)

    # 1. Semi-supervised
    resp_semi = ast.ask("semi-supervised training with target target")
    assert "Semi-Supervised" in resp_semi or "pseudo-labeled" in resp_semi

    # 2. Reinforcement / Bandit
    resp_rl = ast.ask("optimize decisions using bandit with action action and reward reward")
    assert "Bandit" in resp_rl or "Cumulative reward" in resp_rl

    # 3. Self-supervised
    resp_ssl = ast.ask("self-supervised learning on dataset")
    assert "Self-Supervised" in resp_ssl or "embeddings" in resp_ssl
