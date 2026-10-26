"""
smartml.core.reinforcement
==========================
Tabular Contextual Bandits and Multi-Armed Bandit (MAB) optimization.
Used for offline tabular decision optimization, pricing, recommendations, and A/B testing.

Supported strategies:
1. LinUCB (Linear Upper Confidence Bound for contextual bandits with feature vectors)
2. EpsilonGreedyBandit (Exploration-exploitation balance)
3. ThompsonSamplingBandit (Bayesian posterior reward sampling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BanditResult:
    strategy: str
    n_rounds: int
    n_arms: int
    arm_names: list[str]
    cumulative_reward: float
    average_reward: float
    arm_pull_counts: dict[str, int]
    best_arm: str
    best_arm_expected_reward: float
    narrative: str
    steps: list[str] = field(default_factory=list)


class TabularBanditOptimizer:
    """Offline Contextual & Multi-Armed Bandit optimizer for decision-making.

    Args:
        strategy: 'linucb', 'thompson', 'epsilon_greedy', or 'auto'.
        epsilon: Exploration rate for epsilon-greedy (default 0.1).
        alpha: Exploration confidence parameter for LinUCB (default 1.0).
        random_state: Reproducibility seed.
    """

    def __init__(
        self,
        strategy: str = "auto",
        epsilon: float = 0.1,
        alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.strategy = strategy.lower()
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_state = random_state

    def optimize(
        self,
        df: pd.DataFrame,
        action_col: str,
        reward_col: str,
        context_cols: Optional[list[str]] = None,
    ) -> BanditResult:
        steps = []
        rng = np.random.RandomState(self.random_state)

        if action_col not in df.columns:
            raise KeyError(f"Action column '{action_col}' not found.")
        if reward_col not in df.columns:
            raise KeyError(f"Reward column '{reward_col}' not found.")

        unique_arms = [str(a) for a in df[action_col].dropna().unique()]
        n_arms = len(unique_arms)
        if n_arms < 2:
            raise ValueError("Bandit optimization requires at least 2 distinct actions/arms.")

        arm_to_idx = {name: i for i, name in enumerate(unique_arms)}
        idx_to_arm = {i: name for i, name in enumerate(unique_arms)}

        actions = df[action_col].astype(str).map(arm_to_idx).fillna(0).astype(int).values
        rewards = pd.to_numeric(df[reward_col], errors="coerce").fillna(0.0).values

        n_rounds = len(df)
        arm_counts = np.zeros(n_arms, dtype=int)
        arm_rewards = np.zeros(n_arms, dtype=float)

        steps.append(f"Loaded {n_rounds} tabular decision rounds across {n_arms} actions: {unique_arms}.")

        strat = self.strategy
        if strat in ("auto", "linucb") and context_cols:
            # Contextual Bandit (LinUCB)
            num_context = df[context_cols].select_dtypes(include=[np.number]).fillna(0).values
            d = num_context.shape[1] + 1  # add intercept
            A = [np.identity(d) for _ in range(n_arms)]
            b = [np.zeros((d, 1)) for _ in range(n_arms)]

            cum_reward = 0.0
            pulled_arms = []

            for t in range(n_rounds):
                x = np.append(num_context[t], 1.0).reshape(-1, 1)
                observed_action = actions[t]
                observed_reward = rewards[t]

                p = np.zeros(n_arms)
                for a in range(n_arms):
                    A_inv = np.linalg.inv(A[a])
                    theta_a = np.dot(A_inv, b[a])
                    confidence_interval = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))[0, 0]
                    p[a] = np.dot(theta_a.T, x)[0, 0] + confidence_interval

                chosen_arm = int(np.argmax(p))
                pulled_arms.append(chosen_arm)
                arm_counts[chosen_arm] += 1

                # Update policy with observed reward when policy agrees with logged action
                if chosen_arm == observed_action or np.isnan(observed_action):
                    A[chosen_arm] += np.dot(x, x.T)
                    b[chosen_arm] += observed_reward * x
                    cum_reward += observed_reward
                    arm_rewards[chosen_arm] += observed_reward

            algo_name = "LinUCB (Contextual Bandit)"
            steps.append(f"Trained LinUCB on {d-1} context features with exploration confidence alpha={self.alpha}.")

        elif strat == "thompson":
            # Thompson Sampling (Beta-Bernoulli for bounded rewards or Gaussian)
            algo_name = "Thompson Sampling Bandit"
            successes = np.ones(n_arms)
            failures = np.ones(n_arms)
            cum_reward = 0.0

            for t in range(n_rounds):
                # Sample from Beta distribution
                samples = [rng.beta(successes[a], failures[a]) for a in range(n_arms)]
                chosen_arm = int(np.argmax(samples))
                arm_counts[chosen_arm] += 1

                r = rewards[t]
                cum_reward += r
                arm_rewards[chosen_arm] += r

                # Update counts
                if r > 0.5:
                    successes[chosen_arm] += 1
                else:
                    failures[chosen_arm] += 1

            steps.append("Executed Thompson Sampling with Beta posterior updates.")

        else:
            # Epsilon-Greedy Bandit
            algo_name = f"Epsilon-Greedy Bandit (eps={self.epsilon})"
            cum_reward = 0.0
            for t in range(n_rounds):
                if rng.rand() < self.epsilon:
                    chosen_arm = rng.randint(n_arms)
                else:
                    means = np.divide(arm_rewards, np.maximum(arm_counts, 1))
                    chosen_arm = int(np.argmax(means))

                arm_counts[chosen_arm] += 1
                r = rewards[t]
                cum_reward += r
                arm_rewards[chosen_arm] += r

            steps.append(f"Executed Epsilon-Greedy policy (exploration rate = {self.epsilon}).")

        avg_reward = float(cum_reward / max(n_rounds, 1))
        pull_dict = {idx_to_arm[i]: int(arm_counts[i]) for i in range(n_arms)}

        arm_means = np.divide(arm_rewards, np.maximum(arm_counts, 1))
        best_arm_idx = int(np.argmax(arm_means))
        best_arm_name = idx_to_arm[best_arm_idx]
        best_arm_mean = float(arm_means[best_arm_idx])

        narrative = (
            f"{algo_name} evaluated {n_rounds} rounds. Best action/arm: '{best_arm_name}' "
            f"yielding estimated average reward of {best_arm_mean:.4f}. "
            f"Overall policy average reward: {avg_reward:.4f}."
        )

        return BanditResult(
            strategy=algo_name,
            n_rounds=n_rounds,
            n_arms=n_arms,
            arm_names=unique_arms,
            cumulative_reward=round(cum_reward, 4),
            average_reward=round(avg_reward, 4),
            arm_pull_counts=pull_dict,
            best_arm=best_arm_name,
            best_arm_expected_reward=round(best_arm_mean, 4),
            narrative=narrative,
            steps=steps,
        )
