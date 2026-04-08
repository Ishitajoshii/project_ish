"""Regression checks for the production tabular control agent."""

import pytest

from server.baselines import run_heuristic_baseline, run_random_baseline
from server.environment import CircuitEnvironment
from server.policy_agent import (
    AGENT_NAME,
    TabularValueIterationAgent,
    run_policy_episode,
)
from server.task_loader import load_tasks

EXPECTED_POLICY_SCORES = {
    "hp_500hz_budget": 0.8316742247657158,
    "lp_10khz_budget": 0.8502600678683305,
    "lp_1khz_budget": 0.8359742603796119,
    "lp_2khz_low_cost": 0.852065681655602,
}


def test_policy_agent_name_is_stable():
    assert AGENT_NAME == "tabular-value-iteration"


def test_policy_agent_matches_frozen_optimal_scores():
    tasks = load_tasks()
    agent = TabularValueIterationAgent(tasks)

    for task_id, expected_score in EXPECTED_POLICY_SCORES.items():
        result = run_policy_episode(CircuitEnvironment({task_id: tasks[task_id]}), task_id, agent)
        assert result["score"] == pytest.approx(expected_score)
        assert agent.peak_reward_for_task(task_id) == pytest.approx(expected_score)
        assert result["success"] is True


def test_policy_agent_outperforms_random_and_heuristic_baselines():
    tasks = load_tasks()
    agent = TabularValueIterationAgent(tasks)

    for task_id, task in tasks.items():
        policy_result = run_policy_episode(CircuitEnvironment({task_id: task}), task_id, agent)
        heuristic_result = run_heuristic_baseline(CircuitEnvironment({task_id: task}), task_id)
        random_result = run_random_baseline(CircuitEnvironment({task_id: task}), task_id, seed=7)

        assert policy_result["score"] >= heuristic_result["score"]
        assert policy_result["score"] >= random_result["score"]


def test_policy_agent_returns_deterministic_action_path():
    tasks = load_tasks()
    agent = TabularValueIterationAgent(tasks)

    assert agent.peak_path_for_task("lp_1khz_budget") == [
        "c_up",
        "c_up",
        "c_up",
        "r_down",
        "c_up",
        "r_down",
        "c_up",
        "r_down",
    ]
    assert agent.peak_path_for_task("lp_2khz_low_cost") == [
        "r_down",
        "r_down",
        "r_down",
        "r_down",
        "r_down",
        "r_down",
    ]
