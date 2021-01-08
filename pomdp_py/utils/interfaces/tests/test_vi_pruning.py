import sys
import pomdp_py
import subprocess
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdp_file
from pomdp_py.utils.interfaces.solvers import vi_pruning
import os
import io


def test_vi_pruning(pomdp_solve_path):
    print("[testing] test_vi_pruning")
    tiger = make_tiger()

    # Building a policy graph
    print("[testing] solving the tiger problem...")
    policy_graph = vi_pruning(tiger.agent, pomdp_solve_path, discount_factor=0.95,
                              options=["-horizon", "100"],
                              remove_generated_files=False)

    assert str(policy_graph.plan(tiger.agent)) == "listen",\
        "Bad solution. Test failed."

    # Plan with the graph for several steps. So we should get high rewards
    # eventually in the tiger domain.
    got_high_reward = False
    for step in range(10):
        true_state = tiger.env.state
        action = policy_graph.plan(tiger.agent)
        observation = tiger.agent.observation_model.sample(true_state, action)
        reward = tiger.env.reward_model.sample(true_state, action, None)
        print("[testing] simulating computed policy graph"\
              "(step=%d, action=%s, observation=%s, reward=%d)" % (step, action, observation, reward))
        # No belief update needed. Just update the policy graph
        policy_graph.update(tiger.agent, action, observation)
        assert reward == -1 or reward == 10, "Reward is negative. Failed."
        if reward == 10:
            got_high_reward = True
    assert got_high_reward, "Should have gotten high reward. Failed."
    print("Pass.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To run test, do: %s <pomdp-solver-path>" % sys.argv[0])
        print("Download pomdp-solve from https://www.pomdp.org/code/")
        exit(1)
    solver_path = sys.argv[1]
    test_vi_pruning(solver_path)
