import sys

import pomdp_py
import random
import math
import numpy as np
# import sys
import copy

EPSILON = 1e-12

TABLE_X = 2
TABLE_Y = 2
TABLE_DIM = np.array([TABLE_X, TABLE_Y])
PICK_TOLERANCE = 0.1


class BottlePose():
    @staticmethod
    def random():
        x, y = np.random.randint(0, TABLE_DIM, size=2)
        # TODO should this include hand
        region = random.sample(["table 1", "table 2", "table 3"], 1)[0]
        # region = random.sample(["table 1", "table 2"], 1)[0]
        return BottlePose(x, y, region)

    @staticmethod
    def get_bottle_regions():
        return ["hand", "table 1", "table 2", "table 3"]
        # return ["hand", "table 1", "table 2"]

    @staticmethod
    def get_all_bottle_poses():
        bottle_xys = set({(x, y) for x in range(TABLE_X) for y in range(TABLE_Y)})
        return [BottlePose(x, y, region) for (x, y) in bottle_xys for region in BottlePose.get_bottle_regions()]

    def __init__(self, x, y, region):
        self.pos = np.array([x, y])
        if not (region == 'table 1' or region == 'table 2' or region == 'table 3' or region == 'hand'):
            return ValueError("Invalid region {}".format(region))
        self.region = region

    def __eq__(self, other):
        if other is None:
            return False
        return np.array_equal(self.pos, other.pos) and self.region == other.region

    def __hash__(self):
        return hash((self.pos[0], self.pos[1], self.region))

    def __str__(self):
        return "region: {}".format(self.region)


class State(pomdp_py.State):
    def __init__(self, position, b1, b1k, terminal=False):
        """
        position: one of "home", "table 1", "table 2", "table 3"
        b1, b2: BottlePose
        """
        self.position = position
        self.b1 = b1
        # self.b2 = b2
        self.b1k = b1k
        # self.b2k = b2k
        self.terminal = terminal
            # self.position == "home" and (self.b1.region == "hand" or self.b2.region == "hand")


    @staticmethod
    def get_all_positions():
        # return ["home", "table 1", "table 2"]
        return ["home", "table 1", "table 2", "table 3"]

    def is_terminal(self):
        # return self.terminal
        return self.position == "home" and self.b1.region == "hand"

    def __hash__(self):
        return hash((self.position, self.b1))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position \
                   and self.b1 == other.b1 \
                   and self.b1k == other.b1k
                   # and self.b2 == other.b2 \
                   # and self.b2k == other.b2k
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s | %s | %s )" % (str(self.position), str(self.b1), str(self.b1k))


class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name


class MoveAction(Action):
    TABLE1 = 'table 1'  # todo what form do we want the dterminalata here to be to call actions?
    TABLE2 = 'table 2'
    TABLE3 = 'table 3'
    HOME = 'home'

    def __init__(self, motion, name):
        if motion not in {MoveAction.TABLE1, MoveAction.TABLE2,
                          MoveAction.TABLE3, MoveAction.HOME}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(name))


MoveT1 = MoveAction(MoveAction.TABLE1, "table 1")
MoveT2 = MoveAction(MoveAction.TABLE2, "table 2")
MoveT3 = MoveAction(MoveAction.TABLE3, "table 3")
MoveHome = MoveAction(MoveAction.HOME, "home")


class PickAction(Action):
    def __init__(self):
        # if x > TABLE_DIM[0] or x < 0 or y > TABLE_DIM[1] or y < 0:
        #     raise ValueError("Invalid pick location ({}, {})".format(x, y))
        # self.pick_pos = np.array([x, y])
        super().__init__("pick")


class ISMAction(Action):
    def __init__(self):
        super().__init__("look")


class Observation(pomdp_py.Observation):
    def __init__(self, b1, hand_full):
        """
        b1, b2: pose of bottle (might be None)
        """
        self.b1 = b1
        # self.b2 = b2
        self.hand_full = hand_full

    def __hash__(self):
        return hash((self.b1, self.hand_full))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.b1 == other.b1 and self.hand_full == other.hand_full
        elif type(other) == str:
            return str(self) == other

    def __str__(self):
        return str("({}, {})".format(self.b1, self.hand_full))

    def __repr__(self):
        return str("({}, {})".format(self.b1, self.hand_full))


class STObservationModel(pomdp_py.ObservationModel):

    # _obs = [Observation(b1, b2)  for b1 in BottlePose.get_all_bottle_poses() + [None] for b2 in BottlePose.get_all_bottle_poses() + [None]]
    # _obs = [o for o in _obs if o.b1 != o.b2]
    # print(len(_obs))
    # @staticmethod
    # def get_all_observations():
    #     return STObservationModel._obs

    def __init__(self):
        pass

    def probability(self, observation, next_state, action):
        if observation != self.sample(next_state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, next_state, action, argmax=False):
        # if np.random.random() < 0.1:
        #     b1 = next_state.b1
        #     b2 = next_state.b2
        #     return Observation(b1, b2)
        hand_full = next_state.b1.region == "hand"
        if isinstance(action, ISMAction):
            position = next_state.position
            b1 = None
            if hand_full:
                return Observation(None, hand_full)

            if next_state.b1.region == position:
                b1 = next_state.b1
            # if next_state.b2.region == position:
            #     b2 = next_state.b2
            return Observation(b1, hand_full)

        return Observation(None, hand_full)

    def argmax(self, next_state, action):
        """Returns the most likely observation"""
        return self.sample(next_state, action, argmax=True)


class STTransitionModel(pomdp_py.TransitionModel):
    """ The model is deterministic """
    # this doesnot curently work for bottles on the same table
    _states = [State(position, b1, b1k)
               for position in State.get_all_positions()
               for b1 in BottlePose.get_all_bottle_poses()
               for b1k in [True, False]]
    print(len(_states))
    _states = [s for s in _states]
    print(len(_states))

    def __init__(self):
        pass

    @staticmethod
    def get_all_states():
        return STTransitionModel._states


    def probability(self, next_state, state, action, normalized=False, **kwargs):
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, state, action):
        if state.is_terminal():
            return state

        next_position = state.position
        next_b1_region = state.b1.region
        next_b1k = state.b1k

        if action == MoveT1:
            next_position = 'table 1'
        elif action == MoveT2:
            next_position = 'table 2'
        elif action == MoveT3:
            next_position = 'table 3'
        elif action == MoveHome:
            next_position = 'home'

        elif isinstance(action, PickAction):
            if state.b1.region == "hand":
                pass
            elif (state.b1.region == state.position and
                  state.b1k):
                next_b1_region = "hand"

        elif isinstance(action, ISMAction):
            if state.b1.region == state.position:
                next_b1k = True



        next_b1 = BottlePose(state.b1.pos[0], state.b1.pos[1], next_b1_region)
        return State(next_position, next_b1, next_b1k)


class STRewardModel(pomdp_py.RewardModel):
    def __init__(self):
        pass

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        if next_state.is_terminal():
            if state.is_terminal():
                return 0
            else:
                if state.b1.region == "hand":
                    return 10
                else:
                    return -10
        if not state.b1.region == "hand" and next_state.b1.region == "hand":
            return 2

        if not next_state.b1.region == "hand" and isinstance(action, PickAction):
            return -2
        # if isinstance(action, ISMAction):
        #     return -.1
        return -1

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        raise NotImplementedError


class STPolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model according to problem description."""

    def __init__(self):
        self._move_actions = {MoveT1, MoveT2, MoveT3, MoveHome}
        self._pick_actions = {PickAction()}
        self._all_actions = self._pick_actions | self._move_actions | {ISMAction()}
        print(self._all_actions)

    def sample(self, state, normalized=False, **kwargs):
        if state.is_terminal():
            return MoveHome

        action_type = self.sample_action_type(state)
        if action_type == "move":
            return random.sample(self.get_possible_moves(state), 1)[0]
        elif action_type == "pick":
            return random.sample(self._pick_actions, 1)[0]
        elif action_type == "ism":
            return ISMAction()

    def sample_action_type(self, state):
        avaiable_actions = self.get_possible_action_types(state)
        return random.sample(avaiable_actions, 1)[0]

    def get_possible_action_types(self, state):
        if state.position == "home" or state.b1.region == "hand":
            return ["move"]
        if not state.b1k:
            return ["move", "ism"]
        else:
            return ["move", "ism", "pick"]

    def get_possible_moves(self, state):
        move_actions = copy.deepcopy(self._move_actions)
        move_actions.remove("move-" + str(state.position))
        return move_actions

    def get_all_actions(self, **kwargs):
        state = kwargs.get("state", None)
        if state is None:
            return self._all_actions
        move_actions = self.get_possible_moves(state)

        if self.get_possible_action_types(state) == ["move"]:
            return move_actions
        return self._pick_actions | move_actions | {ISMAction()}

    # def sample_move_action(self):
    #     return random.sample([MoveAction("table 1"), MoveAction("table 2"), MoveAction("table 3"), MoveAction("home")], 1)[0]

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError

    def rollout(self, state, history=None):
        return self.sample(state)


class SpotTableProblem(pomdp_py.POMDP):

    @staticmethod
    def get_random_init_state():
        """Returns init_state and rock locations for an instance of RockSample(n,k)"""
        b1 = BottlePose.random()
        # b2 = BottlePose.random()

        # while (b2.region == b1.region and
        #        np.linalg.norm(b2.pos - b1.pos) <= PICK_TOLERANCE):
        #     b2 = BottlePose.random()

        return State("home", b1, False)

    def is_terminal(self):
        return self.env.state.is_terminal()

    def print_state(self):
        print(self.env.state)

    def __init__(self, init_state, init_belief):
        agent = pomdp_py.Agent(init_belief,
                               STPolicyModel(),
                               STTransitionModel(),
                               STObservationModel(),
                               STRewardModel())
        env = pomdp_py.Environment(init_state,
                                   STTransitionModel(),
                                   STRewardModel())
        super().__init__(agent, env, name="SpotTableProblem")


def test_planner(spot_table_problem, planner, nsteps=3, discount=0.9):
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    for i in range(nsteps):
        print("==== Step %d ====" % (i + 1))
        action = planner.plan(spot_table_problem.agent)
        # pomdp_py.visual.visualize_pouct_search_tree(rocksample.agent.tree,
        #                                             max_depth=5, anonymize=False)

        true_state = copy.deepcopy(spot_table_problem.env.state)
        env_reward = spot_table_problem.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(spot_table_problem.env.state)

        real_observation = spot_table_problem.env.provide_observation(spot_table_problem.agent.observation_model,
                                                                      action)
        spot_table_problem.agent.update_history(action, real_observation)
        planner.update(spot_table_problem.agent, action, real_observation)
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        # dbg = pomdp_py.TreeDebugger(spot_table_problem.agent.tree)
        # dbg.p()
        # dbg.mbp
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("World:")
        spot_table_problem.print_state()

        if spot_table_problem.is_terminal():
            break
    return total_reward, total_discounted_reward


def init_particles_belief(num_particles, init_state, belief="uniform"):
    particles = []
    for _ in range(num_particles):
        if belief == "uniform":
            state = SpotTableProblem.get_random_init_state()
        elif belief == "groundtruth":
            state = copy.deepcopy(init_state)
        particles.append(state)
    init_belief = pomdp_py.Particles(particles)
    return init_belief


def main():
    init_state = SpotTableProblem.get_random_init_state()
    print(init_state)

    belief = "uniform"
    init_state_copy = copy.deepcopy(init_state)
    # init belief (uniform), represented in particles;
    # We don't factor the state here; We are also not doing any action prior.
    init_belief = init_particles_belief(5000, init_state, belief=belief)

    spot_table_problem = SpotTableProblem(init_state, init_belief)
    spot_table_problem.print_state()

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=4, discount_factor=0.99,
                           num_sims=10000, exploration_const=1,
                           rollout_policy=spot_table_problem.agent.policy_model)
    # valueit = pomdp_py.ValueIteration(2, epsilon=1e-3)
    tt, ttd = test_planner(spot_table_problem, pomcp, nsteps=100, discount=0.99)


if __name__ == '__main__':
    main()
