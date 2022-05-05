import pomdp_py
import random
import math
import numpy as np
import sys
import copy

EPSILON = 1e-9

class BottlePose():
    def __init__(self, x, y, region):
        self.pos = np.array([x, y])
        if not (region == 'table 1' or region == 'table 2' or region == 'table 3' or region == 'hand'):
            return ValueError("Invalid region {}".format(region))
        self.region = region

    def __hash__(self):
        return hash((self.pos, self.region))

    def __str__(self):
        return "pos: ({}, {}), region: {}".format(self.pos[0], self.pos[1], self.region)

class State(pomdp_py.State):
    def __init__(self, position, b1, b2):
        """
        position: x, y coordinate of spot in world frame
        b1, b2: BottlePose
        """
        self.position = position
        self.b1 = b1
        self.b2 = b2

    def __hash__(self):
        return hash((self.position, self.b1, self.b2))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position \
                   and self.b1 == other.b1 \
                   and self.b2 == other.b2
        else:
            return False

   def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s | %s | %s)" % (str(self.position), str(self.b1), str(self.b2))


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
    TABLE1 = 'table 1'  #todo what form do we want the data here to be to call actions?
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

TABLE_DIM = (300, 300)
PICK_TOLERANCE = 4
class PickAction(Action):
    def __init__(self, x, y):
        if x > TABLE_DIM[0] or x < 0 or y > TABLE_DIM[1] or y < 0:
            raise ValueError("Invalid pick location ({}, {})".format(x, y))
        self.pick_pos = np.array([x, y])
        super().__init__("pick-{}-{}".format(x, y))



class ISMAction(Action):
    def __init__(self):
        super().__init__("look")


class Observation(pomdp_py.Observation):
    def __init__(self, b1, b2):
        """
        b1, b2: pose of bottle (might be None)
        """
        self.b1 = b1
        self.b2 = b2
    def __hash__(self):
        return hash((self.b1, self.b2))
    def __str__(self):
        return str("({}, {})".format(self.b1, self.b2))
    def __repr__(self):
        return "Observation(%s)" % str(self.quality)

class RSTransitionModel(pomdp_py.TransitionModel):

    """ The model is deterministic """

    def __init__(self):
        pass


    def probability(self, next_state, state, action, normalized=False, **kwargs):
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, state, action):
        next_position = state.position
        next_b1 = state.b1
        next_b2 = state.b2
        if action == MoveT1:
            next_position = 'table 1'
        elif action == MoveT2:
            next_position = 'table 2'
        elif action == MoveT3:
            next_position = 'table 3'

        elif isinstance(action, PickAction):
            if state.b1.region == "hand" or state.b2.region == "hand":
                pass
            elif (state.b1.region == state.position and
                  np.linalg.norm(action.pick_pos - state.b1.pos) <= PICK_TOLERANCE):
                next_b1.table = "hand"
            elif (state.b2.region == state.position and
                  np.linalg.norm(action.pick_pos - state.b2.pos) <= PICK_TOLERANCE):
                next_b2.table = "hand"

        elif isinstance(action, ISMAction):
            pass
        return State(next_position, next_b1, next_b1)


