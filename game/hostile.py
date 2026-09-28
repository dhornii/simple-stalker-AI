import random
import math
from collections import namedtuple

from world import World
from player import Player

PATROL_NODES = [
    (5, 2),
    (17, 2),
    (12, 7),
    (17, 9),
    (16, 17),
    (11, 13),
    (4, 17),
    (3, 9)
]

# Bayesian belief model
DIFFUSE_RATE = 0.30      # motion model: player may have moved to a neighbour
CLUE_OUTLIER = 0.02      # likelihood floor (clues can be wrong)
NOT_SEEN_LIK = 0.03      # P(not seeing player | player is in a cell I can see)
DIST_PENALTY = 0.08      # investigation target = belief / (1 + k * distance)

# Director
DIRECTOR_SIGMA = {
    "walk": 4.0, "run": 3.0
}
DIRECTOR_COOLDOWN = {
    "walk": 4.5, "run": 2.0
}

# Alien
STEP_TIME = {
    "PATROL": 0.34,
    "INVESTIGATE": 0.26,
    "HUNT": 0.15
}
VISION_RANGE = 7
LOSE_SIGHT_TIMEOUT = 2.5
INVESTIGATE_TIMEOUT = 15.0                # give up if no clue for this long

Clue = namedtuple("Clue", "pos sigma source time")

class Director:

    def __init__(self, world:World):
        self.world = world
        self.cooldown = 0.0

    def leak_position(self, true_pos:tuple, sigma:float):

        for i in range(50):

            x = round(random.gauss(true_pos[0], sigma))
            y = round(random.gauss(true_pos[1], sigma))

            if self.world.walkable(x, y):
                return (x, y)

        return true_pos

    def update(self, dt:float, player:Player, time_now:float):

        self.cooldown = max(0.0, self.cooldown - dt)

        if self.cooldown > 0 or not player.noise_interlude(time_now):
            return None                  

        mode = "run" if player.last_run else "walk"
        self.cooldown = DIRECTOR_COOLDOWN[mode]
        sigma = DIRECTOR_SIGMA[mode]

        return Clue(self.leak_position(player.pos, sigma), sigma, "director", time_now)

class Alien:

    def __init__(self, world, log):

        self.world = world
        self.log = log
        self.pos = world.alien_start
        self.state = "PATROL"
        self.patrol_nodes = PATROL_NODES
        self.node_idx = 0
        self.path = []
        self.target = None
        self.timer = 0.0
        self.last_seen = None
        self.lost_time = 0.0
        self.last_clue = None
        self.last_clue_time = -999.0
        self.sees_player = False
        self.belief = {}

        self.reset_uniform()

    # ---------------- Bayesian belief map ----------------

    def reset_uniform(self):

        p = 1.0 / len(self.world.walkable_tiles)
        self.belief = {c: p for c in self.world.walkable_tiles}

    def _normalize(self):

        s = sum(self.belief.values())

        if s < 1e-12:
            self.reset_uniform()
            return

        for c in self.belief:
            self.belief[c] /= s

    def diffuse(self, rate=DIFFUSE_RATE):

        """Motion model / prediction step: the player may have moved."""
        new = {c: b * (1 - rate) for c, b in self.belief.items()}
        for c, b in self.belief.items():

            nb = self.world.walkable_neighbors(c)

            if not nb:
                new[c] += b * rate
                continue

            share = b * rate / len(nb)

            for n in nb:
                new[n] += share

        self.belief = new

    def bayes_clue(self, pos, sigma):

        """
        Update with a noisy clue z = pos:

            P(c | z)  ∝  P(z | c) * P(c)

        with a Gaussian likelihood (plus an outlier floor) around the clue.
        """

        two_s2 = 2.0 * sigma * sigma

        for c in self.belief:

            d2 = (c[0] - pos[0]) ** 2 + (c[1] - pos[1]) ** 2
            self.belief[c] *= CLUE_OUTLIER + math.exp(-d2 / two_s2)

        self._normalize()

    def bayes_not_seen(self, visible):

        """Negative evidence: 'I look at these cells and see nothing'."""

        for c in visible:

            if c in self.belief:
                self.belief[c] *= NOT_SEEN_LIK

        self._normalize()

    # ---------------- state handling ----------------

    def set_state(self, new, now=0.0):

        if new == self.state:
            return

        self.log(f"Alien: {self.state} -> {new}")
        self.state = new
        self.path, self.target = [], None

        if new == "PATROL":

            self.reset_uniform()
            dist = self.world.bfs_dist(self.pos)
            self.node_idx = min(range(len(self.nodes)), key=lambda i: dist.get(self.nodes[i], 1e9))

        elif new == "INVESTIGATE":
            
            self.last_clue_time = now

    def hear_clue(self, clue, now):

        if self.state == "HUNT":
            return                            # already has eyes on the player

        self.last_clue, self.last_clue_time = clue, now

        if self.state == "PATROL":
            self.set_state("INVESTIGATE", now)

        self.bayes_clue(clue.pos, clue.sigma)

        self.log(f"Heard {clue.source} clue (sigma={clue.sigma:.1f})")

    # ---------------- main update ----------------

    def update(self, dt, player, now):

        self.sees_player = self.world.visible(self.pos, player.pos, VISION_RANGE)

        if self.sees_player:
            self.last_seen, self.lost_time = player.pos, 0.0
            self.set_state("HUNT", now)

        self.timer -= dt

        if self.timer > 0:
            return

        self.timer = STEP_TIME[self.state]

        if self.state == "PATROL":
            self._tick_patrol()

        elif self.state == "INVESTIGATE":
            self._tick_investigate(now)

        else:

            self._tick_hunt(player, now)

    def _step_along(self, path):

        self.path = path or []

        if path:

            self.pos = path[0]

    def _tick_patrol(self):

        if self.pos == self.patrol_nodes[self.node_idx]:

            self.node_idx = (self.node_idx + 1) % len(self.patrol_nodes)

        self._step_along(self.world.bfs_path(self.pos, self.patrol_nodes[self.node_idx]))

    def _tick_investigate(self, now):

        self.diffuse()

        self.bayes_not_seen(self.world.visible_cells(self.pos, VISION_RANGE))

        if now - self.last_clue_time > INVESTIGATE_TIMEOUT:

            self.log("Alien lost the trail")

            self.set_state("PATROL", now)

            return

        target = self._choose_target()

        self._step_along(self.world.bfs_path(self.pos, target))

    def _choose_target(self):

        dist = self.world.bfs_dist(self.pos)



        def score(c):

            return self.belief[c] / (1 + DIST_PENALTY * dist[c]) if c in dist else 0.0



        best = max(self.belief, key=score)

        # hysteresis: don't flip-flop between similar targets

        if self.target and score(self.target) * 1.25 >= score(best):

            best = self.target

        self.target = best

        return best

    def _tick_hunt(self, player, now):

        if self.sees_player:

            self.reset_uniform()
            self.bayes_clue(player.pos, 0.7)      # belief collapses on the player
            goal = player.pos

        else:

            self.lost_time += STEP_TIME["HUNT"]
            self.diffuse()
            goal = self.last_seen

            if self.pos == goal or self.lost_time > LOSE_SIGHT_TIMEOUT:
                self.set_state("INVESTIGATE", now)
                return

        self._step_along(self.world.bfs_path(self.pos, goal))