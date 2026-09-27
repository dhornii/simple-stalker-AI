"""
STALKER AI  -  inspired by the Alien: Isolation "Director + Alien" split
=========================================================================

Two agents cooperate against the player:

  * DIRECTOR  - omniscient. Knows the player's true position but is NOT allowed
                to tell the Alien directly. It can only leak a *vague clue*
                (a noisy coordinate) whenever the player makes noise.
  * ALIEN     - blind to the player's real position. It keeps a belief map
                P(player is in cell c) and updates it with Bayes' rule:

                    posterior(c)  ∝  likelihood(observation | c) * prior(c)

Alien state machine
  PATROL       walk between fixed nodes (BFS shortest paths)
  INVESTIGATE  a clue was heard -> Bayes-update the belief map, walk to the most
               promising cell (belief discounted by travel distance). Cells the
               alien looks at and finds empty get *negative evidence*.
  HUNT         line of sight to the player -> pursue directly.
               If sight is lost, go to last seen spot, then INVESTIGATE.

Controls
  Arrow keys   move (walking = small noise radius)
  Hold SHIFT   run  (faster, but big noise radius, more Director clues)
  SPACE / E    trigger an adjacent noise generator (decoy)
  B            toggle belief-map overlay (debug)
  R            restart          ESC  quit

Goal: reach the green EXIT without being caught.

Requires:  pip install pygame
"""

import math
import random
import sys
from collections import deque, namedtuple

import pygame

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GRID = 40
CELL = 40
HUD_W = 330
WIN_W, WIN_H = GRID * CELL + HUD_W, GRID * CELL

# Legend: # wall (blocks move + sight)   o obstacle/crate (blocks move only)
#         N noise generator (blocks move) P player start  A alien start  E exit
MAP = [
    "########################################",
    "#.....#.....o......P.....#.......oo....#",
    "#.....#............o.....#...oo........#",
    "#..oo.....###..N...oo............ooo...#",
    "#..oo.....#...........o..#...oo........#",
    "#.........#..oo..........#........oo...#",
    "####.######.###.###..########..........#",
    "#........#..................#....#..#..#",
    "#..N.....#...oo.........N...#....#.....#",
    "#........#..........#............#..o..#",
    "#..###......###.....####....Noo..#.....#",
    "#..#..........#.....#.......ooo..#..#..#",
    "#.....oo.......o....######.............#",
    "#...........#......................#####",
    "######.####.#######.#######...ooo......#",
    "#........#.....A....#..............#...#",
    "#..oo....#...N......#oo..#....##...#...#",
    "#........#..........#....#....##.......#",
    "#..N.........o...........#............E#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
]

PATROL_NODES = [(5, 2), (17, 2), (12, 7), (17, 9),
                (16, 17), (11, 13), (4, 17), (3, 9)]

# Player
WALK_STEP, RUN_STEP = 0.24, 0.12          # seconds per cell
WALK_RADIUS, RUN_RADIUS = 3, 7            # noise radius in cells
NOISE_MEMORY = 0.7                        # how long after a step we count as "noisy"

# Alien
STEP_TIME = {"PATROL": 0.34, "INVESTIGATE": 0.26, "HUNT": 0.15}
VISION_RANGE = 7
LOSE_SIGHT_TIMEOUT = 2.5
INVESTIGATE_TIMEOUT = 15.0                # give up if no clue for this long

# Bayesian belief model
DIFFUSE_RATE = 0.30      # motion model: player may have moved to a neighbour
CLUE_OUTLIER = 0.02      # likelihood floor (clues can be wrong)
NOT_SEEN_LIK = 0.03      # P(not seeing player | player is in a cell I can see)
DIST_PENALTY = 0.08      # investigation target = belief / (1 + k * distance)

# Director
DIRECTOR_SIGMA = {"walk": 4.0, "run": 3.0}   # vagueness of the clue (cells)
DIRECTOR_COOLDOWN = {"walk": 4.5, "run": 2.0}

# Generators
GEN_RADIUS = 5
GEN_COOLDOWN = 8.0

Clue = namedtuple("Clue", "pos sigma source time")

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #
class Generator:
    def __init__(self, pos):
        self.pos = pos
        self.cooldown = 0.0
        self.auto_timer = random.uniform(15, 35)   # occasionally malfunctions

    def ready(self):
        return self.cooldown <= 0

    def trigger(self):
        self.cooldown = GEN_COOLDOWN
        self.auto_timer = random.uniform(15, 35)

    def update(self, dt):
        """Returns True if the generator fires by itself."""
        self.cooldown = max(0.0, self.cooldown - dt)
        self.auto_timer -= dt
        if self.auto_timer <= 0 and self.ready():
            self.trigger()
            return True
        return False


class World:
    def __init__(self, layout):
        assert len(layout) == GRID and all(len(r) == GRID for r in layout)
        self.tiles = [list(r) for r in layout]
        self.player_start = self.alien_start = self.exit = None
        self.generators = []
        for y in range(GRID):
            for x in range(GRID):
                ch = self.tiles[y][x]
                if ch == "P":
                    self.player_start, self.tiles[y][x] = (x, y), "."
                elif ch == "A":
                    self.alien_start, self.tiles[y][x] = (x, y), "."
                elif ch == "E":
                    self.exit, self.tiles[y][x] = (x, y), "E"
                elif ch == "N":
                    self.generators.append(Generator((x, y)))
        self.cells = [(x, y) for y in range(GRID) for x in range(GRID)
                      if self.walkable(x, y)]

    # -- tile queries
    def in_bounds(self, x, y):
        return 0 <= x < GRID and 0 <= y < GRID

    def walkable(self, x, y):
        return self.in_bounds(x, y) and self.tiles[y][x] not in "#oN"

    def blocks_sight(self, x, y):
        return self.tiles[y][x] == "#"

    def neighbors(self, c):
        x, y = c
        return [(x + dx, y + dy) for dx, dy in DIRS if self.walkable(x + dx, y + dy)]

    # -- line of sight (Bresenham)
    @staticmethod
    def line(a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        pts = []
        while True:
            pts.append((x0, y0))
            if (x0, y0) == (x1, y1):
                return pts
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def los(self, a, b):
        return not any(self.blocks_sight(*p) for p in self.line(a, b)[1:-1])

    def can_see(self, a, b, rng):
        return math.dist(a, b) <= rng and self.los(a, b)

    def visible_cells(self, origin, rng):
        ox, oy = origin
        r = int(rng)
        out = []
        for y in range(oy - r, oy + r + 1):
            for x in range(ox - r, ox + r + 1):
                if self.in_bounds(x, y) and self.walkable(x, y) \
                        and self.can_see(origin, (x, y), rng):
                    out.append((x, y))
        return out

    # -- search
    def bfs_path(self, start, goal):
        """Shortest path as a list of cells (excluding start). None if unreachable."""
        if start == goal:
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self.neighbors(cur):
                if n not in prev:
                    prev[n] = cur
                    if n == goal:
                        path = [n]
                        while prev[path[-1]] != start:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    q.append(n)
        return None

    def bfs_dist(self, start):
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self.neighbors(cur):
                if n not in dist:
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return dist


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #
class Player:
    def __init__(self, pos):
        self.pos = pos
        self.timer = 0.0
        self.last_move = -999.0
        self.last_run = False

    @property
    def noise_radius(self):
        return RUN_RADIUS if self.last_run else WALK_RADIUS

    def noisy_recently(self, now):
        return now - self.last_move < NOISE_MEMORY

    def update(self, dt, direction, running, world, now):
        """Returns True if the player moved this frame."""
        self.timer = max(0.0, self.timer - dt)
        if direction and self.timer <= 0:
            nx, ny = self.pos[0] + direction[0], self.pos[1] + direction[1]
            if world.walkable(nx, ny):
                self.pos = (nx, ny)
                self.timer = RUN_STEP if running else WALK_STEP
                self.last_move = now
                self.last_run = running
                return True
        return False


# --------------------------------------------------------------------------- #
# Director  (omniscient; can only leak vague clues)
# --------------------------------------------------------------------------- #
class Director:
    def __init__(self, world):
        self.world = world
        self.cooldown = 0.0

    def _noisy_position(self, true_pos, sigma):
        for _ in range(50):
            x = round(random.gauss(true_pos[0], sigma))
            y = round(random.gauss(true_pos[1], sigma))
            if self.world.walkable(x, y):
                return (x, y)
        return true_pos

    def update(self, dt, player, now):
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.cooldown > 0 or not player.noisy_recently(now):
            return None                       # silent player = invisible player
        mode = "run" if player.last_run else "walk"
        self.cooldown = DIRECTOR_COOLDOWN[mode]
        sigma = DIRECTOR_SIGMA[mode]
        return Clue(self._noisy_position(player.pos, sigma), sigma, "director", now)


# --------------------------------------------------------------------------- #
# Alien
# --------------------------------------------------------------------------- #
class Alien:
    def __init__(self, world, nodes, log):
        self.world = world
        self.log = log
        self.pos = world.alien_start
        self.state = "PATROL"
        self.nodes = nodes
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
        p = 1.0 / len(self.world.cells)
        self.belief = {c: p for c in self.world.cells}

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
            nb = self.world.neighbors(c)
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
            self.node_idx = min(range(len(self.nodes)),
                                key=lambda i: dist.get(self.nodes[i], 1e9))
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
        self.sees_player = self.world.can_see(self.pos, player.pos, VISION_RANGE)
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
        if self.pos == self.nodes[self.node_idx]:
            self.node_idx = (self.node_idx + 1) % len(self.nodes)
        self._step_along(self.world.bfs_path(self.pos, self.nodes[self.node_idx]))

    def _tick_investigate(self, now):
        self.diffuse()                                            # predict
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


# --------------------------------------------------------------------------- #
# Simulation (no rendering - easy to test)
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(self):
        self.world = World(MAP)
        self.time = 0.0
        self.status = "playing"               # playing | caught | escaped
        self.messages = []
        self.noise_events = []                # for visual rings
        self.player = Player(self.world.player_start)
        self.director = Director(self.world)
        self.alien = Alien(self.world, PATROL_NODES, self.log)

    def log(self, msg):
        self.messages.append(msg)
        self.messages = self.messages[-9:]

    def emit_noise(self, pos, radius, kind):
        self.noise_events.append({"pos": pos, "radius": radius, "ttl": 0.6,
                                  "max": 0.6, "kind": kind})
        if self.alien.state != "HUNT" and math.dist(pos, self.alien.pos) <= radius:
            sigma = 1.0 if kind == "generator" else 1.5    # sound localisation error
            heard = self.director._noisy_position(pos, sigma)
            self.alien.hear_clue(Clue(heard, sigma, f"{kind} noise", self.time), self.time)

    def interact(self):
        px, py = self.player.pos
        for g in self.world.generators:
            if abs(g.pos[0] - px) + abs(g.pos[1] - py) == 1 and g.ready():
                g.trigger()
                self.log("You triggered a generator")
                self.emit_noise(g.pos, GEN_RADIUS, "generator")
                return

    def step(self, dt, direction=None, running=False, interact=False):
        if self.status != "playing":
            return
        self.time += dt

        if self.player.update(dt, direction, running, self.world, self.time):
            self.emit_noise(self.player.pos, self.player.noise_radius, "footstep")
        if interact:
            self.interact()
        for g in self.world.generators:
            if g.update(dt):
                self.log("A generator malfunctions!")
                self.emit_noise(g.pos, GEN_RADIUS, "generator")

        clue = self.director.update(dt, self.player, self.time)
        if clue:
            self.alien.hear_clue(clue, self.time)

        self.alien.update(dt, self.player, self.time)

        for e in self.noise_events:
            e["ttl"] -= dt
        self.noise_events = [e for e in self.noise_events if e["ttl"] > 0]

        ax, ay = self.alien.pos
        px, py = self.player.pos
        if abs(ax - px) + abs(ay - py) <= 1:
            self.status = "caught"
        elif self.player.pos == self.world.exit:
            self.status = "escaped"


# --------------------------------------------------------------------------- #
# Rendering / input
# --------------------------------------------------------------------------- #
COL = {
    "floor": (28, 30, 36), "grid": (36, 38, 46), "wall": (72, 76, 94),
    "obstacle": (120, 96, 62), "gen": (220, 180, 40), "exit": (40, 190, 100),
    "player": (70, 150, 255), "player_run": (140, 200, 255),
    "alien": (230, 50, 50), "clue": (255, 165, 0), "text": (220, 220, 225),
    "dim": (140, 140, 150), "hud": (18, 19, 24),
}
STATE_COL = {"PATROL": (90, 200, 120), "INVESTIGATE": (255, 190, 60), "HUNT": (255, 70, 70)}


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Stalker AI - Bayesian Alien")
        self.font = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 14)
        self.big = pygame.font.SysFont(None, 56)
        self.clock = pygame.time.Clock()
        self.show_belief = True
        self.sim = Simulation()

    @staticmethod
    def rect(c):
        return pygame.Rect(c[0] * CELL, c[1] * CELL, CELL, CELL)

    @staticmethod
    def center(c):
        return (c[0] * CELL + CELL // 2, c[1] * CELL + CELL // 2)

    def text(self, s, x, y, color=None):
        self.screen.blit(self.font.render(s, True, color or COL["text"]), (x, y))

    # ---- drawing
    def draw_world(self):
        w = self.sim.world
        for y in range(GRID):
            for x in range(GRID):
                r = self.rect((x, y))
                t = w.tiles[y][x]
                if t == "#":
                    pygame.draw.rect(self.screen, COL["wall"], r)
                else:
                    pygame.draw.rect(self.screen, COL["floor"], r)
                    pygame.draw.rect(self.screen, COL["grid"], r, 1)
                if t == "o":
                    pygame.draw.rect(self.screen, COL["obstacle"], r.inflate(-8, -8), border_radius=4)
                elif t == "E":
                    pygame.draw.rect(self.screen, COL["exit"], r.inflate(-4, -4), border_radius=4)
                    self.text("EX", r.x + 7, r.y + 9, (10, 40, 20))
        for g in w.generators:
            r = self.rect(g.pos).inflate(-6, -6)
            col = COL["gen"] if g.ready() else (110, 90, 30)
            pygame.draw.rect(self.screen, col, r, border_radius=6)
            self.text("N", r.x + 6, r.y + 4, (40, 30, 0))

    def draw_belief(self):
        a = self.sim.alien
        if not self.show_belief or a.state == "PATROL":
            return
        peak = max(a.belief.values())
        ov = pygame.Surface((GRID * CELL, GRID * CELL), pygame.SRCALPHA)
        for c, b in a.belief.items():
            v = b / peak
            if v > 0.02:
                pygame.draw.rect(ov, (255, 60, 60, int(190 * v ** 0.7)), self.rect(c))
        self.screen.blit(ov, (0, 0))

    def draw_effects(self):
        fx = pygame.Surface((GRID * CELL, GRID * CELL), pygame.SRCALPHA)
        for e in self.sim.noise_events:
            k = e["ttl"] / e["max"]
            col = (255, 220, 80) if e["kind"] == "generator" else (120, 190, 255)
            pygame.draw.circle(fx, (*col, int(200 * k)), self.center(e["pos"]),
                               int(e["radius"] * CELL * (1.15 - 0.15 * k)), 2)
        a = self.sim.alien
        if a.last_clue and self.sim.time - a.last_clue.time < 6 and a.state != "HUNT":
            c = self.center(a.last_clue.pos)
            age = self.sim.time - a.last_clue.time
            pygame.draw.circle(fx, (*COL["clue"], int(200 * (1 - age / 6))), c,
                               int(a.last_clue.sigma * CELL), 2)
            pygame.draw.line(fx, COL["clue"], (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), 3)
            pygame.draw.line(fx, COL["clue"], (c[0] - 6, c[1] + 6), (c[0] + 6, c[1] - 6), 3)
        self.screen.blit(fx, (0, 0))
        if a.path and self.show_belief:
            pts = [self.center(a.pos)] + [self.center(p) for p in a.path]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, STATE_COL[a.state], False, pts, 2)
        if a.sees_player:
            pygame.draw.line(self.screen, (255, 70, 70), self.center(a.pos),
                             self.center(self.sim.player.pos), 1)

    def draw_agents(self):
        p, a = self.sim.player, self.sim.alien
        pygame.draw.circle(self.screen, COL["player_run"] if p.last_run and p.noisy_recently(self.sim.time)
                           else COL["player"], self.center(p.pos), CELL // 2 - 5)
        pygame.draw.circle(self.screen, STATE_COL[a.state], self.center(a.pos), CELL // 2 - 3)
        pygame.draw.circle(self.screen, COL["alien"], self.center(a.pos), CELL // 2 - 8)

    def draw_hud(self):
        s, a, p = self.sim, self.sim.alien, self.sim.player
        x0 = GRID * CELL + 14
        pygame.draw.rect(self.screen, COL["hud"], (GRID * CELL, 0, HUD_W, WIN_H))
        y = 12
        self.text("STALKER AI", x0, y, (255, 255, 255)); y += 26
        self.text("Alien state:", x0, y, COL["dim"])
        self.text(a.state, x0 + 100, y, STATE_COL[a.state]); y += 20
        mode = "IDLE (silent)"
        if p.noisy_recently(s.time):
            mode = f"{'RUN' if p.last_run else 'WALK'} (noise r={p.noise_radius})"
        self.text("Player:", x0, y, COL["dim"]); self.text(mode, x0 + 100, y); y += 20
        peak_c = max(a.belief, key=a.belief.get)
        self.text("Belief peak:", x0, y, COL["dim"])
        self.text(f"{a.belief[peak_c]*100:4.1f}% @ {peak_c}", x0 + 100, y); y += 20
        if a.last_clue:
            self.text("Last clue:", x0, y, COL["dim"])
            self.text(f"{a.last_clue.source[:9]} {s.time - a.last_clue.time:4.1f}s ago", x0 + 100, y)
        y += 30

        self.text("CONTROLS", x0, y, COL["dim"]); y += 18
        for line in ("Arrows  move (walk = quiet)", "SHIFT   run (loud, fast)",
                     "SPACE/E trigger adjacent generator", "B       belief overlay",
                     "R       restart"):
            self.text(line, x0, y); y += 16
        y += 12
        self.text("LEGEND", x0, y, COL["dim"]); y += 18
        for line in ("blue  = you       red = alien", "N = generator (decoy noise)",
                     "o = crate (blocks move, not sight)", "orange X/circle = clue + vagueness",
                     "red glow = alien belief map"):
            self.text(line, x0, y); y += 16
        y += 12
        self.text("EVENT LOG", x0, y, COL["dim"]); y += 18
        for m in s.messages:
            self.text(m[:42], x0, y); y += 16

    def draw_overlay(self):
        st = self.sim.status
        if st == "playing":
            return
        shade = pygame.Surface((GRID * CELL, GRID * CELL), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 170))
        self.screen.blit(shade, (0, 0))
        msg, col = ("CAUGHT!", (255, 80, 80)) if st == "caught" else ("YOU ESCAPED!", (80, 230, 130))
        img = self.big.render(msg, True, col)
        self.screen.blit(img, img.get_rect(center=(GRID * CELL // 2, GRID * CELL // 2 - 14)))
        sub = self.font.render("Press R to restart", True, COL["text"])
        self.screen.blit(sub, sub.get_rect(center=(GRID * CELL // 2, GRID * CELL // 2 + 28)))

    def draw(self):
        self.screen.fill(COL["hud"])
        self.draw_world()
        self.draw_belief()
        self.draw_effects()
        self.draw_agents()
        self.draw_hud()
        self.draw_overlay()
        pygame.display.flip()

    # ---- main loop
    def run(self):
        while True:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            interact = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return
                    elif ev.key == pygame.K_r:
                        self.sim = Simulation()
                    elif ev.key == pygame.K_b:
                        self.show_belief = not self.show_belief
                    elif ev.key in (pygame.K_SPACE, pygame.K_e):
                        interact = True

            keys = pygame.key.get_pressed()
            direction = None
            if keys[pygame.K_LEFT]:
                direction = (-1, 0)
            elif keys[pygame.K_RIGHT]:
                direction = (1, 0)
            elif keys[pygame.K_UP]:
                direction = (0, -1)
            elif keys[pygame.K_DOWN]:
                direction = (0, 1)
            running = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

            self.sim.step(dt, direction, running, interact)
            self.draw()


if __name__ == "__main__":
    Game().run()
    pygame.quit()
    sys.exit()