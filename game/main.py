"""
STALKER AI  -  inspired by the Alien: Isolation "Director + Alien" split
=========================================================================

Two agents cooperate against the player:

  * DIRECTOR  - omniscient. Knows the player's true position but is NOT allowed
                to tell the Alien directly. It can only leak a *vague clue*
                (noisy coordinate) whenever the player makes noise.
  * ALIEN     - blind to the player's real position. It keeps a belief map
                P(player is in cell c) and updates it with Bayes' rule:

                    posterior(c)  ∝  likelihood(observation | c) * prior(c)

Alien state machine
  PATROL       walk between fixed nodes (BFS shortest paths).

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
"""

import math

from world import World, Generator
from player import Player
from hostile import Director, Alien, Clue

import pygame
import asyncio

# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(self):
        self.world = World()
        self.time = 0.0
        self.status = "playing" # playing | caught | escaped
        self.messages = []
        self.noise_events = []  # for visual rings
        self.player = Player(self.world.player_start)
        self.director = Director(self.world)
        self.alien = Alien(self.world, self.log)

    def log(self, msg):
        self.messages.append(msg)
        self.messages = self.messages[-9:]

    def emit_noise(self, pos, radius, kind):
        self.noise_events.append({"pos": pos, "radius": radius, "ttl": 0.6,
                                  "max": 0.6, "kind": kind})
        if self.alien.state != "HUNT" and math.dist(pos, self.alien.pos) <= radius:
            sigma = 1.0 if kind == "generator" else 1.5    # sound localisation error
            heard = self.director.leak_position(pos, sigma)
            self.alien.hear_clue(Clue(heard, sigma, f"{kind} noise", self.time), self.time)

    def interact(self):
        px, py = self.player.pos
        for g in self.world.generators:
            if abs(g.pos[0] - px) + abs(g.pos[1] - py) == 1 and g.ready():
                g.trigger()
                self.log("You triggered a generator")
                self.emit_noise(g.pos, Generator.GEN_RADIUS, "generator")
                return

    def step(self, dt, direction=None, running=False, interact=False):
        if self.status != "playing":
            return
        self.time += dt

        if self.player.tick_update(dt, direction, running, self.world, self.time):
            self.emit_noise(self.player.pos, self.player.set_noise_radius(), "footstep")
        if interact:
            self.interact()
        for g in self.world.generators:
            if g.tick_update(dt):
                self.log("A generator malfunctions!")
                self.emit_noise(g.pos, Generator.GEN_RADIUS, "generator")

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
COLORS = {
    "floor": (28, 30, 36), "grid": (36, 38, 46), "wall": (72, 76, 94),
    "obstacle": (120, 96, 62), "gen": (220, 180, 40), "exit": (40, 190, 100),
    "player": (70, 150, 255), "player_run": (140, 200, 255),
    "alien": (230, 50, 50), "clue": (255, 165, 0), "text": (220, 220, 225),
    "dim": (140, 140, 150), "hud": (18, 19, 24),
}
STATE_COLOR = {"PATROL": (90, 200, 120), "INVESTIGATE": (255, 190, 60), "HUNT": (255, 70, 70)}

class Game:
    HUD_W = 330
    WIN_W, WIN_H = World.MAP_W * World.CELL_SIZE + HUD_W, World.MAP_H * World.CELL_SIZE
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption("Stalker AI - Bayesian Alien")
        self.font = pygame.font.Font(None, 19)
        self.big = pygame.font.Font(None, 56)
        self.clock = pygame.time.Clock()
        self.show_belief = True
        self.sim = Simulation()

        # 1. Stiker Player (Tikus)
        self.img_player = pygame.image.load('assets/tikus.png').convert_alpha()
        self.img_player = pygame.transform.scale(self.img_player, (World.CELL_SIZE, World.CELL_SIZE))

        # 2. Stiker Alien (Kucing - 3 Ekspresi)
        self.img_cat_patrol = pygame.image.load('assets/kucing_patrol.png').convert_alpha()
        self.img_cat_patrol = pygame.transform.scale(self.img_cat_patrol, (World.CELL_SIZE, World.CELL_SIZE))

        self.img_cat_invest = pygame.image.load('assets/kucing_invest.png').convert_alpha()
        self.img_cat_invest = pygame.transform.scale(self.img_cat_invest, (World.CELL_SIZE, World.CELL_SIZE))

        self.img_cat_hunt = pygame.image.load('assets/kucing_hunt.png').convert_alpha()
        self.img_cat_hunt = pygame.transform.scale(self.img_cat_hunt, (World.CELL_SIZE, World.CELL_SIZE))

        # gambar tembok
        self.img_wall = pygame.image.load('assets/stone.png').convert()
        self.img_wall = pygame.transform.scale(self.img_wall, (World.CELL_SIZE, World.CELL_SIZE))

    @staticmethod
    def rect(c):
        return pygame.Rect(c[0] * World.CELL_SIZE, c[1] * World.CELL_SIZE, World.CELL_SIZE, World.CELL_SIZE)

    @staticmethod
    def center(c):
        return (c[0] * World.CELL_SIZE + World.CELL_SIZE // 2, c[1] * World.CELL_SIZE + World.CELL_SIZE // 2)

    def text(self, s, x, y, color=None):
        self.screen.blit(self.font.render(s, True, color or COLORS["text"]), (x, y))

    # ---- drawing
    def draw_world(self):
        w = self.sim.world
        for y in range(World.MAP_H):
            for x in range(World.MAP_W):
                r = self.rect((x, y))
                t = w.tiles[y][x]
                if t == "#":
                    self.screen.blit(self.img_wall, (x * World.CELL_SIZE, y * World.CELL_SIZE))
                else:
                    pygame.draw.rect(self.screen, COLORS["floor"], r)
                    pygame.draw.rect(self.screen, COLORS["grid"], r, 1)
                if t == "o":
                    pygame.draw.rect(self.screen, COLORS["obstacle"], r.inflate(-8, -8), border_radius=4)
                elif t == "E":
                    pygame.draw.rect(self.screen, COLORS["exit"], r.inflate(-4, -4), border_radius=4)
                    self.text("EX", r.x + 7, r.y + 9, (10, 40, 20))
        for g in w.generators:
            r = self.rect(g.pos).inflate(-6, -6)
            col = COLORS["gen"] if g.ready() else (110, 90, 30)
            pygame.draw.rect(self.screen, col, r, border_radius=6)
            self.text("N", r.x + 6, r.y + 4, (40, 30, 0))

    def draw_belief(self):
        a = self.sim.alien
        if not self.show_belief or a.state == "PATROL":
            return
        peak = max(a.belief.values())
        ov = pygame.Surface((World.MAP_W * World.CELL_SIZE, World.MAP_H * World.CELL_SIZE), pygame.SRCALPHA)
        for c, b in a.belief.items():
            v = b / peak
            if v > 0.02:
                pygame.draw.rect(ov, (255, 60, 60, int(190 * v ** 0.7)), self.rect(c))
        self.screen.blit(ov, (0, 0))

    def draw_effects(self):
        fx = pygame.Surface((World.MAP_W * World.CELL_SIZE, World.MAP_H * World.CELL_SIZE), pygame.SRCALPHA)
        for e in self.sim.noise_events:
            k = e["ttl"] / e["max"]
            col = (255, 220, 80) if e["kind"] == "generator" else (120, 190, 255)
            pygame.draw.circle(fx, (*col, int(200 * k)), self.center(e["pos"]),
                               int(e["radius"] * World.CELL_SIZE * (1.15 - 0.15 * k)), 2)
        a = self.sim.alien
        if a.last_clue and self.sim.time - a.last_clue.time < 6 and a.state != "HUNT":
            c = self.center(a.last_clue.pos)
            age = self.sim.time - a.last_clue.time
            pygame.draw.circle(fx, (*COLORS["clue"], int(200 * (1 - age / 6))), c,
                               int(a.last_clue.sigma * World.CELL_SIZE), 2)
            pygame.draw.line(fx, COLORS["clue"], (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), 3)
            pygame.draw.line(fx, COLORS["clue"], (c[0] - 6, c[1] + 6), (c[0] + 6, c[1] - 6), 3)
        self.screen.blit(fx, (0, 0))
        if a.path and self.show_belief:
            pts = [self.center(a.pos)] + [self.center(p) for p in a.path]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, STATE_COLOR[a.state], False, pts, 2)
        if a.sees_player:
            pygame.draw.line(self.screen, (255, 70, 70), self.center(a.pos),
                             self.center(self.sim.player.pos), 1)

    def draw_agents(self):
        p, a = self.sim.player, self.sim.alien
   
        # init rotasi gambar tikus
        if not hasattr(self, 'last_p_pos'):
            self.last_p_pos = (p.pos[0], p.pos[1])
            self.p_angle = 0 # asumsi hadap kanan
            
        # mencari arah gerakan
        dx = p.pos[0] - self.last_p_pos[0]
        dy = p.pos[1] - self.last_p_pos[1]
        
        # sudut putar (Pygame berputar berlawanan arah jarum jam)
        if dx > 0: self.p_angle = 90        # Kanan
        elif dx < 0: self.p_angle = -90     # Kiri
        elif dy > 0: self.p_angle = 0       # Bawah
        elif dy < 0: self.p_angle = 180     # Atas
        
        # simpan posisi saat ini untuk pengecekan berikutnya
        self.last_p_pos = (p.pos[0], p.pos[1])
        
        # putar gambar tikus sesuai sudut
        rotated_tikus = pygame.transform.rotate(self.img_player, self.p_angle)
        
        # 1. Gambar Player (Tikus yang sudah diputar)
        self.screen.blit(rotated_tikus, (p.pos[0] * World.CELL_SIZE, p.pos[1] * World.CELL_SIZE))
        
        # 2. Cek status Kucing
        if a.state == "PATROL":
            current_cat_img = self.img_cat_patrol
        elif a.state == "INVESTIGATE":
            current_cat_img = self.img_cat_invest
        elif a.state == "HUNT":
            current_cat_img = self.img_cat_hunt
        else:
            current_cat_img = self.img_cat_patrol
            
        # 3. Gambar Alien (Kucing)
        self.screen.blit(current_cat_img, (a.pos[0] * World.CELL_SIZE, a.pos[1] * World.CELL_SIZE))

    def draw_hud(self):
        s, a, p = self.sim, self.sim.alien, self.sim.player
        x0 = World.MAP_W * World.CELL_SIZE + 14
        pygame.draw.rect(self.screen, COLORS["hud"], (World.MAP_W * World.CELL_SIZE, 0, self.HUD_W, self.WIN_H))
        y = 12
        self.text("STALKER AI", x0, y, (255, 255, 255)); y += 26
        self.text("Alien state:", x0, y, COLORS["dim"])
        self.text(a.state, x0 + 100, y, STATE_COLOR[a.state]); y += 20
        mode = "IDLE (silent)"
        if p.noise_interlude(s.time):
            mode = f"{'RUN' if p.last_run else 'WALK'} (noise r={p.set_noise_radius})"
        self.text("Player:", x0, y, COLORS["dim"]); self.text(mode, x0 + 100, y); y += 20
        peak_c = max(a.belief, key=a.belief.get)
        self.text("Belief peak:", x0, y, COLORS["dim"])
        self.text(f"{a.belief[peak_c]*100:4.1f}% @ {peak_c}", x0 + 100, y); y += 20
        if a.last_clue:
            self.text("Last clue:", x0, y, COLORS["dim"])
            self.text(f"{a.last_clue.source[:9]} {s.time - a.last_clue.time:4.1f}s ago", x0 + 100, y)
        y += 30

        self.text("CONTROLS", x0, y, COLORS["dim"]); y += 18
        for line in ("Arrows  move (walk = quiet)", "SHIFT   run (loud, fast)",
                     "SPACE/E trigger adjacent generator", "B       belief overlay",
                     "R       restart"):
            self.text(line, x0, y); y += 16
        y += 12
        self.text("LEGEND", x0, y, COLORS["dim"]); y += 18
        for line in ("blue  = you       red = alien", "N = generator (decoy noise)",
                     "o = crate (blocks move, not sight)", "orange X/circle = clue + vagueness",
                     "red glow = alien belief map"):
            self.text(line, x0, y); y += 16
        y += 12
        self.text("EVENT LOG", x0, y, COLORS["dim"]); y += 18
        for m in s.messages:
            self.text(m[:42], x0, y); y += 16

    def draw_overlay(self):
        st = self.sim.status
        if st == "playing":
            return
        shade = pygame.Surface((World.MAP_W * World.CELL_SIZE, World.MAP_H * World.CELL_SIZE), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 170))
        self.screen.blit(shade, (0, 0))
        msg, col = ("CAUGHT!", (255, 80, 80)) if st == "caught" else ("YOU ESCAPED!", (80, 230, 130))
        img = self.big.render(msg, True, col)
        self.screen.blit(img, img.get_rect(center=(World.MAP_W * World.CELL_SIZE // 2, World.MAP_H * World.CELL_SIZE // 2 - 14)))
        sub = self.font.render("Press R to restart", True, COLORS["text"])
        self.screen.blit(sub, sub.get_rect(center=(World.MAP_W * World.CELL_SIZE // 2, World.MAP_H * World.CELL_SIZE // 2 + 28)))

    def draw(self):
        self.screen.fill(COLORS["hud"])
        self.draw_world()
        self.draw_belief()
        self.draw_effects()
        self.draw_agents()
        self.draw_hud()
        self.draw_overlay()
        pygame.display.flip()

    # ---- main loop
    async def run(self):
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
            await asyncio.sleep(0)


if __name__ == "__main__":
    game = Game()
    asyncio.run(game.run())