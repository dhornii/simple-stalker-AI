WALK_STEP, RUN_STEP = 0.24, 0.12          # seconds per cell
WALK_RADIUS, RUN_RADIUS = 3, 7            # noise radius in cells
NOISE_MEMORY = 0.7                        # how long after a step we count as "noisy"

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