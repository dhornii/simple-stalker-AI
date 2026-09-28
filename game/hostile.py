import random
import math
from collections import namedtuple

from pathlib import Path
import yaml

from world import World
from player import Player

default = {
    # Director
    'sigma': {'walk': 5.5, 'run': 4.0},
    'cooldown': {'walk': 5.0, 'run': 3.0},

    # Alien
    'patrol_nodes': [(5, 2), (17, 2), (12, 7), (17, 9), (16, 17), (11, 13), (4, 17), (3, 9)],
    'vision': 7,
    'sight_timeout': 2.5,
    'investigate_timeout': 10.0,
    'step_time': {'PATROL': 0.35, 'INVESTIGATE': 0.25, 'HUNT': 0.125},
    # Bayesian model milik Alien
    'diffuse_rate': 0.30,
    'outlier': 0.02,
    'miss': 0.03, 
    'dist_penalty': 0.08
}

parent = Path(__file__).resolve().parent.parent
with open(parent / 'config' / 'config.yaml') as config_file:
    config = yaml.safe_load(config_file)

director_config = config.get('Director', {})    # ambil dari dictionary config
alien_config = config.get('Alien', {})          # get function --> get('key', default-value)

Clue = namedtuple("Clue", "pos sigma source time")

class Director:
    DIRECTOR_SIGMA = director_config.get('sigma', default['sigma'])
    DIRECTOR_COOLDOWN = director_config.get('cooldown', default['cooldown'])

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
        self.cooldown = self.DIRECTOR_COOLDOWN[mode]
        sigma = self.DIRECTOR_SIGMA[mode]

        return Clue(self.leak_position(player.pos, sigma), sigma, "director", time_now)

class Alien:
    # Alien
    MISS    = alien_config.get('miss', default['miss'])     # P(not seeing player | player is in a cell I can see)
    VISION  = alien_config.get('vision', default['vision'])
    OUTLIER = alien_config.get('outlier', default['outlier'])
    STEP_TIME    = alien_config.get('step_time', default['step_time'])
    PATROL_NODES = alien_config.get('patrol_nodes', default['patrol_nodes'])
    PATROL_NODES = [tuple(item) for item in PATROL_NODES]
    DIFFUSE_RATE = alien_config.get('diffuse_rate', default['diffuse_rate'])     # motion model: player may have moved to a neighbour
    DIST_PENALTY = alien_config.get('dist_penalty', default['dist_penalty'])     # investigation target = belief / (1 + k * distance)
    SIGHT_TIMEOUT       = alien_config.get('sight_timeout', default['sight_timeout'])
    INVESTIGATE_TIMEOUT = alien_config.get('investigate_timeout', default['investigate_timeout'])

    def __init__(self, world:World, log):

        self.world = world
        self.log = log
        self.pos = world.alien_start
        self.state = "PATROL"
        self.patrol_nodes = self.PATROL_NODES
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


    # == Bagian Bayesian Belief Map/Network ==
    def reset_uniform(self):
        """
        kondisi awal saat kemungkinan posisi player bisa di mana saja
        (tidak ada tile yang mencolok untuk diinvestigasi)
        """
        p = 1.0 / len(self.world.walkable_tiles)
        self.belief = {c: p for c in self.world.walkable_tiles}

    def normalize(self):
        """
        normalisasi kemungkinan seluruh tile saat ini agar total kemungkinan dari
        semua tile totalnya = 1
        """
        total = sum(self.belief.values())
        if total < 1e-12:
            self.reset_uniform()
            return

        for tile in self.belief:
            self.belief[tile] /= total

    def diffuse(self, rate:float = DIFFUSE_RATE):
        """
        Model sederhana yang memprediksi gerakan player berdasarkan clue. 
        Pemanggilan method ini akan trigger penyebaran nilai belief setiap tile sebesar DIFFUSE_RATE%
        ke tetangganya yang walkable sebagai prediksi sederhana kira-kira player ke mana. 
        
        Misal tile yang kemungkinannya tinggi sudah dikunjungi tapi tidak ada player, belief di tile itu
        berkurang dan belief tile walkable di sekitarnya menjadi lebih tinggi.

        Referensi dan Sumber Inspirasi:
        > 1. Deepia, (2025, May 27). Diffusion Models: DDPM | Generative AI Animated. [Video]. https://www.youtube.com/watch?v=EhndHhIvWWw
        > 
        > 2. Bantuan claude untuk formula implementasi sederhana khusus case ini
        """
        new = {tile: belief * (1 - rate) for tile, belief in self.belief.items()} # kurangi setiap tile menjadi rasio (1 - rate)
        for tile, belief in self.belief.items():
            new_belief = self.world.walkable_neighbors(tile)

            if not new_belief: # tile tetangga tidak ada yang walkable
                new[tile] += belief * rate
                continue

            share = belief * rate / len(new_belief) # besar belief (sudah normalisasi) yang disalurkan ke tetangga
            for n in new_belief:
                new[n] += share

        self.belief = new

    def bayes_clue(self, pos:tuple, sigma:float):
        """
        Taruh clue di titik ini dan hitung persebaran pengaruhnya terhadap tile di sekitar
        sampai seluruh map

        > `P(T | z)  ∝  P(z | T) * P(T)`
            
        > Kemungkinan player ada di tile T, given posisi clue di z sebanding dengan
        > kemungkinan clue ditaruh di z jika player memang benar-benar ada di tile T 
        > dikali kemungkinan awal player ada di tile tersebut.

        note: OUTLIER membatasi kemungkinan terkecil untuk tile yang sangat jauh dari clue
        agar masih ada kemungkinan dicek alien (jadi target jalan selanjutnya)
        """
        denom = 2.0 * sigma * sigma
        for tile in self.belief:
            # rumus distribusi gaussian (normal) tapi di 2 sumbu (X dan Y)
            nume = (tile[0] - pos[0]) ** 2 + (tile[1] - pos[1]) ** 2
            self.belief[tile] *= (self.OUTLIER + math.exp(-nume / denom))

        self.normalize()

    def bayes_not_seen(self, visible:list):
        """Tilenya masuk dalam vision tapi tidak ada player, maka probabilitasnya berkurang"""
        for c in visible:

            if c in self.belief:
                self.belief[c] *= self.MISS

        self.normalize()


    # == Kelompok Utilities Terkait Perubahan State | patrol, investigate, atau hunt ==
    def set_state(self, new:str, now:float=0.0):

        if new == self.state:
            return

        self.log(f"Alien: {self.state} -> {new}")
        self.state = new
        self.path, self.target = [], None

        if new == "PATROL":
            self.reset_uniform()    # reset belief di semua tile
            dist = self.world.bfs_dist(self.pos) # ambil jarak ke semua tile yang bisa dijangkau

            # cari index node patrol yang paling dekat dengan posisi sekarang
            self.node_idx = min(
                range(len(self.patrol_nodes)),  # 0 -> size - 1 atau otomatis berisi list index patrol_nodes

                # ambil nilainya di dictionary dist (kalau tidak ada maka ganti ke 1e9) lalu cari yang terkecil
                key = lambda i: dist.get(self.patrol_nodes[i],1e9) 
            )

        elif new == "INVESTIGATE":
            self.last_clue_time = now

    def hear_clue(self, clue:Clue, time_now:float):

        if self.state == "HUNT": # abaikan clue kalau lagi hunt
            return                            

        self.last_clue = clue
        self.last_clue_time = time_now

        if self.state == "PATROL":  # investigasi clue
            self.set_state("INVESTIGATE", time_now)

        self.bayes_clue(clue.pos, clue.sigma)   # update belief tiap tile

        self.log(f"Heard {clue.source} clue (sigma={clue.sigma:.1f})")


    # == Main Method untuk Control Update per Tick/Frame ==
    def update(self, dt:float, player:Player, time_now:float):

        self.sees_player = self.world.visible(self.pos, player.pos, self.VISION)

        if self.sees_player:
            self.last_seen, self.lost_time = player.pos, 0.0
            self.set_state("HUNT", time_now)

        self.timer -= dt

        if self.timer > 0:
            return

        self.timer = self.STEP_TIME[self.state]

        if self.state == "PATROL":
            self.tick_patrol()

        elif self.state == "INVESTIGATE":
            self.tick_investigate(time_now)

        else:
            self.tick_hunt(player, time_now)

    def step_along(self, path:list):

        self.path = path or []
        if path:
            self.pos = path[0]  # pindah posisi satu langkah

    def tick_patrol(self):

        if self.pos == self.patrol_nodes[self.node_idx]:
            self.node_idx = (self.node_idx + 1) % len(self.patrol_nodes)

        self.step_along(self.world.bfs_path(self.pos, self.patrol_nodes[self.node_idx]))

    def tick_investigate(self, time_now):

        self.diffuse()
        self.bayes_not_seen(self.world.visible_cells(self.pos, self.VISION))

        if time_now - self.last_clue_time > self.INVESTIGATE_TIMEOUT:
            self.log("Alien lost the trail")
            self.set_state("PATROL", time_now)
            return

        target = self.choose_target()
        self.step_along(self.world.bfs_path(self.pos, target))

    def choose_target(self):

        dist = self.world.bfs_dist(self.pos)

        def score(tile):
            # semakin jauh sebuah tile, maka makin kecil kemungkinannya dipilih
            return self.belief[tile] / (1 + self.DIST_PENALTY * dist[tile]) if tile in dist else 0.0
        
        best = max(self.belief, key=score)

        # histeresis: hanya boleh ganti target kalau skor beliefnya lebih 25% dari best saat ini
        if self.target and (score(self.target) * 1.25 >= score(best)):
            best = self.target

        self.target = best
        return best

    def tick_hunt(self, player:Player, time_now):

        if self.sees_player:
            self.reset_uniform()
            self.bayes_clue(player.pos, 0.7) # timpa clue di player karena sudah melihat langsung
            goal = player.pos                # kejar secara langsung

        else:
            # kembali ke investigate kalau line of sight menuju player sudah timeout 
            self.lost_time += self.STEP_TIME["HUNT"]
            self.diffuse()
            goal = self.last_seen

            if self.pos == goal or self.lost_time > self.SIGHT_TIMEOUT:
                self.set_state("INVESTIGATE", time_now)
                return

        self.step_along(self.world.bfs_path(self.pos, goal))