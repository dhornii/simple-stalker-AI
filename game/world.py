import random
import math

from pathlib import Path
import yaml

from collections import deque

default = {
    'gen_rad': 20,
    'gen_cd': 10.0,

    'cell_size': 40,
    'move_dir': [(1, 0), (-1, 0), (0, 1), (0, -1)],
    'map': [ 
        # N: noise generator (selanjutnya disebut 'generator' aja)
        # o: obstacle. membatasi langkah saja 
        # #: wall. membatasi langkah dan vision
        "####################", 
        "#P....#.....o......#",
        "#.....#............#",
        "#..oo.....###..N...#",
        "#..oo.....#........#",
        "#.........#..oo....#",
        "####..#####..##....#",
        "####...####..#######",
        "#........#.....A...#",
        "#..oo....#...N.....#",
        "#........#.........#",
        "#..N.........o....E#",
        "####################",
    ]
}

parent = Path(__file__).resolve().parent.parent
with open(parent / 'config' / 'config.yaml') as config_file:
    config = yaml.safe_load(config_file)

gen_config = config.get('Generator', {})
world_config = config.get('World', {})

class Generator:
    """
    Objek yang bisa menghasilkan suara untuk mengecoh alien (stalker agent),
    bisa nyala otomatis setelah malfungsi atau di-trigger manual sama playernya.

    @Init_Params:
        **`pos`**: koordinat tile (X-Y) tempat generator

    @Attributes:
        **`pos (tuple)`**: koordinat tile (X-Y) objek ini berdasarkan tiles di map
        **`cooldown (float)`**: detik. jeda waktu sampai bisa dinyalakan lagi
        **`malfunction_time (float)`**: detik. jeda waktu otomatis dan random 8 - 13 detik

    **inisiasilasi** class ini butuh input **`pos`**.

    catatan tentang posisi X-Y di pygame --> [Help! How Do I move An Image](https://www.pygame.org/docs/tut/MoveIt.html?highlight=position)
    section *'Screen Coordinates'*
    """

    
    # konstanta class generator
    GEN_RADIUS   = gen_config.get('radius', default['gen_rad'])         # (tile) radius luas noise hasil generator
    GEN_COOLDOWN = gen_config.get('cooldown', default['gen_cd'])      # (detik)

    def __init__(self, pos:tuple):

        self.pos = pos
        self.cooldown = 0.0
        self.malfunction_time = random.uniform(8.0, 13.0) # random 8-13 detik generator malfungsi (kemungkinan tiap waktunya sama)

    def ready(self):
        return self.cooldown <= 0

    def trigger(self):  # kalau ditrigger manual
        self.cooldown = self.GEN_COOLDOWN
        self.malfunction_time = random.uniform(8.0, 13.0)

    def tick_update(self, dt:float):
        """
        update waktu cooldown atau malfungsinya tiap tick (periode) game.

        @params:
            dt: periode (detik), waktu untuk update frame/tick 
        """
        self.cooldown = max(0.0, self.cooldown - dt)
        self.malfunction_time -= dt

        if self.malfunction_time <= 0 and self.ready():
            # fitur QoL: trigger otomatis kalau generator tidak malfungsi
            self.trigger()
            return True
        
        return False

class World:
    """
    Mengatur semua mekanisme terkait map dan world game secara umum, contohnya berupa sifat-sifat tile in-game serta
    bagaimana interaksinya terhadap player. Implementasi Breadth First Search (BFS) ada  di method class ini
    (tepatnya 2 method akhir: **`bfs_path`** dan **`bfs_dist`**) sebagai pembentuk path jalan agent alien.

    Sisa method mengatur bagaimana alien dapat melihat playernya dan sistem + mekanisme tile (e.g., *walkable*, *is_wall/menutup line of sight*,
    dan *boundary check*)
    
    @Attributes:
        **`tiles (list)`**: list 2 dimensi untuk mengakses koordinat petak/tile tertentu
        **`generators (list)`**: list seluruh koordinat yang ditempati generator
        **`player_start (tuple)`**: koordinat spawn player. diisi otomatis berdasarkan **`map_layout`**
        **`alien_start (tuple)`**: koordinat spawn alien. diisi otomatis berdasarkan **`map_layout`**
        **`exit (tuple)`**: koordinat tile exit. diisi otomatis berdasarkan **`map_layout`**
        **`walkalbe_tiles (list)`**: list semua koordinat tile *walkable*. diisi otomatis berdasarkan **`map_layout`**
    """

    CELL_SIZE = world_config.get('cell_size', default['cell_size'])   # (pixel) ukuran tile = CELL_SIZE * CELL_SIZE
    MAP       = world_config.get('map', default['map'])
    MAP_W     = len(MAP[0])
    MAP_H     = len(MAP)          

    MOVE_DIR  = world_config.get('move_dir', default['move_dir'])
    MOVE_DIR  = [tuple(item) for item in MOVE_DIR]

    def __init__(self):

        self.tiles = [list(r) for r in self.MAP]
        self.player_start = self.alien_start = self.exit = None
        self.generators = []

        for y in range(self.MAP_H):
            for x in range(self.MAP_W):

                element = self.tiles[y][x]
                if element == 'P':
                    self.player_start = (x,y)   # simpan koordinat start player
                    self.tiles[y][x] = '.'      # tetapkan sebagai walkable tile
                elif element == 'A':
                    self.alien_start = (x,y)
                    self.tiles[y][x] = '.'
                elif element == 'E':
                    self.exit = (x,y)
                    self.tiles[y][x] = 'E'      # tetapkan sebagai exit tile
                elif element == 'N':
                    self.generators.append(
                        Generator(pos = (x, y))
                    )

        self.walkable_tiles = [
            (x, y) for y in range(self.MAP_H) for x in range(self.MAP_W) if self.walkable(x, y)
        ]


    # == Kelompok Utilities Terkait Tile Map ==
    def boundary_check(self, x:int, y:int):
        """
        return True kalau koordinat (x,y) ada di dalam map
        """
        return (0 <= x < self.MAP_W) and (0 <= y < self.MAP_H)

    def walkable(self, x:int, y:int):
        """
        return True kalau koordinat (x,y) bukan obstacle, wall, atau generator
        """
        return self.boundary_check(x, y) and self.tiles[y][x] not in "#oN"

    def is_wall(self, x, y):
        return self.tiles[y][x] == "#"

    def walkable_neighbors(self, coord:tuple):
        """
        return list koordinat yang walkable di sekitar coord

        @Params:
            coord: tuple (x,y) yang akan dicek
        """
        x, y = coord
        nearby_walkable = []

        for dx, dy in self.MOVE_DIR:
            if self.walkable(x + dx, y + dy):
                nearby_walkable.append((x + dx, y + dy))

        return nearby_walkable
    

    # == Kelompok Utilities Terkait Sight Antara Player dengan Alien == | Untuk menentukan apakah alien melihat player
    def bresenham_line(self, start:tuple, end:tuple):
        """
        algoritma [garis bresenham](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm): 
        
        bentuk garis lurus dari alien menuju player lalu
        catat tile mana saja yang paling tepat untuk bentuk aproksimasi garis tersebut.

        @params:
            start: titik awal garis
            end: titik akhir garis

        sumber yang digunakan sebagai acuan konsep: 
        > NoBS Code, [`Bresenham's Line Algorithm - Demystified Step by Step`](https://www.youtube.com/watch?v=CceepU1vIKo)
        >
        > Andre Prihodko, [`How Your Computer Draws Lines`](https://www.youtube.com/watch?v=8gIhNSAXYcQ)
        """
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1

        vertically_dom = (dy > dx)
        if vertically_dom:
            dx, dy = dy, dx

        decision_point = 2 * dy - dx
        points = []

        x, y = x0, y0
        for i in range(dx + 1):
            points.append((x, y))

            if decision_point > 0:
                if vertically_dom:
                    x += step_x
                else:
                    y += step_y
                decision_point -= 2 * dx

            if vertically_dom:
                y += step_y
            else:
                x += step_x

            decision_point += 2 * dy

        return points

    def line_of_sight(self, start:tuple, end:tuple):
        """
        Return True kalau tidak ada wall di sepanjang tile hasil bresenham.
        """
        return not any(self.is_wall(*p) for p in self.bresenham_line(start, end)[1:-1])

    def visible(self, start:tuple, end:tuple, vision_range:float):
        """
        Return True kalau jarak start ke end di dalam jangkauan vision_range dan tidak ada wall.
        di antara keduanya
        """
        return (math.dist(start, end) <= vision_range) and (self.line_of_sight(start, end))

    def visible_cells(self, origin:tuple, vision_range:float):
        """
        Return list semua tile yang termasuk dalam radius jangkauan vision dan bisa dilihat.
        """
        x_org, y_org = origin
        r = int(vision_range)
        visible_cells = []

        for y in range(y_org - r, y_org + r + 1):
            for x in range(x_org - r, x_org + r + 1):

                if self.boundary_check(x, y) and self.walkable(x, y) and self.visible(origin, (x, y), vision_range):
                    visible_cells.append((x, y))

        return visible_cells
    

    # == Kelompok Implementasi BFS Untuk Pathing Jalan Alien ==
    def bfs_path(self, start:tuple, goal:tuple):
        """
        Return list tile untuk jarak terdekat dari koordinat start menuju goal dengan algoritma BFS.
        Implementasinya melalui Double-Ended queue.

        referensi:
        [Dequeu in Python](https://www.geeksforgeeks.org/python/deque-in-python/) dari geeksforgeeks
        """
        if start == goal:
            return []

        # dictionary child ke parent dari satu tile untuk catat tile yang sudah dikunjungi
        # key adalah child | data adalah parent
        # child: parent
        prev = {
            start: None
        }

        queue = deque([start])  # inisialisasi Double-Ended queue
        found = False

        while queue:
            current = queue.popleft()

            if current == goal:
                found = True
                break

            for n in self.walkable_neighbors(current):
                if n not in prev:
                    prev[n] = current   # catat tile yang sudah dikunjungi
                    queue.append(n)

        if not found:
            return None

        path = []
        current = goal
        while current != start:
            path.append(current)    # append child dari prev
            current = prev[current] # pindah ke parentnya

        path.reverse() 
        return path

    def bfs_dist(self, start:tuple):
        """
        Return dictionary berisi jarak terpendek dari start ke seluruh tile yang reachable.
        """
 
        dist = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for n in self.walkable_neighbors(current):
                if n not in dist:

                    dist[n] = dist[current] + 1 # hitung jarak + 1 setiap mengunjungi tile baru
                    queue.append(n)
        
        return dist