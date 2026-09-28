from world import World

from pathlib import Path
import yaml

default = {
    'walk_step': 0.24,
    'walk_step': 0.24,
    'walk_noise_rad': 3,
    'run_step': 0.12,
    'run_noise_rad': 7,
    'noise_spacing': 0.7
}

parent = Path(__file__).resolve().parent.parent
with open(parent / 'config' / 'config.yaml') as config_file:
    config = yaml.safe_load(config_file)

player_config = config.get('Player', {})

class Player:
    """
    Entitas aktif yang dikendalikan oleh player melalui input keyboard: arrow-keys

    @Init_Params:
        pos: koordinat (X-Y) awal objek ini berdasarkan tiles di map
    
    @Attributes:
        **NAMA** | **SATUAN**. **DESKRIPSI**
        WALK_STEP | detik. waktu untuk melewati 1 tile saat berjalan
        RUN_STEP  | detik. waktu untuk melewati 1 tile saat berlari
        WALK_NOISE_RADIUS | tile. radius noise player berjalan
        RUN_NOISE_RADIUS  | tile. radius noise player berlari
        NOISE_SPACING     | detik. jeda waktu sampai player bisa membuat noise lagi
        
    **inisiasilasi** class ini butuh input **`pos`**.
    """
    WALK_STEP = player_config.get('walk_step', default['walk_step'])    
    RUN_STEP  = player_config.get('run_step', default['run_step'])      
    WALK_NOISE_RADIUS = player_config.get('walk_noise_rad', default['walk_noise_rad'])
    RUN_NOISE_RADIUS = player_config.get('run_noise_rad', default['run_noise_rad'])     
    NOISE_SPACING = player_config.get('noise_spacing', default['noise_spacing'])     
    
    def __init__(self, pos:tuple):
        self.pos = pos
        self.timer = 0.0
        self.last_move = -999.0
        self.last_run = False

    def set_noise_radius(self):
        return self.RUN_NOISE_RADIUS if self.last_run else self.WALK_NOISE_RADIUS

    def noise_interlude(self, time_now:float):
        """
        Return True kalau player habis mengeluarkan noise dan
        sekarang lagi jeda waktu. Bisa dikatakan baru saja bergerak
        """
        return (time_now - self.last_move) < self.NOISE_SPACING

    def tick_update(self, dt:float, direction:tuple, running:bool, world:World, time_now:float):
        """
        Returns True kalau player berhasil bergerak di frame/tick saat ini
        
        @Params:
            dt: (detik) periode waktu untuk update frame/tick
            direction: arah pergerakan, sesuai input player
            running: apakah berlari
            world: world map buat cek apakah tile tujuan walkable
            time_now: waktu saat ini
        """
        self.timer = max(0.0, self.timer - dt)
        if direction and (self.timer <= 0):
            new_x, new_y = self.pos[0] + direction[0], self.pos[1] + direction[1]
            
            if world.walkable(new_x, new_y):
                self.pos = (new_x, new_y)
                self.timer = self.RUN_STEP if running else self.WALK_STEP
                self.last_move = time_now
                self.last_run = running
                return True
            
        return False