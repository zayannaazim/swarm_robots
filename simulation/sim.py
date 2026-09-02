"""
Decentralized Swarm Area-Division Simulator
=============================================

Simulates N rovers dividing a user-defined polygon into cells, using a
CBAA/CBBA-style consensus auction (no central hub) over a *range-limited*
simulated mesh. Rovers physically move toward the cells they win. If a
rover "dies" (click on it), its cells time out on its neighbors and get
re-auctioned automatically -- exactly the behaviour you want to port to
ESP32 + ESP-NOW.

Controls
--------
  Left click empty space (before pressing ENTER)  -> add a polygon vertex
  ENTER                                            -> lock polygon, spawn rovers, start sim
  Left click a rover (after sim starts)            -> kill / revive it (simulate failure)
  SPACE                                            -> pause / resume
  R                                                -> reset everything
  ESC / close window                               -> quit

Run:  pip install pygame --break-system-packages
      python swarm_simulation.py
"""

import math
import random
import sys
import time

import pygame

# ----------------------------- Config ---------------------------------

WIDTH, HEIGHT = 1000, 750
FPS = 60

DEFAULT_POLYGON = [(120, 100), (880, 130), (820, 620), (200, 650), (100, 380)]

NUM_ROVERS = 4
CELL_SIZE = 40          # grid resolution (px) used to tile the polygon
ROVER_SPEED = 90.0      # px/sec
COMM_RADIUS = 260.0     # px -- simulated wireless range (limits mesh hops)
HEARTBEAT_TIMEOUT = 2.0 # sec -- how long before a silent rover is presumed dead
BROADCAST_PERIOD = 0.25 # sec -- how often each rover "transmits" its state

ROVER_COLORS = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113),
    (241, 196, 15), (155, 89, 182), (26, 188, 156),
]

BG_COLOR = (24, 26, 32)
POLY_COLOR = (90, 95, 105)
UNCLAIMED_COLOR = (55, 58, 66)
CONFLICT_COLOR = (200, 40, 40)
DEAD_COLOR = (70, 70, 70)

FONT_NAME = None  # default pygame font


# --------------------------- Geometry utils -----------------------------

def point_in_polygon(x, y, poly):
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    px, py = x, y
    x1, y1 = poly[-1]
    for i in range(n):
        x2, y2 = poly[i]
        if ((y1 > py) != (y2 > py)) and \
           (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ------------------------------- Cell -----------------------------------

class Cell:
    __slots__ = ("id", "cx", "cy", "rect")

    def __init__(self, cid, cx, cy, rect):
        self.id = cid
        self.cx = cx
        self.cy = cy
        self.rect = rect


def build_cells(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    cells = []
    cid = 0
    y = y0
    while y < y1:
        x = x0
        while x < x1:
            cx, cy = x + CELL_SIZE / 2, y + CELL_SIZE / 2
            if point_in_polygon(cx, cy, polygon):
                rect = pygame.Rect(int(x), int(y), CELL_SIZE, CELL_SIZE)
                cells.append(Cell(cid, cx, cy, rect))
                cid += 1
            x += CELL_SIZE
        y += CELL_SIZE
    return cells


# ------------------------------- Rover ------------------------------------

class Rover:
    """
    One swarm agent. Mirrors the CBAA/CBBA state each ESP32 would carry:
      y[cell] -> best known bid value for that cell
      z[cell] -> id of the agent believed to currently hold that cell
    Consensus = max-bid merge whenever two rovers are within comm range.
    """

    def __init__(self, rid, x, y_, color, cells, capacity):
        self.id = rid
        self.pos = [x, y_]
        self.color = color
        self.alive = True
        self.capacity = capacity          # max cells this rover may hold

        n = len(cells)
        self.y = [0.0] * n                # winning bid list
        self.z = [-1] * n                 # winning agent list
        self.bundle = []                  # cell ids this rover currently holds

        self.last_heard = {}              # other_id -> last time we heard from them
        self.last_broadcast = 0.0
        self.target_jitter = (random.uniform(-6, 6), random.uniform(-6, 6))

    # ---- bidding phase -------------------------------------------------
    def bid_score(self, cell):
        d = dist(self.pos, (cell.cx, cell.cy))
        return 1.0 / (1.0 + d)

    def try_bid(self, cells):
        if not self.alive:
            return
        if len(self.bundle) >= self.capacity:
            return
        best_j, best_score = None, -1
        for cell in cells:
            j = cell.id
            if j in self.bundle:
                continue
            score = self.bid_score(cell)
            if score > self.y[j] and score > best_score:
                best_score, best_j = score, j
        if best_j is not None:
            self.y[best_j] = best_score
            self.z[best_j] = self.id
            self.bundle.append(best_j)

    # ---- consensus phase -------------------------------------------------
    def consensus_with(self, other, now):
        """Max-bid merge (the core of CBAA/CBBA conflict resolution)."""
        self.last_heard[other.id] = now
        other.last_heard[self.id] = now
        n = len(self.y)
        for j in range(n):
            if other.y[j] > self.y[j]:
                if self.z[j] == self.id and self.id in self.bundle_owner_check(j):
                    pass
                if self.z[j] == self.id and j in self.bundle:
                    self.bundle.remove(j)
                self.y[j] = other.y[j]
                self.z[j] = other.z[j]
            elif other.y[j] < self.y[j]:
                pass  # our info is better; do nothing (other side will update)

    def bundle_owner_check(self, j):
        return [self.id] if self.z[j] == self.id else []

    # ---- failure detection -------------------------------------------------
    def expire_dead_neighbors(self, alive_ids, now):
        """If we haven't heard from the rover that supposedly owns a cell
        in a while (and it's not us), release that cell for re-auction."""
        n = len(self.y)
        for j in range(n):
            owner = self.z[j]
            if owner == -1 or owner == self.id:
                continue
            last = self.last_heard.get(owner, None)
            timed_out = (last is None) or (now - last > HEARTBEAT_TIMEOUT)
            if timed_out and owner not in alive_ids:
                self.y[j] = 0.0
                self.z[j] = -1

    # ---- movement -------------------------------------------------
    def move(self, cells, dt):
        if not self.alive or not self.bundle:
            return
        # steer toward centroid of currently-held cells
        cx = sum(cells[j].cx for j in self.bundle) / len(self.bundle)
        cy = sum(cells[j].cy for j in self.bundle) / len(self.bundle)
        cx += self.target_jitter[0]
        cy += self.target_jitter[1]
        dx, dy = cx - self.pos[0], cy - self.pos[1]
        d = math.hypot(dx, dy)
        if d > 2:
            self.pos[0] += ROVER_SPEED * dt * dx / d
            self.pos[1] += ROVER_SPEED * dt * dy / d


# ------------------------------ Simulation --------------------------------

class Simulation:
    def __init__(self):
        self.polygon = []
        self.cells = []
        self.rovers = []
        self.state = "DRAW_POLYGON"   # DRAW_POLYGON -> RUNNING
        self.paused = False

    def reset(self):
        self.__init__()

    def finalize_polygon(self):
        if len(self.polygon) < 3:
            self.polygon = list(DEFAULT_POLYGON)
        self.cells = build_cells(self.polygon)
        if not self.cells:
            self.cells = build_cells(DEFAULT_POLYGON)
            self.polygon = list(DEFAULT_POLYGON)

        capacity = max(1, math.ceil(len(self.cells) / NUM_ROVERS) + 1)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

        self.rovers = []
        for i in range(NUM_ROVERS):
            angle = 2 * math.pi * i / NUM_ROVERS
            x = cx + 30 * math.cos(angle)
            y = cy + 30 * math.sin(angle)
            color = ROVER_COLORS[i % len(ROVER_COLORS)]
            self.rovers.append(Rover(i, x, y, color, self.cells, capacity))

        self.state = "RUNNING"

    def alive_ids(self):
        return {r.id for r in self.rovers if r.alive}

    def update(self, dt):
        if self.state != "RUNNING" or self.paused:
            return

        now = time.time()
        alive = [r for r in self.rovers if r.alive]

        # 1) bidding phase: every alive rover tries to claim a free cell
        for r in alive:
            r.try_bid(self.cells)

        # 2) mesh consensus: only rovers within COMM_RADIUS "hear" each other
        for i in range(len(alive)):
            for k in range(i + 1, len(alive)):
                a, b = alive[i], alive[k]
                if dist(a.pos, b.pos) <= COMM_RADIUS:
                    a.consensus_with(b, now)
                    b.consensus_with(a, now)

        # 3) failure detection: drop stale claims from unreachable/dead rovers
        alive_ids = self.alive_ids()
        for r in alive:
            r.expire_dead_neighbors(alive_ids, now)

        # 4) movement
        for r in alive:
            r.move(self.cells, dt)

    # -------------------------- rendering --------------------------------

    def cell_owner_color(self, cell):
        claimants = [r for r in self.rovers if r.alive and cell.id in r.bundle]
        if len(claimants) == 0:
            return UNCLAIMED_COLOR
        if len(claimants) == 1:
            return claimants[0].color
        return CONFLICT_COLOR

    def draw(self, screen, font):
        screen.fill(BG_COLOR)

        if self.state == "DRAW_POLYGON":
            if len(self.polygon) >= 2:
                pygame.draw.lines(screen, POLY_COLOR, False, self.polygon, 2)
            for p in self.polygon:
                pygame.draw.circle(screen, (230, 230, 230), p, 4)
            msg = font.render(
                "Click to add polygon vertices, then press ENTER "
                "(or just press ENTER for the default arena)", True, (220, 220, 220))
            screen.blit(msg, (20, HEIGHT - 30))
            return

        # cells
        for cell in self.cells:
            color = self.cell_owner_color(cell)
            pygame.draw.rect(screen, color, cell.rect)
            pygame.draw.rect(screen, BG_COLOR, cell.rect, 1)

        # polygon outline
        pygame.draw.polygon(screen, POLY_COLOR, self.polygon, 3)

        # rovers
        for r in self.rovers:
            col = r.color if r.alive else DEAD_COLOR
            pygame.draw.circle(screen, col, (int(r.pos[0]), int(r.pos[1])), 12)
            pygame.draw.circle(screen, (0, 0, 0), (int(r.pos[0]), int(r.pos[1])), 12, 2)
            label = font.render(str(r.id), True, (0, 0, 0))
            screen.blit(label, (r.pos[0] - 5, r.pos[1] - 8))
            # comm range (faint)
            if r.alive:
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.circle(s, (*col, 25), (int(r.pos[0]), int(r.pos[1])), int(COMM_RADIUS))
                screen.blit(s, (0, 0))

        # status text
        n_cells = len(self.cells)
        n_claimed = sum(1 for c in self.cells if len(
            [r for r in self.rovers if r.alive and c.id in r.bundle]) == 1)
        n_conflict = sum(1 for c in self.cells if len(
            [r for r in self.rovers if r.alive and c.id in r.bundle]) > 1)
        status = (f"cells: {n_cells}  claimed: {n_claimed}  "
                  f"conflicts(converging): {n_conflict}  "
                  f"[click a rover = kill/revive, SPACE = pause, R = reset]")
        screen.blit(font.render(status, True, (200, 200, 200)), (16, HEIGHT - 28))


# --------------------------------- Main -----------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Decentralized Swarm Area-Division Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, 18)

    sim = Simulation()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN and sim.state == "DRAW_POLYGON":
                    sim.finalize_polygon()
                elif event.key == pygame.K_SPACE and sim.state == "RUNNING":
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    sim.reset()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if sim.state == "DRAW_POLYGON":
                    sim.polygon.append((mx, my))
                elif sim.state == "RUNNING":
                    for r in sim.rovers:
                        if dist(r.pos, (mx, my)) < 14:
                            r.alive = not r.alive
                            break

        sim.update(dt)
        sim.draw(screen, font)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()