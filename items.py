import pygame
import random
import config as cfg
from subfunctions import do_points_intersect, get_random_position


# Function to initialize the wanted items
def init_items():
    items = []
    for i in range(cfg.num_greens):
        item = {"id": i, "color": cfg.green, "timer": 0, "alive": False, "pos": None}
        items.append(item)
    for i in range(cfg.num_reds):
        item = {"id": i, "color": cfg.red, "timer": 0, "alive": False, "pos": None}
        items.append(item)
    for i in range(cfg.num_neutrals):
        item = {"id": i, "color": cfg.blue, "timer": 0, "alive": False, "pos": None}
        items.append(item)
    return items


# Function to render items
def render_items(screen, items):
    # Get current time in milliseconds
    current_time = pygame.time.get_ticks()

    # Check if it's time to spawn a new item (every 10 seconds)
    if current_time % 10000 < 100:
        # Randomly choose one item
        item = random.choice(items)

        # Bring item to life
        item["alive"] = True
        item["timer"] = cfg.item_time

        # Randomly choose place for appearance
        position, dir = get_random_position()

        # Render the item as a circular shape
        pygame.draw.circle(screen, item["color"], position, cfg.item_size)


def item_collision(player, item):
    collision = do_points_intersect(
        player, item, radius=cfg.player_size + cfg.item_size
    )
    if collision:
        item_id = item["id"]
        player["power"].append(item_id)
        player["power_timer"].append(cfg.item_time)

    return player
