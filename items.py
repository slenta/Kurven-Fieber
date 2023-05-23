import pygame
import random
import config as cfg
from subfunctions import do_points_intersect, get_random_position


class item_class:
    def __init__(self, screen, id):
        self.screen = screen
        self.screen_width = cfg.screen_width
        self.screen_height = cfg.screen_height
        self.radius = cfg.item_size
        self.id = id
        # item ids: 0: self slower, 1: self faster, 2: others slower, 3: others faster
        if self.id <= 1:
            self.color = cfg.green
        elif 1 < self.id <= 3:
            self.color = cfg.red
        self.position, self.dir = get_random_position()
        self.item = self.init_item()

    # First determine which speed item is rendered
    def random_id(self):
        id = random.choice(range(4))
        return id

    # Function to initialize the wanted item
    def init_item(self):
        item = {
            "id": self.id,
            "color": self.color,
            "timer": 0,
            "alive": False,
            "pos": self.position,
        }
        return item

    # Function to render the intem
    def render(self):
        pygame.draw.circle(self.screen, self.color, self.position, self.radius)
        self.item["alive"] = True

    def unrender(self):
        pygame.draw.circle(self.screen, cfg.black, self.position, self.radius)
        self.item["alive"] = False
        return self.item


def power_up(item, player, players):
    id = item["id"]
    for play in players:
        if id <= 1:
            if play["id"] == player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                if id == 0:
                    play["speed"] -= 1
                elif id == 1:
                    play["speed"] += 1
        elif 1 < id <= 3:
            if play["id"] is not player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                if id == 2:
                    play["speed"] -= 1
                elif id == 3:
                    play["speed"] += 1
    return players


def power_down(item, player, players):
    id = item["id"]
    for play in players:
        if id <= 1:
            if play["id"] == player["id"]:
                play["items"].pop(0)
                play["item_timer"].pop(0)
                if id == 0:
                    play["speed"] += 1
                elif id == 1:
                    play["speed"] -= 1
        elif 1 < id <= 3:
            if play["id"] is not player["id"]:
                play["items"].pop(0)
                play["item_timer"].pop(0)
                if id == 2:
                    play["speed"] += 1
                elif id == 3:
                    play["speed"] -= 1
    return players


# Function to render items
def render_items(screen, items):
    # Randomly choose one item
    item_id = random.randint(0, 3)
    item_c = item_class(screen, item_id)
    item_c.render()
    items.append(item_c)

    return items


def item_collision(player, players, item, items):
    collision = do_points_intersect(
        player["pos"], item.item["pos"], radius=cfg.player_size + cfg.item_size
    )
    if collision:
        player = power_up(item.item, player, players)
        item.unrender()
        items.remove(item)

    return players, items


def check_for_item_collision(players, items):
    for player in players:
        for item in items:
            players, items = item_collision(player, players, item, items)

    return player
