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
        self.item_letter = cfg.item_letters[self.id]
        if self.id <= 1:
            self.color = cfg.green
        elif 1 < self.id <= 5:
            self.color = cfg.red
        self.position = self.get_position()
        self.item = self.init_item()

    def get_position(self):
        x = random.randint(20, cfg.screen_width - cfg.score_section_width - 20)
        y = random.randint(20, cfg.screen_height - 20)
        position = (x, y)
        return position

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
        # Render the letter surface
        letter_surface = cfg.font.render(self.item_letter, True, cfg.white)
        letter_rect = letter_surface.get_rect(center=self.position)
        # Draw the letter inside the item
        self.screen.blit(letter_surface, letter_rect)
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
                    play["speed"] *= 2 / 3
                elif id == 1:
                    play["speed"] *= 3 / 2
        elif 1 < id <= 3:
            if play["id"] is not player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                if id == 2:
                    play["speed"] *= 2 / 3
                elif id == 3:
                    play["speed"] *= 3 / 2
        elif id == 4:
            if play["id"] is not player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                left = play["left"]
                right = play["right"]
                play["left"] = right
                play["right"] = left
        elif id == 5:
            if play["id"] is not player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                play["del_angle"] += 5
                if play["del_angle"] >= 22:
                    play["del_angle"] = 22
    return players


def power_down(item, item_timer, player):
    id = item["id"]
    player["items"].remove(item)
    player["item_timer"].remove(item_timer)
    if id == 1 or id == 3:
        player["speed"] *= 2 / 3
    elif id == 0 or id == 2:
        player["speed"] *= 3 / 2
    elif id == 4:
        left = player["left"]
        right = player["right"]
        player["left"] = right
        player["right"] = left
    elif id == 5:
        player["del_angle"] -= 5
        if player["del_angle"] <= 2:
            player["del_angle"] = 2
    return player


# Function to render items
def render_items(screen, items):
    # Randomly choose one item
    item_id = random.randint(0, cfg.num_items)
    item_c = item_class(screen, item_id)
    item_c.render()
    items.append(item_c)

    return items


def item_collision(player, players, item, items):
    collision = do_points_intersect(
        player["pos"], item.item["pos"], radius=cfg.player_size + cfg.item_size
    )
    if collision:
        players = power_up(item.item, player, players)
        item.unrender()
        items.remove(item)

    return players, items


def check_for_item_collision(players, items):
    for player in players:
        for item in items:
            players, items = item_collision(player, players, item, items)

    return players, items
