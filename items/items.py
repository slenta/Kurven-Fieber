import pygame
import random
import numpy as np
import config as cfg
from utils.subfunctions import do_points_intersect, get_random_position


class item_class:
    def __init__(self, id, screen=None):
        self.screen = screen
        self.screen_width = cfg.screen_width
        self.screen_height = cfg.screen_height
        self.radius = cfg.item_size
        self.id = id
        self.item_letter = cfg.item_letters[self.id - 5]
        if self.id <= 6:
            self.color = cfg.green
        elif 6 < self.id <= 10:
            self.color = cfg.red
        self.position = self.get_position()
        self.item = self.init_item()
        self.item_x = np.arange(
            int(round(self.position[0], 0)) - int(self.radius),
            int(round(self.position[0], 0)) + int(self.radius + 1),
        )
        self.item_y = np.arange(
            int(round(self.position[1], 0)) - int(self.radius),
            int(round(self.position[1], 0)) + int(self.radius + 1),
        )
        # self.it_pos = np.column_stack((self.item_x, self.item_y))

    def get_position(self):
        x = random.randint(20, cfg.screen_width - cfg.score_section_width - 20)
        y = random.randint(20, cfg.screen_height - 20)
        position = (x, y)
        return position

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
    def render(self, game_state):
        # Update game state
        game_state[self.item_x, self.item_y] = np.where(
            game_state[self.item_x, self.item_y] == 0,
            self.id,
            game_state[self.item_x, self.item_y],
        )
        pygame.draw.circle(self.screen, self.color, self.position, self.radius)
        # Render the letter surface
        letter_surface = cfg.font.render(self.item_letter, True, cfg.white)
        letter_rect = letter_surface.get_rect(center=self.position)
        # Draw the letter inside the item
        self.screen.blit(letter_surface, letter_rect)
        self.item["alive"] = True
        return game_state

    def render_sim(self, game_state):
        # Update game state
        game_state[self.item_x, self.item_y] = np.where(
            game_state[self.item_x, self.item_y] == 0,
            self.id,
            game_state[self.item_x, self.item_y],
        )
        self.item["alive"] = True
        return game_state

    def unrender(self, game_state):
        pygame.draw.circle(self.screen, cfg.black, self.position, self.radius)
        game_state[self.item_x, self.item_y] = np.where(
            game_state[self.item_x, self.item_y] == self.id,
            0,
            game_state[self.item_x, self.item_y],
        )
        self.item["alive"] = False

        return game_state

    def unrender_sim(self, game_state):
        game_state[self.item_x, self.item_y] = np.where(
            game_state[self.item_x, self.item_y] == self.id,
            0,
            game_state[self.item_x, self.item_y],
        )
        self.item["alive"] = False

        return game_state


def power_up(item, player, players):
    id = item["id"]
    for play in players:
        if id <= 6:
            if play["id"] == player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                if id == 5:
                    play["speed"] *= 2 / 3
                elif id == 6:
                    play["speed"] *= 3 / 2
        elif 6 < id <= 8:
            if play["id"] is not player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                if id == 7:
                    play["speed"] *= 2 / 3
                elif id == 8:
                    play["speed"] *= 3 / 2
        elif id == 9:
            if play["id"] is not player["id"]:
                play["items"].append(item)
                play["item_timer"].append(cfg.item_time)
                left = play["left"]
                right = play["right"]
                play["left"] = right
                play["right"] = left
        elif id == 10:
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
    if id == 6 or id == 8:
        player["speed"] *= 2 / 3
    elif id == 5 or id == 7:
        player["speed"] *= 3 / 2
    elif id == 9:
        left = player["left"]
        right = player["right"]
        player["left"] = right
        player["right"] = left
    elif id == 10:
        player["del_angle"] -= 5
        if player["del_angle"] <= 2:
            player["del_angle"] = 2
    return player


# Function to render items
def render_items(items, game_state, screen=None):
    # Randomly choose one item
    item_id = random.randint(5, cfg.num_items + 5)
    item_c = item_class(item_id, screen)
    if screen:
        game_state = item_c.render(game_state)
    else:
        game_state = item_c.render_sim(game_state)
    items.append(item_c)

    return items, game_state


def item_collision(player, players, item, items, game_state):
    # Check for collisions
    collision = do_points_intersect(
        player["pos"], item.item["pos"], radius=cfg.player_size + cfg.item_size
    )
    # Update players, items, game_state
    if collision:
        players = power_up(item.item, player, players)
        if not player["ai"]:
            game_state = item.unrender(game_state)
        else:
            game_state = item.unrender_sim(game_state)
        items.remove(item)

    return players, items, game_state


def check_for_item_collision(players, items, game_state):
    for player in players:
        for item in items:
            players, items, game_state = item_collision(
                player, players, item, items, game_state
            )

    return players, items, game_state
