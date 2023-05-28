import pygame
import math
from decimal import *
import config as cfg
from items import check_for_item_collision, power_down
from subfunctions import (
    get_random_position,
    get_random_gap,
    check_for_collisions,
    update_player_direction,
)


def move_player(player):
    # check if player is alive
    if player["alive"]:
        # Get gap and line timer
        gap_timer = player["gap_timer"]
        line_timer = player["line_timer"]

        # check if gap is on
        if line_timer == 0:
            player["gap"] = True

        # update player direction and get player values
        player = update_player_direction(player)
        x, y = player["pos"]
        dx, dy = player["dir"]

        # move player
        x += dx * player["speed"]
        y += dy * player["speed"]

        # update item timers
        for i in range(len(player["items"])):
            print(len(player["items"]), len(player["item_timer"]), i)
            if player["item_timer"][i] <= 0:
                player = power_down(player["items"][i], player["item_timer"][i], player)
                break
            else:
                player["item_timer"][i] -= 1

        # update timers
        if player["gap"] == False:
            player["line_timer"] -= 1
        else:
            player["gap_timer"] -= 1

        # check if gap timer is 0 and reset timers if necessary
        if gap_timer <= 0:
            player["gap"] = False
            player["gap_timer"], player["line_timer"] = get_random_gap()

        # Update player position
        player["pos"] = (x, y)
        # Update gap history
        player["gap_history"].insert(0, player["gap"])
        # update position history
        player["pos_history"].insert(0, (x, y))

    return player


def move_players(players, items):
    for player in players:
        if player["alive"]:
            if player["dir"] == "stop":
                continue
            else:
                players = check_for_collisions(players)
                players, items = check_for_item_collision(players, items)
                # item_collision(player, items)
                player = move_player(player)


def init_players(num_players, player_keys):
    player_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    players = []
    for i in range(num_players):
        player_key = player_keys[i]
        start_pos, start_dir = get_random_position()
        start_gap, start_line = get_random_gap()
        gap = False
        player = {
            "pos": start_pos,
            "dir": start_dir,
            "angle": 0,
            "color": player_colors[i],
            "alive": True,
            "length": 1,
            "speed": cfg.speed,
            "id": i,
            "pos_history": [start_pos],
            "size": cfg.player_size,
            "gap": gap,
            "gap_history": [gap],
            "gap_timer": start_gap,
            "line_timer": start_line,
            "del_angle": 5,
            "items": [],
            "item_timer": [],
            "left": player_key["left"],
            "right": player_key["right"],
            "ai": False,
        }
        players.append(player)
    return players


def get_winner(players, win_counts):
    alive_players = [player for player in players if player["alive"]]

    if len(alive_players) == 1:
        winner = alive_players[0]
        winner_id = winner["id"] + 1
        # Increment win count for the winning player
        win_counts[winner_id] += 1
        return winner_id
    else:
        return None
