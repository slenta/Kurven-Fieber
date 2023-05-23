import pygame
import math
from decimal import *
import config as cfg
from items import check_for_item_collision
from subfunctions import (
    get_random_position,
    get_random_gap,
    check_for_collisions,
    update_player_direction,
)


def move_player(player, player_key):
    # check if player is alive
    if player["alive"]:
        # Get gap and line timer
        gap_timer = player["gap_timer"]
        line_timer = player["line_timer"]

        # check if gap is on
        if line_timer == 0:
            player["gap"] = True

        # update player direction and get player values
        player = update_player_direction(player, player_key)
        x, y = player["pos"]
        dx, dy = player["dir"]

        # move player
        x += math.floor(dx * player["speed"] * 100) / 100
        y += math.floor(dy * player["speed"] * 100) / 100

        # update player position and position history
        player["pos"] = (x, y)
        player["pos_history"].insert(0, (x, y))

        # update item timers
        # if len(player["items"]) != 0:
        for item, item_timer in zip(player["items"], player["item_timer"]):
            if item_timer <= 0:
                player["item"].pop(item)
                player["item_timer"].pop(item_timer)
            else:
                item_timer -= 1

        # update timers
        if player["gap"] == False:
            player["line_timer"] -= 1
        else:
            player["gap_timer"] -= 1

        # check if gap timer is 0 and reset timers if necessary
        if gap_timer <= 0:
            player["gap"] = False
            player["gap_timer"], player["line_timer"] = get_random_gap()

        # Update gap history
        player["gap_history"].insert(0, player["gap"])

    return player


def move_players(players, player_keys, items):
    for player, player_key in zip(players, player_keys):
        if player["alive"]:
            if player["dir"] == "stop":
                continue
            else:
                check_for_collisions(players)
                check_for_item_collision(players, items)
                # item_collision(player, items)
                move_player(player, player_key)


def init_players(num_players):
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
            "items": [],
            "item_timer": [],
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
