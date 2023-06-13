import pygame
import math
import numpy as np
from decimal import *
import config as cfg
from items import check_for_item_collision, power_down
from ai_player import update_ai_player_direction
from subfunctions import (
    get_random_position,
    get_random_gap,
    check_for_collisions,
    update_player_direction,
)


def move_player(player, game_state, players=None):
    # check if player is alive
    if player["alive"]:
        # Get gap and line timer
        gap_timer = player["gap_timer"]
        line_timer = player["line_timer"]

        # check if gap is on
        if line_timer == 0:
            player["gap"] = True

        # update player direction and get player values
        if player["ai"] == True:
            player = update_ai_player_direction(player, game_state, players)
            player["iteration"] += 1
        else:
            player = update_player_direction(player)
        x, y = player["pos"]
        dx, dy = player["dir"]

        # move player
        x += dx * player["speed"]
        y += dy * player["speed"]

        # update item timers
        for i in range(len(player["items"])):
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
        player["game_state_pos"] = np.array(
            [
                np.arange(
                    int(round(x, 0)) - int(player["size"] / 2),
                    int(round(x, 0)) + int(player["size"] / 2),
                ),
                np.arange(
                    int(round(y, 0)) - int(player["size"] / 2),
                    int(round(y, 0)) + int(player["size"] / 2),
                ),
            ]
        )
        # Update gap history
        player["gap_history"].insert(0, player["gap"])
        # update position history
        player["pos_history"].insert(0, (x, y))
        # Truncate history if it exceeds the maximum length
        # if len(player["pos_history"]) > cfg.player_max_history:
        #     player["pos_history"] = player["pos_history"][-cfg.player_max_history :]
        #     player["gap_history"] = player["gap_history"][-cfg.player_max_history :]

    return player


def move_players(players, items, game_state):
    for player in players:
        if player["alive"]:
            if player["dir"] == "stop":
                continue
            else:
                players, game_state = check_for_collisions(players, game_state)
                players, items, game_state = check_for_item_collision(
                    players, items, game_state
                )
                player = move_player(player, game_state, players)

    return players, game_state


def init_players(num_players, player_keys, players):
    player_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for i in range(cfg.num_ai_players, num_players):
        player_key = player_keys[i - cfg.num_ai_players]
        start_pos, start_dir = get_random_position()
        game_state_pos = np.array(
            [
                np.arange(
                    int(round(start_pos[0], 0)) - int(cfg.player_size / 2),
                    int(round(start_pos[0], 0)) + int(cfg.player_size / 2),
                ),
                np.arange(
                    int(round(start_pos[1], 0)) - int(cfg.player_size / 2),
                    int(round(start_pos[1], 0)) + int(cfg.player_size / 2),
                ),
            ]
        )
        start_gap, start_line = get_random_gap()
        gap = False
        player = {
            "pos": start_pos,
            "game_state_pos": game_state_pos,
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
