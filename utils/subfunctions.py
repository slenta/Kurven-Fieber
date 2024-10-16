import random
import torch
import numpy as np
import config as cfg
import pygame
import math
import psutil

cfg.set_args()


def get_random_position():
    x = random.randint(100, cfg.screen_width - cfg.score_section_width - 100)
    y = random.randint(100, cfg.screen_height - 100)
    position = (x, y)
    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    direction = (dx, dy)
    return position, direction


def get_random_gap():
    start_gap = random.randint(cfg.min_gap, cfg.max_gap)
    start_line = random.randint(cfg.min_line, cfg.max_line)
    return start_gap, start_line


def do_positions_intersect(pos1, pos2, pos_history2):
    """Checks if two positions intersect with each other."""
    for i in range(len(pos_history2) - 1):
        if do_segments_intersect(pos1, pos2, pos_history2[i], pos_history2[i + 1]):
            return True
    return False


def do_points_intersect(point1, point2, radius=2):
    distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return distance < radius


def do_segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
    """Checks if two line segments intersect with each other."""
    x1, y1 = seg1_start
    x2, y2 = seg1_end
    x3, y3 = seg2_start
    x4, y4 = seg2_end

    # calculate the direction of the lines
    u1 = (x2 - x1, y2 - y1)
    u2 = (x4 - x3, y4 - y3)
    v1 = (-u1[1], u1[0])
    v2 = (-u2[1], u2[0])

    # calculate the scalar values of the intersection point along each line segment
    # check if denominators are 0
    de1 = u1[0] * u2[1] - u1[1] * u2[0]
    de2 = u1[0] * u2[1] - u1[1] * u2[0]
    if de1 == 0 or de2 == 0:
        return False

    t1 = ((x3 - x1) * u2[1] - (y3 - y1) * u2[0]) / de1
    t2 = ((x3 - x1) * u1[1] - (y3 - y1) * u1[0]) / de2

    # check if the intersection point is within both line segments
    if t1 >= 0 and t1 <= 1 and t2 >= 0 and t2 <= 1:
        return True
    else:
        return False


def check_for_collisions(players, game_state):
    for player in players:
        if not player["alive"]:
            continue
        # Alternative check using game_state variable
        gs_pos = player["game_state_pos"]
        player_pairs = False

        if not player["gap"]:
            # Select points from game stack that would lead to collision and compare to position
            coll_points = torch.nonzero(
                (game_state >= 1)
                & (game_state <= 5)
                & (game_state != player["id"] + 2),
                as_tuple=True,
            )
            player_pairs = torch.any(
                (coll_points[0][:, torch.newaxis] == gs_pos[1])
                & (coll_points[1][:, torch.newaxis] == gs_pos[0])
            )

        # Check if colliding with walls
        wall_points = torch.nonzero(game_state == -1, as_tuple=True)
        wall_pairs = torch.any(
            (wall_points[0][:, np.newaxis] == gs_pos[1])
            & (wall_points[1][:, np.newaxis] == gs_pos[0])
        )

        # slow list solution
        # line_coll = game_state[
        #     np.where(
        #         (game_state >= 2) & (game_state <= 5) & (game_state != player["id"] + 2)
        #     )[0]
        # ].tolist()
        # coll_points_line = set(zip(*line_coll)) & set(zip(*gs_list))

        # wall_coll = game_state[np.where(game_state == -1)[0]].tolist()
        # coll_points_wall = set(zip(*wall_coll)) & set(zip(*gs_list))

        # If collision: kill player, otherwise: update game state
        if wall_pairs or player_pairs:
            player["alive"] = False
        else:
            # At first change to player id
            game_state[gs_pos[1], gs_pos[0]] = player["id"] + 2

            # After direct collision period with own head: change to 1
            if len(player["pos_history"]) >= cfg.player_min_collision + 3:
                x, y = player["pos_history"][cfg.player_min_collision]
                x_pos = torch.arange(
                    int(round(x, 0)) - int(player["size"] / 2),
                    int(round(x, 0)) + int(player["size"] / 2),
                )

                y_pos = torch.arange(
                    int(round(y, 0)) - int(player["size"] / 2),
                    int(round(y, 0)) + int(player["size"] / 2),
                )
                if all(
                    not player["gap_history"][i]
                    for i in range(
                        cfg.player_min_collision - 2, cfg.player_min_collision + 2
                    )
                ):
                    game_state[y_pos, x_pos] = 1
                else:
                    game_state[y_pos, x_pos] = 0

        # player_pos = player["pos"]
        # player_history = player["pos_history"]
        # gap_history = player["gap_history"]
        # player_size = player["size"]

        # # check collision with own line
        # for pos, gap in zip(player_history[5:], gap_history[5:]):
        #     if not gap:
        #         intersect = do_points_intersect(pos, player_pos)
        #         if intersect:
        #             player["alive"] = False
        #             break

        # # check collision with other player lines
        # for other_player in players:
        #     if other_player["id"] == player["id"]:
        #         continue
        #     other_history = other_player["pos_history"]
        #     other_gap = other_player["gap_history"]
        #     for pos, gap in zip(other_history, other_gap):
        #         if not gap:
        #             intersect = do_points_intersect(pos, player_pos)
        #             if intersect:
        #                 player["alive"] = False
        #                 break

        # # check collision with screen edges
        # if (
        #     player_pos[0] < player_size
        #     or player_pos[0] > cfg.screen_width - cfg.score_section_width - player_size
        #     or player_pos[1] < player_size
        #     or player_pos[1] > cfg.screen_height - player_size
        # ):
        #     player["alive"] = False
        #     break

    return players, game_state


def update_player_direction(player):
    # update directions
    keys = pygame.key.get_pressed()
    # Handle left key
    if keys[pygame.key.key_code(player["left"])]:
        player["angle"] -= player["del_angle"]  # Rotate left by 10 degrees
    # Handle right key
    elif keys[pygame.key.key_code(player["right"])]:
        player["angle"] += player["del_angle"]

    # Normalize the angle to keep it within 0-360 degrees range
    if player["angle"] < 0:
        player["angle"] += 360
    elif player["angle"] >= 360:
        player["angle"] -= 360

    # Convert the angle to radians and update the direction vector
    rad_angle = math.radians(player["angle"])
    player["dir"] = [math.cos(rad_angle), math.sin(rad_angle)]

    return player


# Function to print memory usage
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    total_memory = psutil.virtual_memory().total
    usage_percent = (mem_info.rss / total_memory) * 100

    return usage_percent
