import random
import config as cfg
import pygame
import math

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


def do_points_intersect(point1, point2, radius=5):
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


def check_for_collisions(players):
    for player in players:
        if not player["alive"]:
            continue
        player_pos = player["pos"]
        player_history = player["pos_history"]
        gap_history = player["gap_history"]
        player_size = player["size"]

        # check collision with own line
        for pos, gap in zip(player_history[10:], gap_history[10:]):
            if not gap:
                intersect = do_points_intersect(pos, player_pos)
                if intersect:
                    player["alive"] = False
                    break

        # check collision with other player lines
        for other_player in players:
            if other_player["id"] == player["id"]:
                continue
            other_history = other_player["pos_history"]
            other_gap = other_player["gap_history"]
            for pos, gap in zip(other_history, other_gap):
                if not gap:
                    intersect = do_points_intersect(pos, player_pos)
                    if intersect:
                        player["alive"] = False
                        break

        # check collision with screen edges
        if (
            player_pos[0] < player_size
            or player_pos[0] > cfg.screen_width - cfg.score_section_width - player_size
            or player_pos[1] < player_size
            or player_pos[1] > cfg.screen_height - player_size
        ):
            player["alive"] = False
            break


def update_player_direction(player, player_keys):
    # update directions
    keys = pygame.key.get_pressed()
    # Handle left key
    if keys[pygame.key.key_code(player_keys["left"])]:
        player["angle"] -= 5  # Rotate left by 10 degrees
    # Handle right key
    elif keys[pygame.key.key_code(player_keys["right"])]:
        player["angle"] += 5

    # Normalize the angle to keep it within 0-360 degrees range
    if player["angle"] < 0:
        player["angle"] += 360
    elif player["angle"] >= 360:
        player["angle"] -= 360

    # Convert the angle to radians and update the direction vector
    rad_angle = math.radians(player["angle"])
    player["dir"] = [math.cos(rad_angle), math.sin(rad_angle)]

    return player
