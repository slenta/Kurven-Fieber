import config as cfg
from nn import DQN, compute_loss, reward, preprocessing
from subfunctions import get_random_gap, get_random_position
import math
import torch


def init_ai_player(id, model):
    optimizer = torch.optim.sgd(model.parameters(), lr=cfg.learning_rate)
    color = cfg.colors[id]

    start_pos, start_dir = get_random_position()
    start_gap, start_line = get_random_gap()
    gap = False
    player = {
        "pos": start_pos,
        "dir": start_dir,
        "angle": 0,
        "color": color,
        "alive": True,
        "length": 1,
        "speed": cfg.speed,
        "id": id,
        "pos_history": [start_pos],
        "size": cfg.player_size,
        "gap": gap,
        "gap_history": [gap],
        "gap_timer": start_gap,
        "line_timer": start_line,
        "del_angle": 5,
        "items": [],
        "item_timer": [],
        "left": -1,
        "right": 1,
        "ai": True,
        "model": model,
        "outcomes": [],
        "optimizer": optimizer,
    }

    return model, player


def update_ai_player_direction(player, players, items, screen):
    # Preprocess player data for the nn
    prepro = preprocessing(players, items, screen)
    players_histories, items = prepro.get_game_variables()

    # get model output
    model = player["model"]
    output = model.forward(players_histories, items)
    print(output)

    # update model
    optimizer = player["optimizer"]
    optimizer.zero_grad()
    outcome = reward(player, players)
    player["outcomes"].append(outcome)

    loss = compute_loss(player["outcomes"])
    loss.backwards()
    optimizer.step()

    # update player variables
    player["model"] = model
    player["optimizer"] = optimizer

    # Change direction depending on model output
    if output >= 0.6:
        player["angle"] -= player["del_angle"]
    if output <= 0.4:
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
