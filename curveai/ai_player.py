import config as cfg
import numpy as np
from curveai.nn import (
    DQN,
    compute_loss,
    reward,
    preprocessing,
    softmax_action,
    epsilon_greedy_action,
)
from utils.subfunctions import get_random_gap, get_random_position
import math
import torch


def init_ai_player(id, model, iteration):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    player_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    color = player_colors[id]
    start_pos, start_dir = get_random_position()
    start_gap, start_line = get_random_gap()
    game_state_pos = np.array(
        [
            torch.arange(
                int(round(start_pos[0], 0)) - int(cfg.player_size / 2),
                int(round(start_pos[0], 0)) + int(cfg.player_size / 2),
            ),
            torch.arange(
                int(round(start_pos[1], 0)) - int(cfg.player_size / 2),
                int(round(start_pos[1], 0)) + int(cfg.player_size / 2),
            ),
        ]
    )
    gap = False
    player = {
        "pos": start_pos,
        "game_state_pos": game_state_pos,
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
        "outcomes": torch.tensor([0]),
        "optimizer": optimizer,
        "pred_actions": torch.tensor(data=[]),
        "actions": torch.tensor(data=[], dtype=torch.int64),
    }

    return player


def update_ai_player_direction(player, game_state, players):
    # Preprocess player data for the nn
    prepro = preprocessing(player, game_state)
    section, densities = prepro.get_game_variables()

    # get model output
    model = player["model"]
    q_values = model.forward(section, densities)
    pred_action = torch.argmax(q_values)

    if cfg.training:
        # change predicted actions to actual actions by epsilon or softmax exploration
        action = epsilon_greedy_action(q_values)

        # Update pred_actions and actions
        player["pred_actions"] = torch.cat(
            [player["pred_actions"], q_values.unsqueeze(0)], dim=0
        )
        player["actions"] = torch.cat(
            [player["actions"], action.clone().unsqueeze(0)],
            dim=0,
        )

        # update model
        optimizer = player["optimizer"]
        optimizer.zero_grad()
        outcome = reward(player, players)
        player["outcomes"] = torch.cat(
            [player["outcomes"], outcome.unsqueeze(0)], dim=0
        )

        loss = compute_loss(
            player["outcomes"],
            player["pred_actions"],
            player["actions"],
        )

        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()

        # update player variables
        player["model"] = model
        player["optimizer"] = optimizer

    # Change direction depending on model output
    if pred_action == 0:
        player["angle"] -= player["del_angle"]
    if pred_action == 1:
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
