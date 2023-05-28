import torch
import torch.nn as nn
import config as cfg
from subfunctions import get_random_gap, get_random_position

cfg.set_args()


class DQN(nn.Module):
    def __init__(self, num_players):
        super(DQN, self).__init__()
        self.player_history_size = 64  # Number of previous player positions to consider
        self.player_embedding_size = 32  # Embedding size for player history
        self.item_embedding_size = 16  # Embedding size for item list
        self.screen_dim_size = 2  # Dimension size for screen dimensions
        self.hidden_size = 64  # LSTM hidden size
        self.num_layers = 1  # Number of LSTM layers
        self.num_actions = 3  # Left, Right, Nothing
        self.num_items = 15  # Item size set, if less --> padding

        # Embedding layers
        self.player_embedding = nn.Embedding(num_players, self.player_embedding_size)
        self.item_embedding = nn.Embedding(self.num_items, self.item_embedding_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            self.player_embedding_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            self.hidden_size
            + self.item_embedding_size * num_players
            + self.screen_dim_size,
            64,
        )
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.num_actions)

    def forward(self, player_history, item_list, screen_dims):
        # Embed player history and item list
        player_history = self.player_embedding(player_history)
        item_list = self.item_embedding(item_list)

        # Pass player history through LSTM
        _, (h_n, _) = self.lstm(player_history)
        h_n = h_n.squeeze(dim=0)

        # Concatenate LSTM output with item list and screen dimensions
        x = torch.cat((h_n, item_list.view(item_list.size(0), -1), screen_dims), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class preprocessing:
    def __init__(self, players, items, screen) -> None:
        self.players = players
        self.items = items
        self.screen = screen

    def get_game_variables(self):
        num_players = len(self.players)
        history_len = len(self.players[0]["pos_history"])

        player_histories = torch.zeros(
            size=(num_players, history_len, history_len),
            dtype=torch.float16,
            requires_grad=True,
        )
        for player in self.players:
            history = player["pos_history"]
            gaps = player["gap_history"]
            player_id = player["id"]
            player_histories[player_id, :, :] = torch.tensor(
                [history, gaps], dtype=torch.float16, requires_grad=True
            )

        items = torch.ones(size=(15, 3), dtype=torch.float16, requires_grad=True) * 9999
        for item, i in zip(items, range(len(items))):
            pos = item["pos"]
            item_id = item["id"]
            items[i, :] = torch.tensor(
                [pos, item_id], dtype=torch.float16, requires_grad=True
            )

        return player_histories, items


def reward(player, players):
    # loss for the other players actions
    others_rewards = 0
    for play in players:
        if play != player:
            if play["alive"] == False:
                others_rewards += 1

    # Compute loss for staying alive
    alive = player["alive"]
    if alive == True:
        alive_reward = 1
    else:
        alive_reward = -1

    # Total loss
    reward = alive_reward + others_rewards

    return reward


def loss(rewards):
    # Compute the cumulative rewards
    cumulative_rewards = torch.cumsum(
        torch.tensor(rewards[::-1], dtype=torch.float32), dim=0
    )[::-1]

    # Normalize the cumulative rewards
    normalized_rewards = (
        cumulative_rewards - cumulative_rewards.mean()
    ) / cumulative_rewards.std()

    # Compute the loss as the negative log probabilities weighted by the rewards
    loss = -torch.log(normalized_rewards)

    return loss


def init_ai_player(num_players, id):
    model = DQN(num_players)
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
    }

    return model, player
