import torch
import random
import numpy as np
import torch.nn as nn
import config as cfg

cfg.set_args()


class DQN(nn.Module):
    def __init__(self, num_players):
        super(DQN, self).__init__()
        self.player_history_size = (
            512  # Number of previous player positions to consider
        )
        self.player_embedding_size = 64  # Embedding size for player history
        self.item_embedding_size = 64  # Embedding size for item list
        self.screen_dim_size = 2  # Dimension size for screen dimensions
        self.hidden_size = 64  # LSTM hidden size
        self.num_fc_layers = cfg.num_layers  # Number fc layers
        self.num_lstm_layers = cfg.num_lstm  # Number of LSTM layers
        self.num_actions = 3  # Right, Left, Nothing
        self.num_items = 16  # Item size set, if less --> padding

        # Embedding layers
        self.player_embedding = nn.Embedding(
            cfg.play_screen_width, self.player_embedding_size
        )
        self.item_embedding = nn.Embedding(
            cfg.play_screen_width, self.item_embedding_size
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            self.player_embedding_size,
            self.hidden_size,
            self.num_lstm_layers,
            batch_first=True,
        )

        # Fully connected layers
        self.fc_layers = []
        self.fc_layers.append(
            nn.Linear(
                self.hidden_size
                + self.item_embedding_size * self.num_items * 3
                + self.screen_dim_size,
                64,
            )
        )
        for _ in range(self.num_fc_layers - 2):
            self.fc_layers.append(nn.Linear(64, 64))
        self.fc_layers.append(nn.Linear(64, self.num_actions))

    def forward(self, player_history, item_list):
        # Embed player history and item list, define screen_dims
        player_history = self.player_embedding(player_history)
        items = self.item_embedding(item_list)
        screen_dims = torch.tensor(
            [cfg.play_screen_width, cfg.screen_height],
            dtype=torch.float16,
            requires_grad=True,
        )

        # Pass player history through LSTM
        _, (h_n, _) = self.lstm(player_history.clone().detach())
        h_n = h_n.squeeze(dim=0)

        # Concatenate LSTM output with item list and screen dimensions
        x = torch.cat((h_n, items.flatten(), screen_dims), dim=0)

        # Pass lstm output through fc layers
        for i in range(self.num_fc_layers - 1):
            x = torch.relu(self.fc_layers[i](x))
        x = self.fc_layers[self.num_fc_layers - 1](x)

        return x


class preprocessing:
    def __init__(self, players, items, screen) -> None:
        self.players = players
        self.items = items
        self.screen = screen

    def get_game_variables(self):
        num_players = len(self.players)

        # Design player_histories, padd to 512
        player_histories = np.zeros(shape=(num_players, 512, 3))
        for player in self.players:
            history = np.array(player["pos_history"])[:512]
            gaps = np.array(player["gap_history"])[:512]
            story = np.zeros((len(gaps), 3))
            for i in range(len(gaps)):
                story[i] = np.append(history[i], gaps[i])
            player_id = player["id"]
            player_histories[player_id, : story.shape[0], :] = story

        player_histories = player_histories.flatten()
        player_histories = torch.from_numpy(player_histories)
        player_histories.requires_grad = True
        player_histories = player_histories.int()

        nn_items = np.zeros(shape=(16, 3))
        for item, i in zip(self.items, range(len(self.items))):
            pos = np.array(item.item["pos"])
            item_id = item.item["id"]
            nn_items[i, :] = np.append(pos, item_id)

        nn_items = nn_items.flatten()
        nn_items = torch.from_numpy(nn_items)
        nn_items.requires_grad = True
        nn_items = nn_items.int()

        return player_histories, nn_items


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
    reward = torch.tensor(reward)

    return reward


def compute_loss(rewards, pred_actions, actions):
    # Compute the cumulative rewards
    cumulative_rewards = torch.cumsum(rewards.clone(), dim=0, dtype=torch.float)

    # Normalize the cumulative rewards
    normalized_rewards = (
        cumulative_rewards - cumulative_rewards.mean()
    ) / cumulative_rewards.std()
    normalized_rewards = normalized_rewards[1:]

    # Transform predicted actions
    pred_actions = torch.softmax(pred_actions, dim=1)

    # Compute the negative log probabilities of the predicted actions
    log_probabilities = -torch.log(pred_actions)

    # Select the log probabilities corresponding to the actual actions taken
    selected_log_probabilities = log_probabilities.gather(
        dim=1, index=actions.unsqueeze(1)
    )

    # Compute the loss as the negative log probabilities weighted by the rewards
    loss = -torch.mean(selected_log_probabilities * rewards)

    return loss


def epsilon_greedy_action(q_values):
    epsilon = 0.2
    if random.random() < epsilon:
        # Explore: choose a random action
        action = torch.tensor(random.choice(range(len(q_values))))
    else:
        # Exploit: choose the action with the highest Q-value
        action = torch.argmax(q_values)
    return action


def softmax_action(q_values):
    temperature = 0.5
    probabilities = torch.softmax(q_values / temperature, dim=0)
    action = torch.multinomial(probabilities, 1).item()
    return action
