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
        self.section_embedding_size = 16  # Embedding size for player history
        self.densities_embedding_size = 8  # Embedding size for item list
        self.screen_dim_size = 2  # Dimension size for screen dimensions
        self.hidden_size = 64  # LSTM hidden size
        self.num_fc_layers = cfg.num_layers  # Number fc layers
        self.num_lstm_layers = cfg.num_lstm  # Number of LSTM layers
        self.num_actions = 3  # Right, Left, Nothing
        self.num_items = 16  # Item size set, if less --> padding

        # Embedding layers
        self.section_embedding = nn.Embedding(20, self.section_embedding_size)
        self.density_embedding = nn.Embedding(1, self.densities_embedding_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            self.section_embedding_size,
            self.hidden_size,
            self.num_lstm_layers,
            batch_first=True,
        )

        # Fully connected layers
        self.fc_layers = []
        # self.fc_layers.append(
        #     nn.Linear(
        #         self.player_embedding_size * num_players * 512
        #         + self.item_embedding_size * self.num_items * 3
        #         + self.screen_dim_size,
        #         64,
        #     )
        # )
        self.fc_layers.append(nn.Linear(1474, 64))
        for _ in range(self.num_fc_layers - 2):
            self.fc_layers.append(nn.Linear(64, 64))
        self.fc_layers.append(nn.Linear(64, self.num_actions))

    def forward(self, section, densities):
        # Embed player history and item list, define screen_dims
        sec_emb = self.section_embedding(section)
        den_emb = self.density_embedding(densities)

        screen_dims = torch.tensor(
            [cfg.play_screen_width, cfg.screen_height],
            dtype=torch.float16,
            requires_grad=True,
        )

        # Pass player history through LSTM
        # _, (h_n, _) = self.lstm(player_history.clone().detach())
        # h_n = h_n.squeeze(dim=0)

        # Concatenate LSTM output with item list and screen dimensions
        x = torch.cat((sec_emb.flatten(), den_emb.flatten(), screen_dims), dim=0)

        # Pass lstm output through fc layers
        for i in range(self.num_fc_layers - 1):
            x = torch.relu(self.fc_layers[i](x))
        x = self.fc_layers[self.num_fc_layers - 1](x)

        return x


class preprocessing:
    def __init__(self, player, game_state) -> None:
        self.game_state = game_state
        self.num_sections = 100
        self.player = player

    def get_game_variables(self):
        # Design NN Input: Game state + density of different sections
        # Design sections
        section_width = cfg.play_screen_width // self.num_sections
        section_height = cfg.screen_height // self.num_sections

        # Calculate section density
        densities = np.zeros(self.num_sections)
        curr_section = np.zeros((section_width, section_height))
        for sec in range(self.num_sections):
            section = self.game_state[
                sec * section_width : (sec + 1) * section_width,
                sec * section_height : (sec + 1) * section_height,
            ]
            coll_points = np.where((section >= 1) & (section <= 8))
            sec_density = len(coll_points) / (section_width * section_height)
            densities[sec] = sec_density

            # Check if player is in that section
            x, y = self.player["pos"]
            if (sec * section_width <= x < (sec + 1) * section_width) and (
                sec * section_height <= y < (sec + 1) * section_height
            ):
                curr_section = section

        # Convert to tensors
        curr_section = torch.tensor(
            curr_section, dtype=torch.float16, requires_grad=True
        ).long()
        densities = torch.tensor(
            densities, dtype=torch.float16, requires_grad=True
        ).long()

        return curr_section, densities


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
