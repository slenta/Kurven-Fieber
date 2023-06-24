import pygame
import torch
import numpy as np
from player_functions import get_random_position, get_random_gap
from player_functions import move_players, get_winner
from display_functions import display_text, draw_players, display_winner, draw_screen
from items import render_items, power_down
import config as cfg


def reset_game(players, win_counts, player_keys=None, screen=None):
    # Set all players to be alive and reset their positions and directions
    gap = False
    for player in players:
        player["alive"] = True
        start_pos, start_dir = get_random_position()
        start_gap, start_line = get_random_gap()
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
        player["game_state_pos"] = game_state_pos
        player["speed"] = cfg.speed
        if player["ai"] == False:
            player["left"] = player_keys[player["id"] - cfg.num_ai_players]["left"]
            player["right"] = player_keys[player["id"] - cfg.num_ai_players]["right"]
        player["pos"] = start_pos
        player["dir"] = start_dir
        player["gap_timer"] = start_gap
        player["line_timer"] = start_line
        player["pos_history"] = [start_pos]
        player["del_angle"] = 5
        player["gap"] = gap
        player["gap_history"] = [gap]
        player["items"] = []
        player["item_timer"] = []
        if player["ai"]:
            player["outcomes"] = torch.tensor([0])
            player["pred_actions"] = torch.tensor(data=[])
            player["actions"] = torch.tensor(data=[], dtype=torch.int64)

    # Reset winner
    winner = None
    items = []

    # Refresh the screen
    if screen:
        draw_screen(screen, win_counts)
        pygame.display.flip()

        # Pause briefly to give players time to reposition
        pygame.time.wait(1000)

    game_state = init_game_state()

    return winner, items, game_state


def game_loop(players, player_keys, screen, win_counts, last_spawn_time):
    # set up game variables
    game_over = False
    running = True
    winner = None
    items = []
    game_state = init_game_state()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game_over:
                    winner, items, game_state = reset_game(
                        players, win_counts, player_keys, screen
                    )
                    game_over = False
                elif event.key == pygame.K_ESCAPE and game_over:
                    # Save nn models
                    for i in range(cfg.num_ai_players):
                        if cfg.training:
                            player_iter = players[i]["iteration"]
                            torch.save(
                                players[i]["model"].state_dict(),
                                f"{cfg.save_model_path}\model_checkpoint_iter_{player_iter}_player_{i}.pth",
                            )
                    running = False

        # Get current time in milliseconds
        current_time = pygame.time.get_ticks()

        # Check if it's time to spawn a new item (every 10 seconds)
        if current_time - last_spawn_time >= 1000:
            if len(items) <= cfg.max_items:
                items, game_state = render_items(items, game_state, screen)

            # Update the last spawn time
            last_spawn_time = current_time

        # move all players and check for collisions
        players, game_state = move_players(players, items, game_state)

        # check if any players are alive
        alive_players = [player for player in players if player["alive"]]
        if len(alive_players) <= 1:
            # Set game over to true
            game_over = True

            # Stop all players when the game is over
            for player in players:
                player["dir"] = "stop"

            # Get winning player
            if len(alive_players) == 1:
                if winner == None:
                    winner = get_winner(players, win_counts)
                    display_winner(screen, win_counts)

            # Display options to restart or quit
            display_text(
                screen,
                "Game Over",
                font=cfg.font,
                color=cfg.white,
                x=(cfg.screen_width - cfg.score_section_width) / 2,
                y=cfg.screen_height / 2 - 50,
            )
            display_text(
                screen,
                "Press Space to Restart",
                font=cfg.font,
                color=cfg.white,
                x=(cfg.screen_width - cfg.score_section_width) / 2,
                y=cfg.screen_height / 2 + 10,
            )
            display_text(
                screen,
                "Press Esc to Quit",
                font=cfg.font,
                color=cfg.white,
                x=(cfg.screen_width - cfg.score_section_width) / 2,
                y=cfg.screen_height / 2 + 50,
            )

            # Event handling for options
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        for player, player_key in zip(players, player_keys):
                            for i in range(len(player["items"])):
                                player, player_key = power_down(
                                    player["items"][i],
                                    player["item_timer"][i],
                                    player,
                                )

                        winner, items, game_state = reset_game(
                            players, win_counts, player_keys, screen
                        )
                    elif event.key == pygame.K_ESCAPE:
                        running = False

        # Draw players
        draw_players(players, screen)

        pygame.display.update()
        cfg.clock.tick(60)

    pygame.quit()


def init_game_state():
    game_state = np.zeros(shape=(cfg.play_screen_width - 10, cfg.screen_height - 10))
    game_state = np.pad(game_state, 5, constant_values=-1)
    return game_state


def simulation_game_loop(players, win_counts, last_spawn_time, iteration):
    # set up game variables
    game_over = False
    winner = None
    items = []
    game_state = init_game_state()
    steps = 0

    while iteration < cfg.max_iter:
        # Get current time in milliseconds
        current_time = pygame.time.get_ticks()

        # Check if it's time to spawn a new item (every 10 seconds)
        if current_time - last_spawn_time >= 1000:
            if len(items) <= cfg.max_items:
                items, game_state = render_items(items, game_state)

            # Update the last spawn time
            last_spawn_time = current_time

        # move all players and check for collisions
        players, game_state = move_players(players, items, game_state)
        steps += 1

        # check if any players are alive
        alive_players = [player for player in players if player["alive"]]
        if len(alive_players) <= 1:
            # Set game over to true
            game_over = True

            # Stop all players when the game is over
            for player in players:
                player["dir"] = "stop"

            # Get winning player, update training iteration and reset game
            if len(alive_players) == 1:
                if winner == None:
                    winner = get_winner(players, win_counts)
                    iteration += 1
                    print(iteration, steps, current_time)
                    steps = 0

                    for player in players:
                        for i in range(len(player["items"])):
                            player = power_down(
                                player["items"][i],
                                player["item_timer"][i],
                                player,
                            )

                    winner, items, game_state = reset_game(players, win_counts)

        cfg.clock.tick(60)

    # If max iteration is reached: Save nn state for all players
    for i in range(cfg.num_ai_players):
        torch.save(
            players[i]["model"].state_dict(),
            f"{cfg.save_model_path}model_checkpoint_iter_{iteration}_player_{i}.pth",
        )

    pygame.quit()
