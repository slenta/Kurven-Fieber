import pygame
import config as cfg


def display_text(screen, text, font, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    screen.blit(text_surface, text_rect)


def draw_screen(screen, win_counts, screen_color=cfg.black, outline_color=cfg.white):
    screen.fill(screen_color)  # Fill the screen with black color
    pygame.draw.rect(
        screen,
        cfg.white,
        (0, 0, cfg.screen_width - cfg.score_section_width, cfg.screen_height),
        5,
    )
    pygame.display.set_caption("Curve Fever")

    # Draw the score section
    pygame.draw.rect(
        screen,
        cfg.black,
        (
            cfg.screen_width - cfg.score_section_width,
            0,
            cfg.score_section_width,
            cfg.screen_height,
        ),
    )
    pygame.draw.rect(
        screen,
        cfg.white,
        (0, 0, cfg.screen_width - cfg.score_section_width, cfg.screen_height),
        5,
    )
    pygame.draw.rect(
        screen, outline_color, (0, 0, cfg.screen_width, cfg.screen_height), 5
    )

    # Display win counts
    for player_id, win_count in win_counts.items():
        player_text = f"Player {player_id}: {win_count} wins"
        display_text(
            screen,
            player_text,
            font=cfg.font,
            color=cfg.white,
            x=cfg.screen_width - cfg.score_section_width + 90,
            y=20 + player_id * 20,
        )

    return screen


def draw_players(players, screen):
    for player in players:
        pygame.draw.circle(screen, player["color"], player["pos"], cfg.player_size // 2)
        if player["gap"]:
            pygame.draw.circle(
                screen, cfg.black, player["pos_history"][1], cfg.player_size // 2
            )
        prev_pos = None
        for pos, gap in zip(player["pos_history"], player["gap_history"]):
            if gap:
                prev_pos = None
                continue
            if prev_pos is not None:
                pygame.draw.line(
                    screen,
                    player["color"],
                    prev_pos,
                    pos,
                    cfg.player_size,
                )
                prev_pos = pos


def display_winner(screen, win_counts):
    # Clear the score section
    pygame.draw.rect(
        screen,
        cfg.black,
        (
            cfg.screen_width - cfg.score_section_width,
            0,
            cfg.score_section_width,
            cfg.screen_height,
        ),
    )
    pygame.draw.rect(screen, cfg.white, (0, 0, cfg.screen_width, cfg.screen_height), 5)
    pygame.draw.rect(
        screen,
        cfg.white,
        (0, 0, cfg.screen_width - cfg.score_section_width, cfg.screen_height),
        5,
    )
    # Display win counts
    for player_id, win_count in win_counts.items():
        player_text = f"Player {player_id}: {win_count} wins"
        display_text(
            screen,
            player_text,
            font=cfg.font,
            color=cfg.white,
            x=cfg.screen_width - cfg.score_section_width + 90,
            y=20 + player_id * 20,
        )
