import pygame
import config as cfg


def get_player_keys(num_players, screen, color=cfg.white, font=cfg.font):
    player_keys = []
    for i in range(cfg.num_ai_players, num_players):
        keys = {}
        keys["left"] = None
        keys["right"] = None
        while not all(keys.values()):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if not keys["left"]:
                        keys["left"] = pygame.key.name(event.key)
                    elif not keys["right"]:
                        keys["right"] = pygame.key.name(event.key)

            # Draw screen
            screen.fill(color)
            text = font.render(f"Player {i+1}, select keys:", True, (0, 0, 0))
            screen.blit(
                text,
                (
                    cfg.screen_width / 2 - text.get_width() / 2,
                    cfg.screen_height / 2 - text.get_height() / 2 - 50,
                ),
            )

            # get up and down keys
            text = font.render(f"Left: {keys['left']}", True, (0, 0, 0))
            screen.blit(
                text,
                (
                    cfg.screen_width / 2 - text.get_width() / 2,
                    cfg.screen_height / 2 - text.get_height() / 2 + 20,
                ),
            )
            text = font.render(f"Right: {keys['right']}", True, (0, 0, 0))
            screen.blit(
                text,
                (
                    cfg.screen_width / 2 - text.get_width() / 2,
                    cfg.screen_height / 2 - text.get_height() / 2 + 50,
                ),
            )

            pygame.display.update()

        player_keys.append(keys)

    return player_keys
