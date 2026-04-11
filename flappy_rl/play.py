# play.py
"""
Human-controlled play entry point.

Usage:
    python play.py                          # default config, discrete physics
    python play.py --config config/config1.yaml
    python play.py --physics continuous     # override physics mode
    python play.py --debug                  # show hitboxes + velocity vector

Controls:
    SPACE / UP    flap
    R             restart after death
    D             toggle debug overlay
    ESC / Q       quit
"""

from __future__ import annotations
import argparse
import sys
import pygame

from envs.config import EnvConfig, PhysicsMode
from envs.game_state import GameState
from rendering.renderer import Renderer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default=None,
                   help="Path to a YAML config file")
    p.add_argument("--physics", choices=["discrete", "continuous"], default=None,
                   help="Override physics mode from config")
    p.add_argument("--debug",   action="store_true",
                   help="Enable debug overlay")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Config ---
    if args.config:
        cfg = EnvConfig.from_yaml(args.config)
    else:
        cfg = EnvConfig()

    if args.physics:
        cfg.physics_mode = PhysicsMode(args.physics)

    # --- Pygame init ---
    pygame.init()
    clock   = pygame.time.Clock()
    renderer = Renderer(cfg.screen_w, cfg.screen_h)
    debug    = args.debug

    # --- Episode state ---
    state  = GameState.reset(cfg)
    dead   = False

    print(f"[play] {cfg.screen_w}x{cfg.screen_h}  physics={cfg.physics_mode.value}  fps={cfg.fps}")
    print("[play] SPACE/UP=flap  R=restart  D=debug  ESC/Q=quit")

    while True:
        dt = clock.tick(cfg.fps) / 1000.0   # seconds since last frame

        # ----------------------------------------------------------------
        # Event handling
        # ----------------------------------------------------------------
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_d:
                    debug = not debug
                if event.key == pygame.K_r and dead:
                    state = GameState.reset(cfg)
                    dead  = False

        # Hold-key flap: feels more natural than single keydown event
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        # ----------------------------------------------------------------
        # Simulation
        # ----------------------------------------------------------------
        if not dead:
            state = state.step(action, dt=dt)
            if state.terminated:
                dead = True

        # ----------------------------------------------------------------
        # Render
        # ----------------------------------------------------------------
        if dead:
            renderer.draw(state, debug=debug)
            renderer.draw_death_screen(state)
        else:
            renderer.draw(state, debug=debug)

    # unreachable — exits via sys.exit() above


if __name__ == "__main__":
    main()