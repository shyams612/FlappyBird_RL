# rendering/renderer.py
"""
Renderer: the ONLY file allowed to import pygame.

Takes a GameState snapshot and draws it.  Never mutates state.
Instantiated once by play.py; never touched during headless RL training.

Colour palette:
  Sky      #87CEEB   Pipe     #2ECC40   Pipe border  #27AE60
  Bird     #F4D03F   Ground   #8B6914   HUD bg       #00000099
"""

from __future__ import annotations
import pygame

from envs.game_state import GameState
from envs.entities import Rect, PipeType


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

SKY      = (135, 206, 235)
GROUND   = (139, 105, 20)
BIRD_COL = (244, 208, 63)
BIRD_BDR = (180, 140, 20)
HUD_BG   = (0, 0, 0, 150)
WHITE    = (255, 255, 255)
RED      = (220, 50, 50)
GREEN    = (50, 200, 80)

GROUND_H = 40   # pixels of ground strip at the bottom

# HARD pipe — flat colour (no gradient, instant death)
PIPE_COLOURS: dict[PipeType, tuple[tuple, tuple]] = {
    PipeType.HARD: ((46, 204, 64), (39, 174, 96)),   # green
}

# Gradient colours for non-hard pipes: (gap_edge_colour, far_edge_colour, border_colour)
# gap_edge = dark/saturated  — high damage zone
# far_edge = light/washed    — low damage zone
PIPE_GRADIENT: dict[PipeType, tuple[tuple, tuple, tuple]] = {
    PipeType.SOFT:    ((30,  90, 200), (180, 215, 255), (41,  128, 185)),  # blue
    PipeType.BRITTLE: ((200,  40,   0), (255, 190, 140), (180,  60,   0)),  # orange-red
    PipeType.FOAM:    ((190,  50, 130), (255, 210, 235), (170,  60, 130)),  # pink
}

GRADIENT_STRIP_H = 2   # height of each colour strip in pixels

# Shattered pipe overlay — semi-transparent dark tint drawn over a broken pipe
SHATTER_TINT = (30, 30, 30, 160)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Flappy RL")
        self._font_sm = pygame.font.SysFont("monospace", 16)
        self._font_lg = pygame.font.SysFont("monospace", 36, bold=True)
        # Persistent surface for semi-transparent HUD backgrounds
        self._hud_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, state: GameState, debug: bool = False) -> None:
        s = self.screen
        s.fill(SKY)

        self._draw_pipes(s, state)
        self._draw_bird(s, state)
        self._draw_ground(s)
        self._draw_hud(s, state)

        if debug:
            self._draw_debug(s, state)

        pygame.display.flip()

    def draw_death_screen(self, state: GameState) -> None:
        """Overlay shown when the episode ends."""
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        msg  = self._font_lg.render("GAME OVER", True, WHITE)
        sub  = self._font_sm.render(f"Score: {state.score}   Press R to restart", True, WHITE)
        cx = self.screen_w // 2
        cy = self.screen_h // 2
        self.screen.blit(msg, msg.get_rect(center=(cx, cy - 30)))
        self.screen.blit(sub, sub.get_rect(center=(cx, cy + 20)))
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Internal draw helpers
    # ------------------------------------------------------------------

    def _draw_pipes(self, s: pygame.Surface, state: GameState) -> None:
        for pipe in state.pipes:
            if pipe.pipe_type == PipeType.HARD:
                # Flat colour — instant death, no gradient needed
                fill, border = PIPE_COLOURS[PipeType.HARD]
                self._filled_rect(s, pipe.top_rect, fill, border)
                bot = Rect(pipe.x, pipe.gap_bottom, pipe.width,
                           self.screen_h - GROUND_H - pipe.gap_bottom)
                if bot.h > 0:
                    self._filled_rect(s, bot, fill, border)
            else:
                # Gradient — dark at gap edge, light far from gap
                gap_col, far_col, border = PIPE_GRADIENT[pipe.pipe_type]
                # Top segment: gap edge at bottom (gap_top), far edge at top (y=0)
                self._draw_gradient_segment(
                    s, pipe, y_start=0, y_end=int(pipe.gap_top),
                    gap_col=gap_col, far_col=far_col,
                    gap_is_at_bottom=True,
                )
                # Bottom segment: gap edge at top (gap_bottom), far edge at screen bottom
                bot_h = self.screen_h - GROUND_H - int(pipe.gap_bottom)
                if bot_h > 0:
                    self._draw_gradient_segment(
                        s, pipe, y_start=int(pipe.gap_bottom),
                        y_end=int(pipe.gap_bottom) + bot_h,
                        gap_col=gap_col, far_col=far_col,
                        gap_is_at_bottom=False,
                    )
                # Border over both segments
                if pipe.top_rect.h > 0:
                    pygame.draw.rect(s, border,
                                     (int(pipe.x), 0,
                                      int(pipe.width), int(pipe.gap_top)), 2)
                bot_h2 = self.screen_h - GROUND_H - int(pipe.gap_bottom)
                if bot_h2 > 0:
                    pygame.draw.rect(s, border,
                                     (int(pipe.x), int(pipe.gap_bottom),
                                      int(pipe.width), bot_h2), 2)

                # Shattered overlay — dark tint to show pipe is broken
                if pipe.shattered:
                    surf = pygame.Surface((int(pipe.width), self.screen_h), pygame.SRCALPHA)
                    surf.fill(SHATTER_TINT)
                    s.blit(surf, (int(pipe.x), 0))

    def _draw_gradient_segment(
        self,
        s: pygame.Surface,
        pipe,
        y_start: int,
        y_end: int,
        gap_col: tuple,
        far_col: tuple,
        gap_is_at_bottom: bool,
    ) -> None:
        """
        Draw a pipe segment with a vertical colour gradient.
        gap_is_at_bottom=True  → gap edge is at y_end   (top segment)
        gap_is_at_bottom=False → gap edge is at y_start  (bottom segment)
        """
        total_h = max(y_end - y_start, 1)
        x, w = int(pipe.x), int(pipe.width)

        y = y_start
        while y < y_end:
            strip_h = min(GRADIENT_STRIP_H, y_end - y)
            # t=0 → far edge colour, t=1 → gap edge colour
            if gap_is_at_bottom:
                t = (y - y_start) / total_h        # increases toward gap
            else:
                t = 1.0 - (y - y_start) / total_h  # decreases away from gap

            r = int(far_col[0] + t * (gap_col[0] - far_col[0]))
            g = int(far_col[1] + t * (gap_col[1] - far_col[1]))
            b = int(far_col[2] + t * (gap_col[2] - far_col[2]))
            pygame.draw.rect(s, (r, g, b), (x, y, w, strip_h))
            y += strip_h

    def _draw_bird(self, s: pygame.Surface, state: GameState) -> None:
        r = state.bird.rect
        pygame.draw.ellipse(s, BIRD_COL,
                            (int(r.x), int(r.y), int(r.w), int(r.h)))
        pygame.draw.ellipse(s, BIRD_BDR,
                            (int(r.x), int(r.y), int(r.w), int(r.h)), 2)
        # Simple eye
        eye_x = int(r.x + r.w * 0.70)
        eye_y = int(r.y + r.h * 0.28)
        pygame.draw.circle(s, (30, 30, 30), (eye_x, eye_y), 4)
        pygame.draw.circle(s, WHITE,        (eye_x, eye_y), 2)

    def _draw_ground(self, s: pygame.Surface) -> None:
        pygame.draw.rect(s, GROUND,
                         (0, self.screen_h - GROUND_H, self.screen_w, GROUND_H))

    def _draw_hud(self, s: pygame.Surface, state: GameState) -> None:
        # Score
        score_surf = self._font_lg.render(str(state.score), True, WHITE)
        s.blit(score_surf, score_surf.get_rect(centerx=self.screen_w // 2, top=12))

        # Health bar (top-left)
        self._draw_health_bar(s, state.health)

        # Frame counter (small, bottom-left)
        frame_surf = self._font_sm.render(f"frame {state.frame}", True, WHITE)
        s.blit(frame_surf, (8, self.screen_h - GROUND_H - 22))

    def _draw_health_bar(self, s: pygame.Surface, health: float) -> None:
        bar_x, bar_y, bar_w, bar_h = 12, 12, 160, 18
        pygame.draw.rect(s, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
        fill = int(bar_w * max(health, 0) / 100)
        colour = GREEN if health > 50 else (220, 180, 0) if health > 25 else RED
        if fill > 0:
            pygame.draw.rect(s, colour, (bar_x, bar_y, fill, bar_h))
        pygame.draw.rect(s, WHITE, (bar_x, bar_y, bar_w, bar_h), 1)
        label = self._font_sm.render(f"HP {int(health)}", True, WHITE)
        s.blit(label, (bar_x + 4, bar_y + 1))

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

    def _draw_debug(self, s: pygame.Surface, state: GameState) -> None:
        # Bird hitbox
        r = state.bird.rect
        pygame.draw.rect(s, RED,
                         (int(r.x), int(r.y), int(r.w), int(r.h)), 1)

        # Pipe hitboxes
        for pipe in state.pipes:
            top = pipe.top_rect
            pygame.draw.rect(s, RED,
                             (int(top.x), int(top.y), int(top.w), int(top.h)), 1)
            bot = Rect(pipe.x, pipe.gap_bottom, pipe.width,
                       self.screen_h - GROUND_H - pipe.gap_bottom)
            if bot.h > 0:
                pygame.draw.rect(s, RED,
                                 (int(bot.x), int(bot.y), int(bot.w), int(bot.h)), 1)

        # Velocity vector
        bx, by = int(state.bird.pos.x), int(state.bird.pos.y)
        vx = bx + int(state.bird.vel.x * 4)
        vy = by + int(state.bird.vel.y * 4)
        pygame.draw.line(s, (255, 80, 80), (bx, by), (vx, vy), 2)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _filled_rect(s: pygame.Surface, rect: Rect,
                     fill: tuple, border: tuple, border_w: int = 2) -> None:
        r = (int(rect.x), int(rect.y), int(rect.w), int(rect.h))
        if rect.h > 0 and rect.w > 0:
            pygame.draw.rect(s, fill,   r)
            pygame.draw.rect(s, border, r, border_w)