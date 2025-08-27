import pygame
import numpy as np
from collections import deque

SLOW_FPS = 9600
FAST_FPS = 60

class Renderer:
    def __init__(
            self,
            env = None,
    ) -> None:
        self.env = env
        self._enabled = True
        self._flags = pygame.DOUBLEBUF
        if not pygame.get_init(): 
            pygame.init()
        if not pygame.display.get_init(): 
            pygame.display.init()
        self.screen = None
        self._size = (8, 8)
        try:
            self.env.unwrapped.metadata["render_fps"] = 9000
        except Exception:
            pass

        self.fast = False                  # False = normal/slow, True = fast
        self.render_interval_slow = 1      # draw every step
        self.render_interval_fast = 16     # draw every 16th step
        self.fps_slow = 60                 # throttle UI in slow mode for smoothness
        self.fps_fast = 0                  # 0 or None => no throttling in fast mode
        self._render_step = 0
        self._clock = pygame.time.Clock()

        self._font = None
        self._hud_pad = 6

        self.chart_enabled = True
        self.chart_height = 120              # pixels
        self._rewards = deque(maxlen=50_000) # keep plenty; we downsample to width

    def _get_font(self):
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            try:
                # Use a common system font; fallback to default if missing
                self._font = pygame.font.SysFont("DejaVu Sans", 16)
            except Exception:
                self._font = pygame.font.Font(None, 16)
        return self._font

    @property
    def _interval(self):
        return self.render_interval_fast if self.fast else self.render_interval_slow
    
    @property
    def _tick(self):
        return self.fps_fast if self.fast else self.fps_slow
    
    def _draw_hud(self, episode=None, eval_ep=None):
        mode = "FAST" if self.fast else "SLOW"
        lines = [f"{mode}  (SPACE to toggle)"]
        if episode is not None:               
            lines.append(f"Episode {episode}")
        if eval_ep is not None:
            lines.append(f"Checkpoint {eval_ep}")

        font = self._get_font()
        pad = self._hud_pad
        texts = [font.render(t, True, (255, 255, 255)) for t in lines]
        spacing = 2
        w = max(s.get_width() for s in texts) + 2 * pad
        h = sum(s.get_height() for s in texts) + (len(texts)-1)*spacing + 2*pad

        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        y = pad
        for s in texts:
            bg.blit(s, (pad, y))
            y += s.get_height() + spacing

        self.screen.blit(bg, (8, 8))


    def _draw_chart(self, x: int, y: int, w: int, h: int) -> None:
        """Draw a simple live line chart of rewards into rect (x, y, w, h)."""
        # Background
        chart_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (16, 16, 16), chart_rect)

        # If no data, draw an axis line and return
        n = len(self._rewards)
        if n == 0:
            pygame.draw.line(self.screen, (64, 64, 64), (x, y + h - 1), (x + w, y + h - 1))
            return

        # Determine the slice to plot: last "w" points max (one per pixel)
        if n <= w:
            data = self._rewards
        else:
            data = list(self._rewards)[-w:]

        # Compute y-scale
        mn = min(data)
        mx = max(data)
        if mx == mn:
            # avoid flat division: show mid baseline
            mx = mn + 1.0

        for frac in (0.0, 0.5, 1.0):
            yy = int(y + (1.0 - frac) * (h - 1))
            pygame.draw.line(self.screen, (40, 40, 40), (x, yy), (x + w, yy))

        # Build polyline points
        # Map i -> x+i, value -> y + scaled_y
        scale = (h - 1) / (mx - mn)
        pts = []
        left = max(0, w - len(data))
        for i, val in enumerate(data):
            px = x + left + i
            py = y + (h - 1) - int((val - mn) * scale)
            pts.append((px, py))

        if len(pts) >= 2:
            pygame.draw.aalines(self.screen, (200, 220, 255), False, pts)

        font = self._get_font()
        fmt = "{:.15f}"
        lo_surf = font.render(fmt.format(mn), True, (200, 200, 200))
        hi_surf = font.render(fmt.format(mx), True, (200, 200, 200))
        self.screen.blit(hi_surf, (x + 4, y + 2))
        self.screen.blit(lo_surf, (x + 4, y + h - lo_surf.get_height() - 2))
        title = font.render("Reward", True, (220, 220, 220))
        self.screen.blit(title, (x + w - title.get_width() - 6, y + 2))
    
    def add_reward(self, r: float) -> None:
        try:
            self._rewards.append(float(r))
        except Exception:
            pass
    
    def pump_events(self):
        if not pygame.display.get_init() or not self._enabled: return None, None
        esc = False; 
        space = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                esc = True # TODO: doesn't work yet
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: esc = True
                elif e.key == pygame.K_SPACE: space = True
        return esc, space
    
    def render(self, episode=None, eval_ep=None) -> None:
        self._render_step += 1
        if (not self._enabled or 
            not pygame.display.get_init() or
            self._render_step % self._interval != 0):
            return
        
        frame = self.env.render()
        if frame is None:
            return
    
        arr = np.ascontiguousarray(frame.swapaxes(0, 1))
        w_env, h_env = arr.shape[0], arr.shape[1]
        
        extra_h = self.chart_height if self.chart_enabled else 0
        total_w, total_h = w_env, h_env + extra_h

        if pygame.display.get_surface() is None:
            self.screen = pygame.display.set_mode((w_env, h_env), self._flags)
            self._size = (w_env, h_env)

        if (
            self.screen is None
            or self._size != (total_w, total_h)
            or pygame.display.get_surface() is None
        ):
            self.screen = pygame.display.set_mode((total_w, total_h), self._flags)
            self._size = (total_w, total_h)

        env_sub = self.screen.subsurface(pygame.Rect(0, 0, w_env, h_env))
        arr = np.ascontiguousarray(frame.swapaxes(0, 1))
        pygame.surfarray.blit_array(env_sub, arr)
        
        self._draw_hud(episode=episode, eval_ep=eval_ep)
        if self.chart_enabled and self.chart_height > 0:
            self._draw_chart(0, h_env, w_env, self.chart_height)
        pygame.display.flip()

        self._clock.tick(self._tick)

    def close(self) -> None:
        # TODO: This doesn't work. It crashes the pygame window
        self._enabled = False
        self.screen = None
        self._size = (0,0)
        if pygame.display.get_init():
            pygame.display.quit()
        try:
            pygame.event.pump()
            pygame.event.clear()
        except pygame.error as e:
            print(e)

    def toggle_speed(self) -> None:
        self.fast = not self.fast
