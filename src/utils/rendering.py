from gymnasium import Env
import pygame
import numpy as np
from types import MethodType

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
    
    def pump_events(self):
        """Returns ('esc'=True to stop rendering for this checkpoint,
                    'space'=True to toggle fast mode)."""
        if not pygame.display.get_init() or not self._enabled: return None, None
        esc = False; 
        space = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                esc = True
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
        w, h = arr.shape[0], arr.shape[1]
        
        if pygame.display.get_surface() is None:
            w, h = arr.shape[0], arr.shape[1]
            self.screen = pygame.display.set_mode((w, h), self._flags)
            self._size = (w, h)

        pygame.surfarray.blit_array(self.screen, arr)
        self._draw_hud(episode=episode, eval_ep=eval_ep)
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
