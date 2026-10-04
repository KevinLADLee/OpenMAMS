"""Compatibility entry point; presentation code now lives in recorder.demo."""
from .demo import caption_capture, caption_overlay, main, render_demo, wrap_caption

__all__ = ['caption_capture', 'caption_overlay', 'main', 'render_demo', 'wrap_caption']

if __name__ == '__main__':
    main()
