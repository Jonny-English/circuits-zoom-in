"""Cross-platform Chinese font configuration for matplotlib."""
import os
import matplotlib
import matplotlib.font_manager as fm

_FONT_CANDIDATES = [
    '/System/Library/Fonts/STHeiti Medium.ttc',        # macOS
    '/System/Library/Fonts/PingFang.ttc',              # macOS
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Linux (Noto)
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Linux (Noto alt)
    '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',     # Linux (WenQuanYi)
    'C:/Windows/Fonts/msyh.ttc',                       # Windows
]


def setup_chinese_font():
    """Detect and configure a Chinese font for matplotlib. Returns font name or None."""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return name
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None
