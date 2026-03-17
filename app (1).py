"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   GALAXY COLLISION  —  Interactive Streamlit N-Body Simulator               ║
║   Run:  streamlit run app.py                                                ║
║   Deps: streamlit numpy numba matplotlib pillow                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import os, time, io, math
os.environ.setdefault("NUMBA_NUM_THREADS", "10")   # i7-1255U sweet-spot

# ── third-party ────────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numba import njit, prange

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Galaxy Collision Simulator",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  — deep-space aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

/* ── root palette ── */
:root {
    --void:    #02020a;
    --surface: #080818;
    --glass:   rgba(15,15,40,0.85);
    --accent1: #7DF9FF;   /* electric cyan  */
    --accent2: #FF6BFF;   /* neon magenta   */
    --accent3: #FFD700;   /* star gold      */
    --text:    #c8d8ff;
    --muted:   #556080;
}

/* ── global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--void) !important;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--glass) !important;
    border-right: 1px solid rgba(125,249,255,0.12);
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.08em;
}
h1 {
    font-size: 1.9rem !important;
    font-weight: 900;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin-bottom: 0 !important;
}
h3 { color: var(--accent1) !important; font-size: 0.85rem !important; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid rgba(125,249,255,0.18);
    border-radius: 8px;
    padding: 10px 14px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--accent1) !important;
    font-size: 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
}

/* ── buttons ── */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.1em;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 4px;
    border: 1px solid var(--accent1) !important;
    background: transparent !important;
    color: var(--accent1) !important;
    transition: all 0.2s;
    padding: 8px 20px;
    width: 100%;
}
.stButton > button:hover {
    background: var(--accent1) !important;
    color: var(--void) !important;
    box-shadow: 0 0 18px var(--accent1);
}

/* ── image frame ── */
[data-testid="stImage"] {
    border: 1px solid rgba(125,249,255,0.15);
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(125,249,255,0.06);
}

/* ── expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid rgba(125,249,255,0.12) !important;
    border-radius: 6px;
}
summary { font-family: 'Orbitron', monospace !important; font-size: 0.8rem !important; }

/* ── selectbox ── */
[data-testid="stSelectbox"] > div {
    background: var(--surface) !important;
    border: 1px solid rgba(125,249,255,0.25) !important;
    color: var(--text) !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--accent1); border-radius: 2px; }

/* ── status badge ── */
.status-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
}
.badge-running  { border: 1px solid #7DF9FF; color: #7DF9FF; background: rgba(125,249,255,0.08); }
.badge-paused   { border: 1px solid #FFD700; color: #FFD700; background: rgba(255,215,0,0.08); }
.badge-idle     { border: 1px solid #556080; color: #556080; background: transparent; }

/* ── subtitle ── */
.subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    margin-top: 2px;
    margin-bottom: 18px;
}
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(125,249,255,0.1);
    padding-bottom: 4px;
    margin: 16px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA PHYSICS KERNEL
# ══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True, fastmath=True)
def _direct_forces(pos: np.ndarray, mass: np.ndarray,
                   G: float, eps2: float) -> np.ndarray:
    """
    Parallelised O(N²/2) direct summation.
    Vectorised over the outer loop via prange — Numba distributes across
    P-cores (high IPC) and E-cores (high throughput) automatically.
    Safe for N ≤ 3 000; above that we use the tiled vectorised NumPy path.
    """
    n  = pos.shape[0]
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        fx = 0.0; fy = 0.0
        xi = pos[i, 0]; yi = pos[i, 1]
        for j in range(n):
            if i == j:
                continue
            dx  = pos[j, 0] - xi
            dy  = pos[j, 1] - yi
            r2  = dx*dx + dy*dy + eps2
            inv = G * mass[j] / (r2 * math.sqrt(r2))
            fx += dx * inv
            fy += dy * inv
        ax[i] = fx
        ay[i] = fy
    return np.column_stack((ax, ay))


def _tiled_forces(pos: np.ndarray, mass: np.ndarray,
                  G: float, eps2: float) -> np.ndarray:
    """
    Vectorised NumPy O(N²) for N > 3 000.
    Processes in tiles of TILE rows to stay within L2 cache (≈1.25 MB tile).
    Avoids a full N×N distance matrix so RAM stays manageable.
    """
    n    = len(pos)
    TILE = 512
    acc  = np.zeros((n, 2), dtype=np.float64)

    for i0 in range(0, n, TILE):
        i1  = min(i0 + TILE, n)
        ri  = pos[i0:i1, np.newaxis, :]          # (tile, 1, 2)
        rj  = pos[np.newaxis, :, :]              # (1, N, 2)
        dr  = rj - ri                            # (tile, N, 2)
        r2  = (dr**2).sum(-1) + eps2             # (tile, N)
        inv = G * mass[np.newaxis, :] / (r2 * np.sqrt(r2))
        # zero self-interaction
        idx = np.arange(i0, i1)[:, np.newaxis]
        inv[idx - i0, idx.T] = 0.0              # broadcast trick for diag
        acc[i0:i1] = (inv[:, :, np.newaxis] * dr).sum(1)

    return acc


def compute_forces(pos, mass, G, softening):
    eps2 = softening * softening
    if len(pos) <= 2500:
        return _direct_forces(pos, mass, G, eps2)
    return _tiled_forces(pos, mass, G, eps2)


# ══════════════════════════════════════════════════════════════════════════════
#  LEAPFROG INTEGRATOR  DKD
# ══════════════════════════════════════════════════════════════════════════════

def leapfrog_step(pos, vel, mass, dt, G, softening):
    pos_h = pos + 0.5 * dt * vel
    acc   = compute_forces(pos_h, mass, G, softening)
    vel_n = vel + dt * acc
    pos_n = pos_h + 0.5 * dt * vel_n
    return pos_n, vel_n


# ══════════════════════════════════════════════════════════════════════════════
#  GALAXY INITIALISER
# ══════════════════════════════════════════════════════════════════════════════

def make_galaxy(n, cx, cy, vx, vy, angle, disk_r, G_val, rng):
    r   = rng.exponential(disk_r / 3.0, n).clip(0.01)
    phi = rng.uniform(0, 2*np.pi, n)

    x = r * np.cos(phi) + rng.normal(0, 0.015, n)
    y = r * np.sin(phi) + rng.normal(0, 0.015, n)

    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca*x - sa*y + cx
    yr = sa*x + ca*y + cy

    M_enc  = np.tanh(r / (disk_r * 0.3))
    v_circ = np.sqrt(np.clip(G_val * M_enc / (r + 0.05), 0, None))

    tx = -np.sin(phi); ty = np.cos(phi)
    vx_ = ca*tx - sa*ty
    vy_ = sa*tx + ca*ty

    pos  = np.column_stack([xr, yr])
    vel  = np.column_stack([vx_*v_circ + vx,  vy_*v_circ + vy])
    mass = np.full(n, 1.0 / n, dtype=np.float64)
    return pos.astype(np.float64), vel.astype(np.float64), mass


def init_simulation(n_stars, G_val):
    rng  = np.random.default_rng(int(time.time()) % (2**31))
    half = n_stars // 2

    pos1, vel1, m1 = make_galaxy(half,   cx=-2.4, cy= 0.25,
                                          vx= 0.26, vy=-0.04,
                                          angle=0.3, disk_r=1.2,
                                          G_val=G_val, rng=rng)
    pos2, vel2, m2 = make_galaxy(n_stars - half,
                                          cx= 2.4, cy=-0.25,
                                          vx=-0.26, vy= 0.04,
                                          angle=np.pi*0.6, disk_r=1.1,
                                          G_val=G_val, rng=rng)
    pos  = np.vstack([pos1, pos2])
    vel  = np.vstack([vel1, vel2])
    mass = np.concatenate([m1, m2])
    gid  = np.array([0]*half + [1]*(n_stars-half), dtype=np.int8)
    return pos, vel, mass, gid


# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB FRAME RENDERER  → PNG bytes
# ══════════════════════════════════════════════════════════════════════════════

CMAPS = {
    "magma":    plt.get_cmap("magma"),
    "plasma":   plt.get_cmap("plasma"),
    "inferno":  plt.get_cmap("inferno"),
    "cyan":     mcolors.LinearSegmentedColormap.from_list(
                    "cyan_custom",
                    ["#001020","#003060","#00a0d0","#7DF9FF","#ffffff"]),
    "gold":     mcolors.LinearSegmentedColormap.from_list(
                    "gold_custom",
                    ["#0a0500","#3a1500","#c05000","#FFD700","#fffff0"]),
    "violet":   mcolors.LinearSegmentedColormap.from_list(
                    "violet_custom",
                    ["#05000a","#2a0040","#8000ff","#FF6BFF","#ffffff"]),
}

# ── Render constants — tuned for i7-1255U thermal budget ─────────────────────
RENDER_DPI = 75        # ↓ from 100: 75 DPI cuts pixel count by ~44%, biggest
                       #   single render speedup. Canvas is still crisp at
                       #   browser zoom levels due to CSS scaling.
RENDER_FIG_W = 8       # inches — combined with DPI gives 600×600 px canvas
RENDER_FIG_H = 8


def _make_figure(fig_w=RENDER_FIG_W, fig_h=RENDER_FIG_H, dpi=RENDER_DPI):
    """
    Build the Matplotlib figure ONCE.  Never called again after the first frame.

    Optimisations baked in:
    • dpi=75 → ~44% fewer pixels than dpi=100 → savefig & PNG encode ~2× faster
    • Agg canvas is pre-allocated at this size; buffer_rgba() reuses it every frame
    • Static starfield scatter drawn once; never touched again
    • Galaxy scatter artist 'sc' exposes set_offsets() / set_array() for zero-alloc updates
    """
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#02020a")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#02020a")
    ax.axis("off")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    # Static background starfield — drawn once, never redrawn
    rng2 = np.random.default_rng(7)
    bx = rng2.uniform(-6, 6, 400)
    by = rng2.uniform(-6, 6, 400)
    ax.scatter(bx, by, s=0.10, c="white", alpha=0.28, linewidths=0)

    # Galaxy particle scatter — s=0.6 compensates for lower DPI
    dummy = np.zeros((1, 2))
    sc = ax.scatter(
        dummy[:, 0], dummy[:, 1],
        s=0.60, c=[0.0], cmap="magma", vmin=0, vmax=1,
        linewidths=0, alpha=0.85,
    )

    # Force Agg backend to allocate its internal RGBA buffer now
    fig.canvas.draw()

    return fig, ax, sc


def render_frame(pos, vel, gid, cmap_name):
    """
    Render the current N-body state and return raw RGBA bytes.

    Performance path (fastest → slowest, tried in order):
    ┌─────────────────────────────────────────────────────────────────┐
    │  canvas.draw()  +  buffer_rgba()  →  PIL encode PNG in RAM     │
    │  ≈ 3–5× faster than fig.savefig(..., format="png")             │
    │  because savefig re-runs the full layout pass and calls the     │
    │  PNG C-extension with compression; buffer_rgba() just reads     │
    │  the Agg pixel buffer that is already current in memory.        │
    └─────────────────────────────────────────────────────────────────┘
    Falls back to savefig if PIL/Pillow is unavailable (it always is
    when streamlit is installed, but guard is defensive).

    All updates are in-place (set_offsets, set_array, set_cmap) —
    zero allocation of new Matplotlib artists per frame.
    """
    # ── Retrieve or build the cached figure ───────────────────────────────────
    cache_key = "_mpl_fig_main"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = _make_figure()
    fig, ax, sc = st.session_state[cache_key]

    # ── Colormap swap (cheap, no new artist) ──────────────────────────────────
    sc.set_cmap(CMAPS.get(cmap_name, CMAPS["magma"]))

    # ── Colour particles by speed, galaxy-offset for visual separation ─────────
    # Uses float32 for the norm — halves memory bandwidth vs float64
    speed = np.linalg.norm(vel, axis=1).astype(np.float32)
    smin  = float(speed.min())
    smax  = float(speed.max())
    c = (speed - smin) / (smax - smin + 1e-9)
    # Galaxy 1 → lower end of cmap, Galaxy 2 → slightly shifted
    c = np.clip(c + np.where(gid == 0, 0.0, 0.12), 0.0, 1.0)

    # ── Update scatter in-place (zero allocation) ─────────────────────────────
    sc.set_offsets(pos)
    sc.set_array(c)

    # ── Auto-zoom: keep all stars visible with 15% padding ────────────────────
    xc   = float(pos[:, 0].mean())
    yc   = float(pos[:, 1].mean())
    span = max(float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 1]))) * 0.5 + 1.0
    ax.set_xlim(xc - span, xc + span)
    ax.set_ylim(yc - span, yc + span)

    # ── Fast render path: canvas.draw() + buffer_rgba() + PIL PNG encode ──────
    fig.canvas.draw()          # redraws only dirty artists (Agg incremental)
    try:
        from PIL import Image
        # buffer_rgba() returns the Agg framebuffer as a memoryview — zero copy
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgba = rgba.reshape(h, w, 4)
        img  = Image.fromarray(rgba, "RGBA")
        buf  = io.BytesIO()
        # PNG with compress_level=1 (fastest): ~6× less CPU than default level 6
        img.save(buf, format="PNG", compress_level=1)
        buf.seek(0)
        return buf.read()
    except Exception:
        # Fallback: standard savefig (always works, slower)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=RENDER_DPI,
                    facecolor="#02020a", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def _init_state(n_stars, G_val):
    pos, vel, mass, gid = init_simulation(n_stars, G_val)
    st.session_state.pos   = pos
    st.session_state.vel   = vel
    st.session_state.mass  = mass
    st.session_state.gid   = gid
    st.session_state.step  = 0
    st.session_state.t_sim = 0.0
    st.session_state.fps_history = []

if "pos" not in st.session_state:
    _init_state(2000, 1.0)
if "running" not in st.session_state:
    st.session_state.running = False

# JIT warm-up (runs once, cached)
if "jit_ready" not in st.session_state:
    _tiny = st.session_state.pos[:100].copy()
    _m    = st.session_state.mass[:100].copy()
    _direct_forces(_tiny, _m, 1.0, 0.01)
    st.session_state.jit_ready = True


# ══════════════════════════════════════════════════════════════════════════════
#  ЕДИНЫЙ ФРАГМЕНТ — содержит весь UI кроме CSS и заголовка
#
#  Архитектура (окончательная):
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  Внешний скрипт:  только CSS + один заголовок + вызов фрагмента        │
#  │  Фрагмент:        sidebar + метрики + canvas + физика + рендер         │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  Почему это решает все проблемы:
#  • Слайдер/кнопка срабатывает → Streamlit перезапускает ТОЛЬКО фрагмент
#    (не внешний скрипт) → DOM не перестраивается → нет белого экрана
#  • Слоты (st.empty) создаются внутри фрагмента → при его перезапуске
#    они пересоздаются корректно на своих местах → нет мёртвых ссылок
#  • st.rerun() полностью убран → нет полного перезапуска страницы
#  • СБРОС: просто пишет в session_state и ждёт следующего тика фрагмента
# ══════════════════════════════════════════════════════════════════════════════

# Статичный заголовок — единственное что живёт вне фрагмента
st.markdown("<h1>СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>N-ТЕЛО · ЛЯГУШАЧИЙ ПРЫЖОК · NUMBA · FRAGMENT</div>",
    unsafe_allow_html=True,
)


@st.fragment(run_every=0.1)
def simulation_loop():
    ss = st.session_state

    # ── Сайдбар — внутри фрагмента, поэтому его взаимодействия не вызывают
    #    полный перезапуск скрипта, а перезапускают только этот фрагмент ──────
    with st.sidebar:
        st.markdown(
            "<h1 style='font-size:1.15rem!important'>🌌 УПРАВЛЕНИЕ</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='subtitle'>N-ТЕЛЬНОЕ СТОЛКНОВЕНИЕ v5.0</div>",
                    unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Параметры симуляции</div>",
                    unsafe_allow_html=True)
        st.slider("⭐ Количество звёзд на галактику",
                  200, 3000, step=100, key="p_n_stars",
                  help="Применяется при нажатии СБРОС")
        st.slider("G  Гравитационная постоянная",
                  0.1, 4.0, step=0.1, key="p_G_val")
        st.slider("Δt  Шаг времени",
                  0.0005, 0.008, step=0.0005, format="%.4f", key="p_dt")
        st.slider("ε  Сглаживание (Softening)",
                  0.01, 0.3, step=0.01, key="p_soft",
                  help="Предотвращает бесконечные силы при сближении")
        st.slider("⚡ Шагов физики на кадр",
                  1, 12, key="p_render_skip",
                  help="Рекомендуется 3–8")

        st.markdown("<div class='section-label'>Визуальные настройки</div>",
                    unsafe_allow_html=True)
        st.selectbox("🎨 Цветовая тема",
                     ["magma","plasma","inferno","cyan","gold","violet"],
                     key="p_cmap")

        st.markdown("<div class='section-label'>Управление</div>",
                    unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            lbl = "⏸ СТОП" if ss.running else "▶ СТАРТ"
            if st.button(lbl, key="btn_run"):
                ss.running = not ss.running
        with cb:
            if st.button("↺ СБРОС", key="btn_reset"):
                # Очищаем кэш matplotlib и данные — НЕТ st.rerun()
                # Фрагмент сам подхватит изменение на следующем тике
                for k in list(ss.keys()):
                    if k.startswith("_mpl_fig_"):
                        del ss[k]
                _init_state(ss.p_n_stars * 2, ss.p_G_val)
                ss.running = False

        st.markdown("<div class='section-label'>О симуляции</div>",
                    unsafe_allow_html=True)
        with st.expander("📖 Физика и методы"):
            st.markdown("""
**Закон всемирного тяготения**  `F = G·m₁·m₂ / (r² + ε²)`

**Интегратор Лягушачьего прыжка (DKD)**
1. Дрейф ½ шага → позиции
2. Толчок полный шаг → скорости
3. Дрейф ½ шага → позиции

**Вычисление сил**
- N ≤ 2500 → Numba `@njit(parallel=True)`
- N > 2500 → Тайловый NumPy (L2-кэш)

**Без белого экрана**  
Весь UI живёт внутри `@st.fragment` — любое  
взаимодействие перезапускает только фрагмент,  
страница не перестраивается.
            """)

    # ── Метрики ───────────────────────────────────────────────────────────────
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)

    # ── Холст симуляции ───────────────────────────────────────────────────────
    st.markdown("---")
    frame_slot = st.empty()

    # ── Футер ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;"
        "color:#334;text-align:center;letter-spacing:0.12em'>"
        "СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК · NUMBA · ЛЯГУШАЧИЙ ПРЫЖОК · "
        "ОПТИМИЗИРОВАНО ДЛЯ INTEL i7-1255U"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Физика ────────────────────────────────────────────────────────────────
    if ss.running:
        t0 = time.perf_counter()
        for _ in range(ss.p_render_skip):
            ss.pos, ss.vel = leapfrog_step(
                ss.pos, ss.vel, ss.mass,
                ss.p_dt, ss.p_G_val, ss.p_soft,
            )
            ss.step  += 1
            ss.t_sim += ss.p_dt
        elapsed = time.perf_counter() - t0

        fps_val = ss.p_render_skip / max(elapsed, 1e-6)
        hist = ss.fps_history
        hist.append(fps_val)
        if len(hist) > 30:
            hist.pop(0)
        if elapsed > 0.4:
            time.sleep(0.04)

    # ── Рендер ────────────────────────────────────────────────────────────────
    img_bytes = render_frame(ss.pos, ss.vel, ss.gid, ss.p_cmap)
    frame_slot.image(img_bytes, use_container_width=True)

    # ── Метрики (пишем после рендера чтобы FPS был актуальным) ───────────────
    fps_h = ss.fps_history
    fps_v = f"{sum(fps_h)/len(fps_h):.1f}" if fps_h else "—"
    with mc1: st.metric("ВСЕГО ЧАСТИЦ",   f"{len(ss.pos):,}")
    with mc2: st.metric("ШАГ СИМУЛЯЦИИ",  f"{ss.step:,}")
    with mc3: st.metric("СКОРОСТЬ (FPS)", fps_v)
    with mc4: st.metric("ВРЕМЯ СИМ.",     f"{ss.t_sim:.3f}")


# ── Единственный вызов во внешнем скрипте ─────────────────────────────────────
simulation_loop()
