"""
СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК
Зависимости: streamlit numpy numba matplotlib pillow
Запуск: streamlit run app.py
"""
import os, time, io, math
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numba import njit, prange

# ─────────────────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ СТРАНИЦЫ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Симулятор Галактик",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

:root {
    --bg:      #02020a;
    --surface: #080818;
    --glass:   rgba(12,12,35,0.92);
    --c1:      #7DF9FF;
    --c2:      #FF6BFF;
    --text:    #c8d8ff;
    --muted:   #4a5878;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--glass) !important;
    border-right: 1px solid rgba(125,249,255,0.10);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.55rem !important;
    font-weight: 900;
    letter-spacing: .08em;
    background: linear-gradient(90deg, var(--c1), var(--c2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 4px 0 !important;
}
.sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: .70rem;
    color: var(--muted);
    letter-spacing: .18em;
    margin-bottom: 16px;
}
.sec {
    font-family: 'Orbitron', monospace;
    font-size: .58rem;
    color: var(--muted);
    letter-spacing: .22em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(125,249,255,0.08);
    padding-bottom: 3px;
    margin: 14px 0 8px;
}
.badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: .70rem;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: .08em;
    margin-bottom: 8px;
}
.run  { border: 1px solid #7DF9FF; color: #7DF9FF; background: rgba(125,249,255,0.07); }
.idle { border: 1px solid #4a5878; color: #4a5878; }

[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid rgba(125,249,255,0.13);
    border-radius: 8px;
    padding: 10px 14px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--c1) !important;
    font-size: 1.15rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--muted) !important;
    font-size: .62rem !important;
    letter-spacing: .14em;
    text-transform: uppercase;
}

[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--c1), var(--c2)) !important;
}

.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .10em;
    border-radius: 4px;
    border: 1px solid var(--c1) !important;
    background: transparent !important;
    color: var(--c1) !important;
    transition: all .18s;
    width: 100%;
    padding: 8px 0;
}
.stButton > button:hover {
    background: var(--c1) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 16px var(--c1);
}

/* Картинка — чёрная рамка */
[data-testid="stImage"] img {
    border: 1px solid rgba(125,249,255,0.12);
    border-radius: 6px;
    display: block;
}

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid rgba(125,249,255,0.10) !important;
    border-radius: 6px;
}
summary { font-family: 'Orbitron', monospace !important; font-size: .75rem !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--c1); border-radius: 2px; }

/* Убираем лишние отступы вокруг изображения */
[data-testid="stImage"] { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ФИЗИКА
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _direct_forces(pos, mass, G, eps2):
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


def _tiled_forces(pos, mass, G, eps2):
    n    = len(pos)
    TILE = 256
    acc  = np.zeros((n, 2), dtype=np.float64)
    for i0 in range(0, n, TILE):
        i1  = min(i0 + TILE, n)
        ri  = pos[i0:i1, np.newaxis, :]
        dr  = pos[np.newaxis, :, :] - ri
        r2  = (dr**2).sum(-1) + eps2
        inv = G * mass[np.newaxis, :] / (r2 * np.sqrt(r2))
        inv[np.arange(i1-i0), np.arange(i0, i1)] = 0.0
        acc[i0:i1] = (inv[:, :, np.newaxis] * dr).sum(1)
    return acc


def compute_forces(pos, mass, G, soft):
    eps2 = soft * soft
    if len(pos) <= 1200:
        return _direct_forces(pos, mass, G, eps2)
    return _tiled_forces(pos, mass, G, eps2)


def leapfrog(pos, vel, mass, dt, G, soft):
    ph  = pos + 0.5 * dt * vel
    acc = compute_forces(ph, mass, G, soft)
    vn  = vel + dt * acc
    return ph + 0.5 * dt * vn, vn


def _make_galaxy(n, cx, cy, vx, vy, angle, disk_r, G, rng):
    r   = rng.exponential(disk_r / 3.0, n).clip(0.01)
    phi = rng.uniform(0, 2 * np.pi, n)
    ca, sa = np.cos(angle), np.sin(angle)
    x   = r * np.cos(phi) + rng.normal(0, 0.012, n)
    y   = r * np.sin(phi) + rng.normal(0, 0.012, n)
    xr  = ca*x - sa*y + cx
    yr  = sa*x + ca*y + cy
    vc  = np.sqrt(np.clip(G * np.tanh(r / (disk_r*0.3)) / (r + 0.05), 0, None))
    tx  = -np.sin(phi);  ty = np.cos(phi)
    vel = np.column_stack([(ca*tx - sa*ty)*vc + vx,
                           (sa*tx + ca*ty)*vc + vy])
    return (np.column_stack([xr, yr]).astype(np.float64),
            vel.astype(np.float64),
            np.full(n, 1.0/n, dtype=np.float64))


def new_simulation(n_total, G):
    rng  = np.random.default_rng(int(time.time()) % (2**31))
    half = n_total // 2
    p1,v1,m1 = _make_galaxy(half,         -2.5,  0.3,  0.25,-0.04, 0.3,      1.2, G, rng)
    p2,v2,m2 = _make_galaxy(n_total-half,  2.5, -0.3, -0.25, 0.04, np.pi*0.6,1.1, G, rng)
    return (np.vstack([p1, p2]),
            np.vstack([v1, v2]),
            np.concatenate([m1, m2]),
            np.array([0]*half + [1]*(n_total-half), dtype=np.int8))


# ─────────────────────────────────────────────────────────────────────────────
#  РЕНДЕР
# ─────────────────────────────────────────────────────────────────────────────

CMAPS = {
    "magma":   plt.get_cmap("magma"),
    "plasma":  plt.get_cmap("plasma"),
    "inferno": plt.get_cmap("inferno"),
    "cyan": mcolors.LinearSegmentedColormap.from_list(
        "cyan", ["#000d1a","#003060","#0090d0","#7DF9FF","#ffffff"]),
    "gold": mcolors.LinearSegmentedColormap.from_list(
        "gold", ["#0a0400","#401800","#c04000","#FFD700","#fffff0"]),
    "violet": mcolors.LinearSegmentedColormap.from_list(
        "violet",["#04000a","#280040","#7000e0","#FF6BFF","#ffffff"]),
}


def _build_fig():
    """Создаётся один раз и кешируется в session_state["_fig"]."""
    fig = plt.figure(figsize=(7, 7), dpi=80, facecolor="#02020a")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#02020a")
    ax.axis("off")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    # статичное звёздное небо
    rng2 = np.random.default_rng(42)
    ax.scatter(rng2.uniform(-6, 6, 350), rng2.uniform(-6, 6, 350),
               s=0.09, c="white", alpha=0.22, linewidths=0)
    # scatter для частиц (данные обновляются in-place каждый кадр)
    sc = ax.scatter([0], [0], s=0.7, c=[0.5], cmap="magma",
                    vmin=0, vmax=1, linewidths=0, alpha=0.88)
    fig.canvas.draw()
    return fig, ax, sc


def make_frame(pos, vel, gid, cmap_name):
    """Возвращает PNG-байты текущего кадра."""
    if "_fig" not in st.session_state:
        st.session_state["_fig"] = _build_fig()
    fig, ax, sc = st.session_state["_fig"]

    # цветовая карта
    sc.set_cmap(CMAPS.get(cmap_name, CMAPS["magma"]))

    # цвет по скорости
    spd = np.linalg.norm(vel, axis=1).astype(np.float32)
    c   = (spd - spd.min()) / (spd.max() - spd.min() + 1e-9)
    c   = np.clip(c + np.where(gid == 0, 0.0, 0.13), 0.0, 1.0)

    # обновляем scatter без пересоздания
    sc.set_offsets(pos)
    sc.set_array(c)

    # авто-зум
    xc   = float(pos[:, 0].mean());  yc = float(pos[:, 1].mean())
    span = max(float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 1]))) * 0.52 + 1.1
    ax.set_xlim(xc - span, xc + span)
    ax.set_ylim(yc - span, yc + span)

    # рендер в буфер (быстрый путь через PIL)
    fig.canvas.draw()
    try:
        from PIL import Image
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img  = Image.fromarray(rgba.reshape(h, w, 4), "RGBA")
        buf  = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        buf.seek(0)
        return buf.read()
    except Exception:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, facecolor="#02020a")
        buf.seek(0)
        return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

# значения по умолчанию для слайдеров
_DEF = dict(p_n=400, p_G=1.0, p_dt=0.002,
            p_soft=0.05, p_skip=3, p_cmap="magma")
for k, v in _DEF.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "pos" not in st.session_state:
    st.session_state.running = False
    pos, vel, mass, gid = new_simulation(
        st.session_state.p_n * 2, st.session_state.p_G)
    st.session_state.update(
        pos=pos, vel=vel, mass=mass, gid=gid,
        step=0, t=0.0, fps_hist=[])

# прогрев JIT один раз
if "jit_ok" not in st.session_state:
    _direct_forces(
        st.session_state.pos[:40].copy(),
        st.session_state.mass[:40].copy(), 1.0, 0.01)
    st.session_state.jit_ok = True


def do_reset():
    ss = st.session_state
    st.session_state.pop("_fig", None)
    pos, vel, mass, gid = new_simulation(ss.p_n * 2, ss.p_G)
    ss.update(pos=pos, vel=vel, mass=mass, gid=gid,
              step=0, t=0.0, fps_hist=[], running=False)


# ─────────────────────────────────────────────────────────────────────────────
#  САЙДБАР  ← строго вне @st.fragment (ограничение Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1 style='font-size:1.05rem!important;"
                "background:linear-gradient(90deg,#7DF9FF,#FF6BFF);"
                "-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
                "🌌 УПРАВЛЕНИЕ</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>СТОЛКНОВЕНИЕ ГАЛАКТИК</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='sec'>Параметры</div>", unsafe_allow_html=True)

    st.slider("⭐ Звёзд на галактику", 100, 1500, step=100, key="p_n",
              help="Итого = 2×. Применяется при СБРОС.")
    st.slider("G  Гравитация", 0.1, 4.0, step=0.1, key="p_G")
    st.slider("Δt  Шаг времени", 0.0005, 0.008, step=0.0005,
              format="%.4f", key="p_dt")
    st.slider("ε  Сглаживание", 0.01, 0.30, step=0.01, key="p_soft")
    st.slider("⚡ Шагов физики / кадр", 1, 8, key="p_skip",
              help="2–4 рекомендуется")

    st.markdown("<div class='sec'>Вид</div>", unsafe_allow_html=True)
    st.selectbox("🎨 Цветовая тема",
                 ["magma","plasma","inferno","cyan","gold","violet"],
                 key="p_cmap")

    st.markdown("<div class='sec'>Управление</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        lbl = "⏸ СТОП" if st.session_state.running else "▶ СТАРТ"
        if st.button(lbl, key="btn_run"):
            st.session_state.running = not st.session_state.running
    with c2:
        if st.button("↺ СБРОС", key="btn_rst"):
            do_reset()

    st.markdown("<div class='sec'>О симуляции</div>", unsafe_allow_html=True)
    with st.expander("📖 Физика и методы"):
        st.markdown("""
**Закон тяготения** `F = Gm₁m₂/(r²+ε²)`

**Лягушачий прыжок (DKD)**
1. Дрейф ½ шага
2. Толчок полный шаг
3. Дрейф ½ шага

**Силы**
N ≤ 1200 → Numba `@njit(parallel=True)`
N > 1200 → Тайловый NumPy

**Без мерцания**
`@st.fragment(run_every=0.1)` — физика
и рендер в изолированном фрагменте.
Сайдбар — вне фрагмента (требование
Streamlit). Нет `st.rerun()`.
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  ФРАГМЕНТ — содержит весь основной контент и анимацию
#
#  Правила Streamlit для @st.fragment:
#  ✓ можно: st.markdown, st.image, st.metric, st.columns, st.empty
#  ✗ нельзя: st.sidebar, любые контейнеры внешнего скрипта
#
#  Всё что фрагмент создаёт — живёт внутри его DOM-поддерева.
#  Каждый тик (0.1 с) фрагмент перерисовывает своё поддерево целиком.
#  Сайдбар и остальная страница при этом не трогаются.
# ─────────────────────────────────────────────────────────────────────────────

@st.fragment(run_every=0.1)
def _loop():
    ss = st.session_state

    # ── Заголовок ─────────────────────────────────────────────────────────────
    st.markdown("<h1>СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК</h1>",
                unsafe_allow_html=True)
    st.markdown("<div class='sub'>N-ТЕЛО · ЛЯГУШАЧИЙ ПРЫЖОК · "
                "NUMBA · FRAGMENT</div>", unsafe_allow_html=True)

    # ── Статус ────────────────────────────────────────────────────────────────
    is_run = ss.running
    cls    = "run" if is_run else "idle"
    txt    = "● РАБОТАЕТ" if is_run else "○  СТОП"
    st.markdown(f"<span class='badge {cls}'>{txt}</span>",
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Метрики ───────────────────────────────────────────────────────────────
    h     = ss.fps_hist
    fps_v = f"{sum(h)/len(h):.1f}" if h else "—"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ВСЕГО ЧАСТИЦ",   f"{len(ss.pos):,}")
    m2.metric("ШАГ СИМУЛЯЦИИ",  f"{ss.step:,}")
    m3.metric("СКОРОСТЬ (FPS)", fps_v)
    m4.metric("ВРЕМЯ СИМ.",     f"{ss.t:.3f}")

    st.markdown("---")

    # ── Физика ────────────────────────────────────────────────────────────────
    if is_run:
        t0 = time.perf_counter()
        for _ in range(ss.p_skip):
            ss.pos, ss.vel = leapfrog(
                ss.pos, ss.vel, ss.mass,
                ss.p_dt, ss.p_G, ss.p_soft)
            ss.step += 1
            ss.t    += ss.p_dt
        elapsed = time.perf_counter() - t0

        fps = ss.p_skip / max(elapsed, 1e-6)
        ss.fps_hist.append(fps)
        if len(ss.fps_hist) > 40:
            ss.fps_hist.pop(0)
        if elapsed > 0.6:
            time.sleep(0.06)

    # ── Рендер → картинка ─────────────────────────────────────────────────────
    img_bytes = make_frame(ss.pos, ss.vel, ss.gid, ss.p_cmap)
    st.image(img_bytes, use_container_width=True)

    # ── Футер ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;"
        "font-size:.58rem;color:#2a3050;text-align:center;"
        "letter-spacing:.10em'>"
        "СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК · NUMBA · "
        "STREAMLIT FRAGMENT"
        "</div>",
        unsafe_allow_html=True)


_loop()
