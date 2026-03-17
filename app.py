"""
СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК
Зависимости: streamlit numpy numba matplotlib
Запуск: streamlit run app.py
"""
import os, time, io, math
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit, prange

# ── Страница ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Симулятор Галактик",
    page_icon="🌌",
    layout="wide",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #02020a !important; color: #c8d8ff;
}
[data-testid="stSidebar"] {
    background: rgba(8,8,22,0.97) !important;
    border-right: 1px solid rgba(125,249,255,0.08);
}
[data-testid="stSidebar"] * { color: #c8d8ff !important; }
[data-testid="stMetricValue"] {
    color: #7DF9FF !important;
    font-family: monospace !important;
    font-size: 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #4a5878 !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="metric-container"] {
    background: #080818 !important;
    border: 1px solid rgba(125,249,255,0.12);
    border-radius: 8px;
}
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid rgba(125,249,255,0.10);
}
.stButton > button {
    border: 1px solid #7DF9FF !important;
    background: transparent !important;
    color: #7DF9FF !important;
    border-radius: 4px;
    width: 100%;
    font-weight: 700;
}
.stButton > button:hover {
    background: #7DF9FF !important;
    color: #02020a !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #7DF9FF, #FF6BFF) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Физика ────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _forces_nb(pos, mass, G, eps2):
    n  = pos.shape[0]
    ax = np.zeros(n, np.float64)
    ay = np.zeros(n, np.float64)
    for i in prange(n):
        fx = 0.0; fy = 0.0
        xi = pos[i, 0]; yi = pos[i, 1]
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            r2 = dx*dx + dy*dy + eps2
            f  = G * mass[j] / (r2 * math.sqrt(r2))
            fx += dx * f
            fy += dy * f
        ax[i] = fx
        ay[i] = fy
    return np.column_stack((ax, ay))


def _forces_np(pos, mass, G, eps2):
    n   = len(pos)
    acc = np.zeros((n, 2), np.float64)
    T   = 200
    for i0 in range(0, n, T):
        i1  = min(i0 + T, n)
        ri  = pos[i0:i1, np.newaxis, :]
        dr  = pos[np.newaxis, :, :] - ri
        r2  = (dr**2).sum(-1) + eps2
        inv = G * mass[np.newaxis, :] / (r2 * np.sqrt(r2))
        inv[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0
        acc[i0:i1] = (inv[:, :, np.newaxis] * dr).sum(1)
    return acc


def physics_step(pos, vel, mass, dt, G, soft):
    eps2 = soft * soft
    ph   = pos + 0.5 * dt * vel
    acc  = (_forces_nb(ph, mass, G, eps2)
            if len(ph) <= 800
            else _forces_np(ph, mass, G, eps2))
    vn   = vel + dt * acc
    return ph + 0.5 * dt * vn, vn


def make_sim(n_total, G):
    rng  = np.random.default_rng(int(time.time()) % 2**31)
    half = n_total // 2

    def galaxy(n, cx, cy, vx, vy, ang, R):
        r   = rng.exponential(R / 3, n).clip(0.01)
        phi = rng.uniform(0, 2 * np.pi, n)
        ca, sa = np.cos(ang), np.sin(ang)
        x   = r * np.cos(phi) + rng.normal(0, 0.01, n)
        y   = r * np.sin(phi) + rng.normal(0, 0.01, n)
        vc  = np.sqrt(np.clip(G * np.tanh(r / (R * 0.3)) / (r + 0.05), 0, None))
        tx  = -np.sin(phi); ty = np.cos(phi)
        vel = np.column_stack(
            [(ca*tx - sa*ty)*vc + vx, (sa*tx + ca*ty)*vc + vy])
        pos = np.column_stack([ca*x - sa*y + cx, sa*x + ca*y + cy])
        return pos.astype(np.float64), vel.astype(np.float64), np.full(n, 1.0/n)

    p1, v1, m1 = galaxy(half,         -2.5,  0.3,  0.25, -0.04, 0.30,      1.2)
    p2, v2, m2 = galaxy(n_total-half,  2.5, -0.3, -0.25,  0.04, np.pi*0.6, 1.1)
    pos  = np.vstack([p1, p2])
    vel  = np.vstack([v1, v2])
    mass = np.concatenate([m1, m2])
    gid  = np.array([0]*half + [1]*(n_total-half), dtype=np.int8)
    return pos, vel, mass, gid


# ── Рендер ────────────────────────────────────────────────────────────────────
# Используем fig.savefig() — самый надёжный метод на любом сервере.
# figure кешируется в session_state и обновляется in-place каждый кадр.

SAFE_CMAPS = ["magma", "plasma", "inferno", "viridis", "hot", "cool"]


def _get_fig():
    ss = st.session_state
    if "_fig" not in ss:
        fig = plt.figure(figsize=(7, 7), dpi=80, facecolor="#000008")
        ax  = fig.add_axes([0, 0, 1, 1], facecolor="#000008")
        ax.axis("off")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        # фоновые звёзды (рисуются один раз)
        rng2 = np.random.default_rng(42)
        ax.scatter(rng2.uniform(-8, 8, 500), rng2.uniform(-8, 8, 500),
                   s=0.1, c="white", alpha=0.18, linewidths=0)
        # scatter для частиц — будет обновляться каждый кадр
        sc = ax.scatter([0], [0], s=1.2, c=[0.5],
                        cmap="magma", vmin=0, vmax=1,
                        linewidths=0, alpha=0.9)
        ss["_fig"] = (fig, ax, sc)
    return ss["_fig"]


def render_png(pos, vel, gid, cmap_name):
    fig, ax, sc = _get_fig()

    # цветовая карта
    safe_cmap = cmap_name if cmap_name in SAFE_CMAPS else "magma"
    sc.set_cmap(safe_cmap)

    # цвет по скорости
    speed = np.linalg.norm(vel, axis=1)
    lo, hi = speed.min(), speed.max()
    c = (speed - lo) / (hi - lo + 1e-9)
    c = np.clip(c + np.where(gid == 0, 0.0, 0.13), 0.0, 1.0)

    # позиции и цвета (обновляем in-place, не пересоздаём scatter)
    sc.set_offsets(pos)
    sc.set_array(c)

    # авто-зум
    xc   = float(pos[:, 0].mean())
    yc   = float(pos[:, 1].mean())
    span = max(float(pos[:, 0].max() - pos[:, 0].min()),
               float(pos[:, 1].max() - pos[:, 1].min())) * 0.55 + 1.0
    ax.set_xlim(xc - span, xc + span)
    ax.set_ylim(yc - span, yc + span)

    # сохраняем в буфер через savefig — самый надёжный метод
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#000008", dpi=80)
    buf.seek(0)
    return buf.read()


# ── Session state ─────────────────────────────────────────────────────────────

if "initialized" not in st.session_state:
    ss = st.session_state
    ss.initialized = True
    ss.running     = False
    ss.p_n         = 300
    ss.p_G         = 1.0
    ss.p_dt        = 0.002
    ss.p_soft      = 0.05
    ss.p_skip      = 2
    ss.p_cmap      = "magma"
    ss.p_delay     = 0.35
    ss.step        = 0
    ss.t           = 0.0
    ss.fps_hist    = []
    p, v, m, g     = make_sim(ss.p_n * 2, ss.p_G)
    ss.pos = p; ss.vel = v; ss.mass = m; ss.gid = g

# прогрев Numba JIT (один раз при старте)
if "jit_ok" not in st.session_state:
    _forces_nb(st.session_state.pos[:30].copy(),
               st.session_state.mass[:30].copy(), 1.0, 0.01)
    st.session_state.jit_ok = True


def do_reset():
    ss = st.session_state
    ss.pop("_fig", None)          # удаляем кешированный figure
    p, v, m, g = make_sim(ss.p_n * 2, ss.p_G)
    ss.pos = p; ss.vel = v; ss.mass = m; ss.gid = g
    ss.step = 0; ss.t = 0.0; ss.fps_hist = []; ss.running = False


# ── Сайдбар ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌌 Управление")

    st.markdown("**Параметры симуляции**")
    st.slider("⭐ Звёзд на галактику", 100, 1000, step=100, key="p_n",
              help="Итого = 2×, применяется при Сбросе")
    st.slider("G  Гравитация", 0.1, 4.0, step=0.1, key="p_G")
    st.slider("Δt  Шаг времени", 0.0005, 0.008, step=0.0005,
              format="%.4f", key="p_dt")
    st.slider("ε  Сглаживание", 0.01, 0.30, step=0.01, key="p_soft")
    st.slider("⚡ Шагов физики / кадр", 1, 5, key="p_skip")

    st.markdown("**Скорость отображения**")
    st.slider("🕐 Задержка между кадрами (с)", 0.1, 3.0,
              step=0.05, format="%.2f", key="p_delay",
              help="Увеличьте если картинка не обновляется на Streamlit Cloud")

    st.markdown("**Внешний вид**")
    st.selectbox("🎨 Цветовая тема", SAFE_CMAPS, key="p_cmap")

    st.markdown("**Управление**")
    col1, col2 = st.columns(2)
    with col1:
        lbl = "⏸ Стоп" if st.session_state.running else "▶ Старт"
        if st.button(lbl, key="btn_toggle"):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button("↺ Сброс", key="btn_reset"):
            do_reset()

    with st.expander("📖 О симуляции"):
        st.markdown("""
**Закон тяготения** `F = Gm₁m₂/(r²+ε²)`

**Интегратор: лягушачий прыжок**
Drift → Kick → Drift

**Ядро расчёта сил**
- N ≤ 800: Numba `@njit(parallel=True)`
- N > 800: Тайловый NumPy

**Анимация**
`st.image()` + `time.sleep()` + `st.rerun()`
Задержка даёт серверу время отправить кадр.
        """)


# ── Основной экран ────────────────────────────────────────────────────────────

ss = st.session_state

st.markdown("## СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК")
st.caption("N-тело · Лягушачий прыжок · Numba")

# ── Физика ────────────────────────────────────────────────────────────────────
if ss.running:
    t0 = time.perf_counter()
    for _ in range(ss.p_skip):
        ss.pos, ss.vel = physics_step(
            ss.pos, ss.vel, ss.mass,
            ss.p_dt, ss.p_G, ss.p_soft)
        ss.step += 1
        ss.t    += ss.p_dt
    elapsed = time.perf_counter() - t0
    fps = ss.p_skip / max(elapsed, 1e-9)
    ss.fps_hist.append(fps)
    if len(ss.fps_hist) > 30:
        ss.fps_hist.pop(0)

# ── Рендер ────────────────────────────────────────────────────────────────────
img_bytes = render_png(ss.pos, ss.vel, ss.gid, ss.p_cmap)

# ── Статус + метрики ──────────────────────────────────────────────────────────
status = "🟢 РАБОТАЕТ" if ss.running else "⚪ СТОП"
h      = ss.fps_hist
fps_v  = f"{sum(h)/len(h):.1f}" if h else "—"

st.markdown(f"**{status}**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("ВСЕГО ЧАСТИЦ",   f"{len(ss.pos):,}")
m2.metric("ШАГ СИМУЛЯЦИИ",  f"{ss.step:,}")
m3.metric("СКОРОСТЬ (FPS)", fps_v)
m4.metric("ВРЕМЯ СИМ.",     f"{ss.t:.3f}")

st.markdown("---")

# ── Картинка ──────────────────────────────────────────────────────────────────
# st.image() напрямую — самый надёжный способ отобразить изображение.
# Никаких placeholder, container или fragment — только прямой вывод.
st.image(img_bytes, use_container_width=True)

st.markdown("---")
st.caption("Симулятор Столкновения Галактик · Numba · Streamlit")

# ── Цикл анимации ─────────────────────────────────────────────────────────────
# time.sleep() — даём серверу время отправить текущий кадр в браузер
# прежде чем начать следующий rerun.
if ss.running:
    time.sleep(ss.p_delay)
    st.rerun()
