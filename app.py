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

# ─── Конфигурация ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Симулятор Галактик",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
:root { --bg:#02020a; --sur:#080818; --c1:#7DF9FF; --c2:#FF6BFF;
        --txt:#c8d8ff; --dim:#4a5878; }
html,body,[data-testid="stAppViewContainer"]
    { background:var(--bg)!important; color:var(--txt); }
[data-testid="stSidebar"]
    { background:rgba(10,10,28,0.95)!important;
      border-right:1px solid rgba(125,249,255,0.10); }
[data-testid="stSidebar"] * { color:var(--txt)!important; }
h1  { font-family:'Orbitron',monospace!important; font-size:1.5rem!important;
      font-weight:900; letter-spacing:.07em;
      background:linear-gradient(90deg,var(--c1),var(--c2));
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      margin-bottom:2px!important; }
.sub{ font-family:'Share Tech Mono',monospace; font-size:.68rem;
      color:var(--dim); letter-spacing:.16em; margin-bottom:18px; }
.sec{ font-family:'Orbitron',monospace; font-size:.58rem; color:var(--dim);
      letter-spacing:.20em; text-transform:uppercase;
      border-bottom:1px solid rgba(125,249,255,0.08);
      padding-bottom:3px; margin:13px 0 7px; }
.badge{ display:inline-block; font-family:'Share Tech Mono',monospace;
        font-size:.70rem; padding:4px 14px; border-radius:20px;
        letter-spacing:.08em; margin-bottom:10px; }
.run { border:1px solid #7DF9FF; color:#7DF9FF;
       background:rgba(125,249,255,0.07); }
.idle{ border:1px solid #4a5878; color:#4a5878; }
[data-testid="metric-container"]
    { background:var(--sur)!important;
      border:1px solid rgba(125,249,255,0.14); border-radius:8px; }
[data-testid="stMetricValue"]
    { font-family:'Share Tech Mono',monospace!important;
      color:var(--c1)!important; font-size:1.1rem!important; }
[data-testid="stMetricLabel"]
    { font-family:'Share Tech Mono',monospace!important;
      color:var(--dim)!important; font-size:.60rem!important;
      letter-spacing:.13em; text-transform:uppercase; }
[data-testid="stSlider"]>div>div>div>div
    { background:linear-gradient(90deg,var(--c1),var(--c2))!important; }
.stButton>button
    { font-family:'Orbitron',monospace!important; font-size:.72rem;
      font-weight:700; letter-spacing:.10em; border-radius:4px;
      border:1px solid var(--c1)!important; background:transparent!important;
      color:var(--c1)!important; transition:all .18s; width:100%; }
.stButton>button:hover
    { background:var(--c1)!important; color:var(--bg)!important;
      box-shadow:0 0 16px var(--c1); }
[data-testid="stImage"] img
    { border:1px solid rgba(125,249,255,0.12); border-radius:6px; }
[data-testid="stExpander"]
    { background:var(--sur)!important;
      border:1px solid rgba(125,249,255,0.10)!important; border-radius:6px; }
summary{ font-family:'Orbitron',monospace!important; font-size:.78rem!important; }
::-webkit-scrollbar{ width:4px; }
::-webkit-scrollbar-thumb{ background:var(--c1); border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ─── Физика ──────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _forces_direct(pos, mass, G, eps2):
    n  = pos.shape[0]
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        fx = 0.0; fy = 0.0
        xi = pos[i, 0]; yi = pos[i, 1]
        for j in range(n):
            if i == j: continue
            dx = pos[j,0]-xi;  dy = pos[j,1]-yi
            r2 = dx*dx + dy*dy + eps2
            f  = G * mass[j] / (r2 * math.sqrt(r2))
            fx += dx*f;  fy += dy*f
        ax[i] = fx;  ay[i] = fy
    return np.column_stack((ax, ay))


def _forces_tiled(pos, mass, G, eps2):
    n = len(pos);  T = 256
    acc = np.zeros((n, 2), dtype=np.float64)
    for i0 in range(0, n, T):
        i1  = min(i0+T, n)
        ri  = pos[i0:i1, np.newaxis, :]
        dr  = pos[np.newaxis, :, :] - ri
        r2  = (dr**2).sum(-1) + eps2
        inv = G * mass[np.newaxis, :] / (r2 * np.sqrt(r2))
        inv[np.arange(i1-i0), np.arange(i0, i1)] = 0.0
        acc[i0:i1] = (inv[:, :, np.newaxis] * dr).sum(1)
    return acc


def step_leapfrog(pos, vel, mass, dt, G, soft):
    e2  = soft * soft
    ph  = pos + 0.5*dt*vel
    acc = (_forces_direct(ph, mass, G, e2)
           if len(ph) <= 1000 else _forces_tiled(ph, mass, G, e2))
    vn  = vel + dt*acc
    return ph + 0.5*dt*vn, vn


def new_galaxy(n, cx, cy, vx, vy, ang, R, G, rng):
    r   = rng.exponential(R/3, n).clip(0.01)
    phi = rng.uniform(0, 2*np.pi, n)
    ca, sa = np.cos(ang), np.sin(ang)
    x = r*np.cos(phi)+rng.normal(0,.012,n)
    y = r*np.sin(phi)+rng.normal(0,.012,n)
    vc = np.sqrt(np.clip(G*np.tanh(r/(R*.3))/(r+.05), 0, None))
    tx = -np.sin(phi);  ty = np.cos(phi)
    vel = np.column_stack([(ca*tx-sa*ty)*vc+vx, (sa*tx+ca*ty)*vc+vy])
    return (np.column_stack([ca*x-sa*y+cx, sa*x+ca*y+cy]).astype(np.float64),
            vel.astype(np.float64),
            np.full(n, 1.0/n, np.float64))


def new_sim(n_total, G):
    rng  = np.random.default_rng(int(time.time()) % 2**31)
    half = n_total // 2
    p1,v1,m1 = new_galaxy(half,       -2.5, .3,  .25,-.04, .30,      1.2, G, rng)
    p2,v2,m2 = new_galaxy(n_total-half, 2.5,-.3, -.25, .04, np.pi*.6, 1.1, G, rng)
    return (np.vstack([p1,p2]), np.vstack([v1,v2]),
            np.concatenate([m1,m2]),
            np.array([0]*half+[1]*(n_total-half), dtype=np.int8))


# ─── Рендер ──────────────────────────────────────────────────────────────────

CMAPS = {
    "magma":   plt.get_cmap("magma"),
    "plasma":  plt.get_cmap("plasma"),
    "inferno": plt.get_cmap("inferno"),
    "cyan":   mcolors.LinearSegmentedColormap.from_list("c",
              ["#000d1a","#003060","#0090d0","#7DF9FF","#fff"]),
    "gold":   mcolors.LinearSegmentedColormap.from_list("g",
              ["#0a0400","#401800","#c04000","#FFD700","#fffff0"]),
    "violet": mcolors.LinearSegmentedColormap.from_list("v",
              ["#04000a","#280040","#7000e0","#FF6BFF","#fff"]),
}


def _build_fig():
    fig = plt.figure(figsize=(6, 6), dpi=80, facecolor="#02020a")
    ax  = fig.add_axes([0,0,1,1], facecolor="#02020a")
    ax.axis("off");  ax.set_xlim(-6,6);  ax.set_ylim(-6,6)
    rng2 = np.random.default_rng(7)
    ax.scatter(rng2.uniform(-6,6,300), rng2.uniform(-6,6,300),
               s=.09, c="white", alpha=.22, linewidths=0)
    sc = ax.scatter([0],[0], s=.8, c=[.5], cmap="magma",
                   vmin=0, vmax=1, linewidths=0, alpha=.88)
    fig.canvas.draw()
    return fig, ax, sc


def render(pos, vel, gid, cmap_name):
    ss = st.session_state
    if "_fig" not in ss:
        ss["_fig"] = _build_fig()
    fig, ax, sc = ss["_fig"]

    sc.set_cmap(CMAPS.get(cmap_name, CMAPS["magma"]))
    spd = np.linalg.norm(vel, axis=1).astype(np.float32)
    c   = (spd-spd.min()) / (spd.max()-spd.min()+1e-9)
    c   = np.clip(c + np.where(gid==0, 0., .13), 0., 1.)
    sc.set_offsets(pos);  sc.set_array(c)

    xc   = float(pos[:,0].mean());  yc = float(pos[:,1].mean())
    span = max(float(np.ptp(pos[:,0])), float(np.ptp(pos[:,1])))*0.52 + 1.1
    ax.set_xlim(xc-span, xc+span);  ax.set_ylim(yc-span, yc+span)
    fig.canvas.draw()

    # PNG через PIL (быстрый путь) или savefig (резервный)
    try:
        from PIL import Image
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img  = Image.fromarray(rgba.reshape(h, w, 4), "RGBA")
        buf  = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        return buf.getvalue()
    except Exception:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, facecolor="#02020a")
        return buf.getvalue()


# ─── Session state ────────────────────────────────────────────────────────────

DEFAULTS = dict(p_n=300, p_G=1.0, p_dt=0.002, p_soft=0.05,
                p_skip=2, p_cmap="magma", p_delay=0.25)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "pos" not in st.session_state:
    st.session_state.running = False
    p, v, m, g = new_sim(st.session_state.p_n*2, st.session_state.p_G)
    st.session_state.update(pos=p, vel=v, mass=m, gid=g,
                             step=0, t=0.0, fps_hist=[])

# Прогрев JIT
if "jit_ok" not in st.session_state:
    _forces_direct(st.session_state.pos[:40].copy(),
                   st.session_state.mass[:40].copy(), 1.0, 0.01)
    st.session_state.jit_ok = True


def do_reset():
    ss = st.session_state
    ss.pop("_fig", None)
    p, v, m, g = new_sim(ss.p_n*2, ss.p_G)
    ss.update(pos=p, vel=v, mass=m, gid=g,
              step=0, t=0.0, fps_hist=[], running=False)


# ─── Сайдбар ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h1 style='font-size:1.0rem!important;"
        "background:linear-gradient(90deg,#7DF9FF,#FF6BFF);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
        "🌌 УПРАВЛЕНИЕ</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>СТОЛКНОВЕНИЕ ГАЛАКТИК</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='sec'>Параметры</div>", unsafe_allow_html=True)
    st.slider("⭐ Звёзд на галактику", 100, 1200, step=100, key="p_n",
              help="Итого = 2×. Применяется при СБРОС.")
    st.slider("G  Гравитация", 0.1, 4.0, step=0.1, key="p_G")
    st.slider("Δt  Шаг времени", 0.0005, 0.008, step=0.0005,
              format="%.4f", key="p_dt")
    st.slider("ε  Сглаживание", 0.01, 0.30, step=0.01, key="p_soft")
    st.slider("⚡ Шагов физики / кадр", 1, 6, key="p_skip")

    st.markdown("<div class='sec'>Скорость</div>", unsafe_allow_html=True)
    st.slider("🕐 Задержка между кадрами (с)", 0.05, 2.0,
              step=0.05, format="%.2f", key="p_delay",
              help="Больше = медленнее, но стабильнее на слабых серверах")

    st.markdown("<div class='sec'>Вид</div>", unsafe_allow_html=True)
    st.selectbox("🎨 Цветовая тема",
                 ["magma","plasma","inferno","cyan","gold","violet"],
                 key="p_cmap")

    st.markdown("<div class='sec'>Управление</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        lbl = "⏸ СТОП" if st.session_state.running else "▶ СТАРТ"
        if st.button(lbl, key="btn_run"):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button("↺ СБРОС", key="btn_rst"):
            do_reset()

    st.markdown("<div class='sec'>О симуляции</div>", unsafe_allow_html=True)
    with st.expander("📖 Физика и методы"):
        st.markdown("""
**Закон тяготения** `F = Gm₁m₂/(r²+ε²)`

**Лягушачий прыжок (DKD)**
1. Дрейф ½ шага → позиции
2. Толчок полный шаг → скорости
3. Дрейф ½ шага → позиции

**Вычисление сил**
N ≤ 1000 → Numba `parallel=True`
N > 1000 → Тайловый NumPy

**Как работает анимация**
Один кадр физики → рендер PNG →
`st.image()` → `time.sleep(задержка)`
→ `st.rerun()`. Простой и надёжный
цикл без фрагментов.

**Задержка между кадрами**
Увеличьте если сервер не справляется.
0.25–0.5 с — комфортно для Cloud.
        """)


# ─── Основной контент ─────────────────────────────────────────────────────────
#
#  Ключевое решение: st.empty() + with placeholder.container()
#  Весь контент (заголовок, метрики, картинка) рисуется ВНУТРИ контейнера.
#  st.rerun() пересоздаёт контейнер на том же месте — DOM стабилен,
#  страница не прокручивается, картинка отображается гарантированно.
#
# ─────────────────────────────────────────────────────────────────────────────

placeholder = st.empty()

ss = st.session_state

# ── Физика (если запущено) ────────────────────────────────────────────────────
if ss.running:
    t0 = time.perf_counter()

    for _ in range(ss.p_skip):
        ss.pos, ss.vel = step_leapfrog(
            ss.pos, ss.vel, ss.mass,
            ss.p_dt, ss.p_G, ss.p_soft)
        ss.step += 1
        ss.t    += ss.p_dt

    elapsed = time.perf_counter() - t0
    fps = ss.p_skip / max(elapsed, 1e-9)
    ss.fps_hist.append(fps)
    if len(ss.fps_hist) > 30: ss.fps_hist.pop(0)

# ── Рендер ────────────────────────────────────────────────────────────────────
img_bytes = render(ss.pos, ss.vel, ss.gid, ss.p_cmap)

# ── Отображение в placeholder (всё в одном контейнере) ────────────────────────
with placeholder.container():

    st.markdown("<h1>СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК</h1>",
                unsafe_allow_html=True)
    st.markdown("<div class='sub'>N-ТЕЛО · ЛЯГУШАЧИЙ ПРЫЖОК · "
                "NUMBA · STREAMLIT</div>", unsafe_allow_html=True)

    is_run = ss.running
    st.markdown(
        f"<span class='badge {'run' if is_run else 'idle'}'>"
        f"{'● РАБОТАЕТ' if is_run else '○  СТОП'}</span>",
        unsafe_allow_html=True)

    st.markdown("---")

    # Метрики
    h     = ss.fps_hist
    fps_v = f"{sum(h)/len(h):.1f}" if h else "—"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ВСЕГО ЧАСТИЦ",   f"{len(ss.pos):,}")
    m2.metric("ШАГ СИМУЛЯЦИИ",  f"{ss.step:,}")
    m3.metric("СКОРОСТЬ (FPS)", fps_v)
    m4.metric("ВРЕМЯ СИМ.",     f"{ss.t:.3f}")

    st.markdown("---")

    # Картинка — главный результат
    st.image(img_bytes, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;"
        "font-size:.58rem;color:#1e2540;text-align:center;"
        "letter-spacing:.10em'>"
        "СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК · NUMBA · STREAMLIT"
        "</div>", unsafe_allow_html=True)

# ── Цикл анимации ─────────────────────────────────────────────────────────────
# time.sleep() — искусственное замедление по запросу пользователя.
# Даёт серверу время отправить кадр в браузер до следующего rerun.
if ss.running:
    time.sleep(ss.p_delay)
    st.rerun()
