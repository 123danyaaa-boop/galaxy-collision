"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК  —  N-Body Streamlit App                 ║
║   Запуск:  streamlit run app.py                                             ║
║   Зависимости: streamlit numpy numba matplotlib pillow                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
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

st.set_page_config(page_title="Симулятор Галактик", page_icon="🌌",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
:root{--void:#02020a;--surface:#080818;--glass:rgba(15,15,40,0.85);
      --accent1:#7DF9FF;--accent2:#FF6BFF;--text:#c8d8ff;--muted:#556080;}
html,body,[data-testid="stAppViewContainer"]{background:var(--void)!important;color:var(--text);}
[data-testid="stSidebar"]{background:var(--glass)!important;border-right:1px solid rgba(125,249,255,0.12);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3{font-family:'Orbitron',monospace!important;letter-spacing:.08em;}
h1{font-size:1.6rem!important;font-weight:900;
   background:linear-gradient(90deg,var(--accent1),var(--accent2));
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0!important;}
[data-testid="metric-container"]{background:var(--surface)!important;
   border:1px solid rgba(125,249,255,0.18);border-radius:8px;}
[data-testid="stMetricValue"]{font-family:'Share Tech Mono',monospace!important;
   color:var(--accent1)!important;font-size:1.1rem!important;}
[data-testid="stMetricLabel"]{font-family:'Share Tech Mono',monospace!important;
   color:var(--muted)!important;font-size:.65rem!important;
   letter-spacing:.12em;text-transform:uppercase;}
[data-testid="stSlider"]>div>div>div>div{
   background:linear-gradient(90deg,var(--accent1),var(--accent2))!important;}
.stButton>button{font-family:'Orbitron',monospace!important;letter-spacing:.1em;
   font-size:.75rem;font-weight:700;border-radius:4px;
   border:1px solid var(--accent1)!important;background:transparent!important;
   color:var(--accent1)!important;transition:all .2s;width:100%;}
.stButton>button:hover{background:var(--accent1)!important;color:var(--void)!important;
   box-shadow:0 0 18px var(--accent1);}
[data-testid="stImage"]{border:1px solid rgba(125,249,255,0.15);border-radius:6px;
   box-shadow:0 0 40px rgba(125,249,255,0.06);}
[data-testid="stExpander"]{background:var(--surface)!important;
   border:1px solid rgba(125,249,255,0.12)!important;border-radius:6px;}
summary{font-family:'Orbitron',monospace!important;font-size:.8rem!important;}
.subtitle{font-family:'Share Tech Mono',monospace;font-size:.72rem;
   color:var(--muted);letter-spacing:.15em;margin-bottom:14px;}
.sec{font-family:'Orbitron',monospace;font-size:.6rem;color:var(--muted);
   letter-spacing:.2em;text-transform:uppercase;
   border-bottom:1px solid rgba(125,249,255,0.1);padding-bottom:4px;margin:14px 0 8px;}
.badge{display:inline-block;font-family:'Share Tech Mono',monospace;
   font-size:.72rem;padding:3px 10px;border-radius:20px;letter-spacing:.08em;}
.run{border:1px solid #7DF9FF;color:#7DF9FF;background:rgba(125,249,255,0.08);}
.idle{border:1px solid #556080;color:#556080;}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:var(--accent1);border-radius:2px}
</style>
""", unsafe_allow_html=True)


# ── Физика ────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _direct_forces(pos, mass, G, eps2):
    n = pos.shape[0]
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        fx = 0.0; fy = 0.0
        xi = pos[i, 0]; yi = pos[i, 1]
        for j in range(n):
            if i == j: continue
            dx = pos[j,0]-xi; dy = pos[j,1]-yi
            r2 = dx*dx + dy*dy + eps2
            inv = G * mass[j] / (r2 * math.sqrt(r2))
            fx += dx*inv; fy += dy*inv
        ax[i] = fx; ay[i] = fy
    return np.column_stack((ax, ay))


def _tiled_forces(pos, mass, G, eps2):
    n = len(pos); TILE = 256
    acc = np.zeros((n, 2), dtype=np.float64)
    for i0 in range(0, n, TILE):
        i1  = min(i0+TILE, n)
        ri  = pos[i0:i1, np.newaxis, :]
        dr  = pos[np.newaxis,:,:] - ri
        r2  = (dr**2).sum(-1) + eps2
        inv = G * mass[np.newaxis,:] / (r2 * np.sqrt(r2))
        idx = np.arange(i0, i1)
        inv[np.arange(len(idx)), idx] = 0.0
        acc[i0:i1] = (inv[:,:,np.newaxis] * dr).sum(1)
    return acc


def compute_forces(pos, mass, G, soft):
    eps2 = soft*soft
    return (_direct_forces(pos, mass, G, eps2)
            if len(pos) <= 1500 else _tiled_forces(pos, mass, G, eps2))


def leapfrog_step(pos, vel, mass, dt, G, soft):
    ph  = pos + 0.5*dt*vel
    acc = compute_forces(ph, mass, G, soft)
    vn  = vel + dt*acc
    return ph + 0.5*dt*vn, vn


def make_galaxy(n, cx, cy, vx, vy, angle, disk_r, G_val, rng):
    r   = rng.exponential(disk_r/3.0, n).clip(0.01)
    phi = rng.uniform(0, 2*np.pi, n)
    ca, sa = np.cos(angle), np.sin(angle)
    x = r*np.cos(phi)+rng.normal(0,0.015,n)
    y = r*np.sin(phi)+rng.normal(0,0.015,n)
    xr = ca*x-sa*y+cx; yr = sa*x+ca*y+cy
    vc = np.sqrt(np.clip(G_val*np.tanh(r/(disk_r*0.3))/(r+0.05),0,None))
    tx=-np.sin(phi); ty=np.cos(phi)
    vel = np.column_stack([(ca*tx-sa*ty)*vc+vx,(sa*tx+ca*ty)*vc+vy])
    return (np.column_stack([xr,yr]).astype(np.float64),
            vel.astype(np.float64),
            np.full(n, 1.0/n, dtype=np.float64))


def init_simulation(n_total, G_val):
    rng  = np.random.default_rng(int(time.time()) % (2**31))
    half = n_total // 2
    p1,v1,m1 = make_galaxy(half,     -2.4, 0.25, 0.26,-0.04,0.3,      1.2,G_val,rng)
    p2,v2,m2 = make_galaxy(n_total-half,2.4,-0.25,-0.26,0.04,np.pi*0.6,1.1,G_val,rng)
    return (np.vstack([p1,p2]), np.vstack([v1,v2]),
            np.concatenate([m1,m2]),
            np.array([0]*half+[1]*(n_total-half), dtype=np.int8))


# ── Рендер ────────────────────────────────────────────────────────────────────

CMAPS = {
    "magma":   plt.get_cmap("magma"),
    "plasma":  plt.get_cmap("plasma"),
    "inferno": plt.get_cmap("inferno"),
    "cyan":    mcolors.LinearSegmentedColormap.from_list("c",
               ["#001020","#003060","#00a0d0","#7DF9FF","#fff"]),
    "gold":    mcolors.LinearSegmentedColormap.from_list("g",
               ["#0a0500","#3a1500","#c05000","#FFD700","#fffff0"]),
    "violet":  mcolors.LinearSegmentedColormap.from_list("v",
               ["#05000a","#2a0040","#8000ff","#FF6BFF","#fff"]),
}

def _make_figure():
    fig = plt.figure(figsize=(7,7), dpi=75, facecolor="#02020a")
    ax  = fig.add_axes([0,0,1,1], facecolor="#02020a")
    ax.axis("off"); ax.set_xlim(-6,6); ax.set_ylim(-6,6)
    rng2 = np.random.default_rng(7)
    ax.scatter(rng2.uniform(-6,6,300), rng2.uniform(-6,6,300),
               s=0.08, c="white", alpha=0.25, linewidths=0)
    sc = ax.scatter([0],[0], s=0.6, c=[0.0], cmap="magma",
                    vmin=0, vmax=1, linewidths=0, alpha=0.85)
    fig.canvas.draw()
    return fig, ax, sc


def render_frame(pos, vel, gid, cmap_name):
    if "_fig" not in st.session_state:
        st.session_state["_fig"] = _make_figure()
    fig, ax, sc = st.session_state["_fig"]

    sc.set_cmap(CMAPS.get(cmap_name, CMAPS["magma"]))
    speed = np.linalg.norm(vel, axis=1).astype(np.float32)
    c = (speed-speed.min())/(speed.max()-speed.min()+1e-9)
    c = np.clip(c + np.where(gid==0, 0.0, 0.12), 0.0, 1.0)
    sc.set_offsets(pos); sc.set_array(c)

    xc = float(pos[:,0].mean()); yc = float(pos[:,1].mean())
    span = max(float(np.ptp(pos[:,0])), float(np.ptp(pos[:,1])))*0.5+1.0
    ax.set_xlim(xc-span, xc+span); ax.set_ylim(yc-span, yc+span)
    fig.canvas.draw()

    try:
        from PIL import Image
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img  = Image.fromarray(rgba.reshape(h,w,4), "RGBA")
        buf  = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        buf.seek(0); return buf.read()
    except Exception:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=75, facecolor="#02020a")
        buf.seek(0); return buf.read()


# ── Session state bootstrap ───────────────────────────────────────────────────

DEFAULTS = dict(p_n_stars=500, p_G_val=1.0, p_dt=0.002,
                p_soft=0.05, p_render_skip=3, p_cmap="magma")
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "pos" not in st.session_state:
    st.session_state.running = False
    p,v,m,g = init_simulation(st.session_state.p_n_stars*2,
                               st.session_state.p_G_val)
    st.session_state.pos=p; st.session_state.vel=v
    st.session_state.mass=m; st.session_state.gid=g
    st.session_state.step=0; st.session_state.t_sim=0.0
    st.session_state.fps_history=[]

if "jit_ready" not in st.session_state:
    _direct_forces(st.session_state.pos[:50].copy(),
                   st.session_state.mass[:50].copy(), 1.0, 0.01)
    st.session_state.jit_ready = True


def _do_reset():
    ss = st.session_state
    if "_fig" in ss: del ss["_fig"]
    p,v,m,g = init_simulation(ss.p_n_stars*2, ss.p_G_val)
    ss.pos=p; ss.vel=v; ss.mass=m; ss.gid=g
    ss.step=0; ss.t_sim=0.0; ss.fps_history=[]; ss.running=False


# ══════════════════════════════════════════════════════════════════════════════
#  САЙДБАР — только во внешнем скрипте (запрещено внутри @st.fragment)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<h1 style='font-size:1.1rem!important'>🌌 УПРАВЛЕНИЕ</h1>",
                unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>СТОЛКНОВЕНИЕ ГАЛАКТИК v6.0</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='sec'>Параметры симуляции</div>",
                unsafe_allow_html=True)
    st.slider("⭐ Количество звёзд (×2)", 200, 2000, step=100,
              key="p_n_stars", help="Применяется при СБРОС")
    st.slider("G  Гравитационная постоянная", 0.1, 4.0, step=0.1, key="p_G_val")
    st.slider("Δt  Шаг времени", 0.0005, 0.008, step=0.0005,
              format="%.4f", key="p_dt")
    st.slider("ε  Сглаживание (Softening)", 0.01, 0.3, step=0.01, key="p_soft")
    st.slider("⚡ Шагов физики на кадр", 1, 8, key="p_render_skip",
              help="3–5 рекомендуется для Streamlit Cloud")

    st.markdown("<div class='sec'>Визуальные настройки</div>",
                unsafe_allow_html=True)
    st.selectbox("🎨 Цветовая тема",
                 ["magma","plasma","inferno","cyan","gold","violet"],
                 key="p_cmap")

    st.markdown("<div class='sec'>Управление</div>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        lbl = "⏸ СТОП" if st.session_state.running else "▶ СТАРТ"
        if st.button(lbl, key="btn_run"):
            st.session_state.running = not st.session_state.running
    with cb:
        if st.button("↺ СБРОС", key="btn_reset"):
            _do_reset()   # нет st.rerun() — фрагмент сам подхватит

    st.markdown("<div class='sec'>О симуляции</div>", unsafe_allow_html=True)
    with st.expander("📖 Физика и методы"):
        st.markdown("""
**Закон тяготения** `F = G·m₁·m₂ / (r² + ε²)`

**Интегратор Лягушачьего прыжка (DKD)**
1. Дрейф ½ шага → позиции
2. Толчок полный шаг → скорости
3. Дрейф ½ шага → позиции

**Вычисление сил**
- N ≤ 1500 → Numba `@njit(parallel=True)`
- N > 1500 → Тайловый NumPy

**Архитектура без мерцания**
Сайдбар — внешний скрипт.
Физика + рендер — `@st.fragment(run_every=0.1)`.
Нет `st.rerun()` → нет белого экрана.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  ОСНОВНОЙ КОНТЕНТ  — заголовок + слоты
#  Создаются во внешнем скрипте один раз. Фрагмент только ЗАПИСЫВАЕТ в них.
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<h1>СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>N-ТЕЛО · ЛЯГУШАЧИЙ ПРЫЖОК · NUMBA · FRAGMENT</div>",
            unsafe_allow_html=True)

status_ph = st.empty()
st.markdown("---")

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1: ph_parts = st.empty()
with mc2: ph_step  = st.empty()
with mc3: ph_fps   = st.empty()
with mc4: ph_time  = st.empty()

st.markdown("---")
canvas_ph = st.empty()
st.markdown("---")
st.markdown(
    "<div style='font-family:Share Tech Mono,monospace;font-size:.6rem;"
    "color:#334;text-align:center;letter-spacing:.1em'>"
    "СИМУЛЯТОР СТОЛКНОВЕНИЯ ГАЛАКТИК · NUMBA · ЛЯГУШАЧИЙ ПРЫЖОК · "
    "STREAMLIT FRAGMENT PATTERN"
    "</div>",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ФРАГМЕНТ — только физика + рендер
#  НЕ содержит: st.sidebar, st.columns (кроме внутри самого фрагмента),
#               любые layout-контейнеры внешнего скрипта.
#  СОДЕРЖИТ: обращения к st.empty() слотам через замыкание.
# ══════════════════════════════════════════════════════════════════════════════

@st.fragment(run_every=0.1)
def _physics_loop():
    ss = st.session_state

    # Физика
    if ss.running:
        t0 = time.perf_counter()
        for _ in range(ss.p_render_skip):
            ss.pos, ss.vel = leapfrog_step(
                ss.pos, ss.vel, ss.mass,
                ss.p_dt, ss.p_G_val, ss.p_soft)
            ss.step  += 1
            ss.t_sim += ss.p_dt
        elapsed = time.perf_counter() - t0
        fps = ss.p_render_skip / max(elapsed, 1e-6)
        ss.fps_history.append(fps)
        if len(ss.fps_history) > 30: ss.fps_history.pop(0)
        if elapsed > 0.5: time.sleep(0.05)

    # Рендер
    img = render_frame(ss.pos, ss.vel, ss.gid, ss.p_cmap)

    # Запись в слоты (доступны через замыкание)
    is_run = ss.running
    status_ph.markdown(
        f"<span class='badge {'run' if is_run else 'idle'}'>"
        f"{'● РАБОТАЕТ' if is_run else '○  СТОП'}</span>",
        unsafe_allow_html=True)

    canvas_ph.image(img, use_container_width=True)

    h = ss.fps_history
    fps_v = f"{sum(h)/len(h):.1f}" if h else "—"
    ph_parts.metric("ВСЕГО ЧАСТИЦ",   f"{len(ss.pos):,}")
    ph_step .metric("ШАГ СИМУЛЯЦИИ",  f"{ss.step:,}")
    ph_fps  .metric("СКОРОСТЬ (FPS)", fps_v)
    ph_time .metric("ВРЕМЯ СИМ.",     f"{ss.t_sim:.3f}")


_physics_loop()
