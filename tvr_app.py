# TVR — Territorial Viability Rating
# Louis Viallard — Université Lumière Lyon 2 — 2026

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

st.set_page_config(page_title="TVR — Territorial Viability Rating", layout="wide")

def clamp(v): return max(0, min(100, int(v)))
def noise(lat, lon, s=1): return math.sin(lat*0.3+s)*math.cos(lon*0.2+s*1.7)*5

def is_land(lat, lon):
    if -60<lat<72 and -70<lon<-15:
        if -10<lat<15 and -20<lon<-5: return True
        return False
    if lon<-100 and lat<60:
        if 7<lat<20 and -92<lon<-77: return True
        if -55<lat<-17 and -82<lon<-68: return True
        return False
    if lon>155 and lat<15: return False
    if lon>165 and lat<50: return False
    if -60<lat<0 and 55<lon<90: return False
    if 8<lat<22 and 58<lon<65: return False
    return True

def tvr(lat, lon, w=1.8, s=1.2):
    if not is_land(lat, lon): return None
    a = abs(lat)
    if a>70: d1=75-w*2
    elif a>60: d1=82-w*3
    elif a>50: d1=78-w*4
    elif a>40: d1=72-w*6
    elif a>30: d1=58-w*9
    elif a>20: d1=42-w*12
    elif a>10: d1=38-w*13
    else: d1=35-w*14
    if 20<lat<30 and 45<lon<60: d1-=25
    if 15<lat<35 and -15<lon<35: d1-=18
    if 15<lat<28 and 68<lon<88: d1-=14
    if 18<lat<26 and 85<lon<106: d1-=16
    if -35<lat<-20 and 125<lon<145: d1-=15
    if 28<lat<40 and 70<lon<100: d1+=22
    if -20<lat<5 and -80<lon<-65: d1+=18
    if 44<lat<48 and 5<lon<15: d1+=8
    if 60<lat<70 and 20<lon<32: d1+=6
    if 35<lat<44 and -5<lon<36: d1-=8+w*4
    if 22<lat<28 and 98<lon<106: d1+=20
    if 10<lat<18 and -18<lon<40: d1-=15
    d1=clamp(d1+noise(lat,lon,1))
    if a>65: d2=80
    elif a>55: d2=82
    elif a>45: d2=74
    elif a>35: d2=58
    elif a>25: d2=42
    elif a>15: d2=50
    else: d2=62
    if 15<lat<35 and -15<lon<35: d2-=45
    if 20<lat<30 and 45<lon<60: d2-=48
    if -35<lat<-15 and 125<lon<145: d2-=30
    if 10<lat<18 and -18<lon<40: d2-=30
    if -15<lat<5 and -75<lon<-50: d2+=20
    if 58<lat<70 and 14<lon<32: d2+=15
    if 22<lat<28 and 98<lon<106: d2+=15
    if 15<lat<28 and 68<lon<88: d2-=18+w*5
    if 35<lat<44 and -5<lon<36: d2-=10+w*6
    d2=clamp(d2-w*3+noise(lat,lon,2))
    d3=88
    if 21<lat<24 and 88<lon<92: d3=max(5,8-s*8)
    if 51<lat<53 and 3<lon<7: d3=max(5,28-s*10)
    if 9<lat<12 and 104<lon<107: d3=max(5,18-s*10)
    if 24<lat<28 and -82<lon<-78: d3=max(5,25-s*12)
    if 24<lat<42 and -82<lon<-70: d3=max(5,45-s*7)
    if 28<lat<40 and 70<lon<100: d3=96
    if 44<lat<48 and 5<lon<15: d3=95
    if 60<lat<70 and 14<lon<30: d3=94
    if 22<lat<28 and 98<lon<106: d3=93
    d3=clamp(max(0,d3)+noise(lat,lon,3))
    if a>65: d4=55
    elif a>55: d4=72
    elif a>45: d4=70
    elif a>35: d4=60
    elif a>25: d4=48
    elif a>15: d4=55
    else: d4=58
    if 15<lat<35 and -15<lon<35: d4-=40
    if 20<lat<30 and 45<lon<60: d4-=38
    if 10<lat<18 and -18<lon<40: d4-=28
    if 35<lat<42 and 108<lon<125: d4-=20
    if 35<lat<50 and -100<lon<-80: d4+=15
    if 22<lat<28 and 98<lon<106: d4+=12
    if 58<lat<70 and 14<lon<32: d4+=10
    d4=clamp(d4-w*2+noise(lat,lon,4))
    if a>70: d5=65
    elif a>60: d5=80
    elif a>50: d5=72
    elif a>40: d5=62
    elif a>30: d5=50
    elif a>20: d5=55
    else: d5=70
    if -15<lat<5 and -75<lon<-50: d5+=28
    if -5<lat<5 and 15<lon<35: d5+=22
    if 22<lat<28 and 98<lon<106: d5+=24
    if 15<lat<35 and -15<lon<35: d5-=35
    if 20<lat<30 and 45<lon<60: d5-=30
    if 10<lat<18 and -18<lon<40: d5-=25
    if 35<lat<42 and 108<lon<125: d5-=18
    d5=clamp(d5-w*4+noise(lat,lon,5))
    return [d1,d2,d3,d4,d5]

PARAMS = {
    ("SSP1","2040"):(0.5,0.3),("SSP1","2070"):(1.0,0.6),("SSP1","2100"):(1.5,1.0),
    ("SSP2","2040"):(0.8,0.5),("SSP2","2070"):(1.8,1.2),("SSP2","2100"):(2.7,2.0),
    ("SSP5","2040"):(1.2,0.8),("SSP5","2070"):(3.0,2.0),("SSP5","2100"):(4.4,3.5),
}

CMAP = LinearSegmentedColormap.from_list("TVR",[
    (0.00,"#9B2226"),(0.25,"#E76F51"),(0.40,"#F4A261"),
    (0.55,"#F4D03F"),(0.65,"#95D5B2"),(0.75,"#52B788"),
    (0.85,"#2D6A4F"),(1.00,"#1B4332")],N=256)

STEP = 2
LATS = np.arange(-88, 89, STEP, dtype=float)
LONS = np.arange(-178, 179, STEP, dtype=float)

@st.cache_data(show_spinner="Calcul de la grille mondiale (~2 min)...")
def build_cache():
    cache = {}
    for (sc,hz),(w,s) in PARAMS.items():
        # grille (nlat, nlon, 5)
        grid = np.full((len(LATS), len(LONS), 5), np.nan)
        for i,lat in enumerate(LATS):
            for j,lon in enumerate(LONS):
                r = tvr(float(lat), float(lon), w, s)
                if r:
                    grid[i,j] = r
        cache[f"{sc}_{hz}"] = grid
    return cache

# ── INTERFACE ────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0D1B2A;padding:12px 18px;border-radius:8px;
border-left:4px solid #2D6A4F;margin-bottom:14px'>
<h2 style='color:white;margin:0;font-size:18px'>
TERRITORIAL VIABILITY RATING (TVR)</h2>
<p style='color:#95D5B2;margin:3px 0 0;font-size:11px'>
Indice de performance écologique territoriale ·
Louis Viallard · Université Lumière Lyon 2 · 2026</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    sc_full = st.selectbox("Scénario",[
        "SSP1 — Optimiste (+2°C)",
        "SSP2 — Intermédiaire (+2.7°C)",
        "SSP5 — Pessimiste (+4.4°C)"],index=1)
    sc = sc_full[:4]
with c2:
    hz = st.selectbox("Horizon",["2040","2070","2100"],index=1)
with c3:
    dim_name = st.selectbox("Dimension",[
        "Score moyen","D1 — Thermique","D2 — Eau",
        "D3 — Submersion","D4 — Air & Sols","D5 — Écosystème"])
    dim_idx = ["Score moyen","D1 — Thermique","D2 — Eau",
               "D3 — Submersion","D4 — Air & Sols","D5 — Écosystème"].index(dim_name)

cache = build_cache()
grid = cache[f"{sc}_{hz}"]  # shape (nlat, nlon, 5)

# Score à afficher
if dim_idx == 0:
    data = np.nanmean(grid, axis=2)
else:
    data = grid[:,:,dim_idx-1]

# Masque terres
land = np.array([[is_land(float(lat),float(lon))
                  for lon in LONS] for lat in LATS])
data_plot = np.where(land, data, np.nan)

# ── CARTE ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16,8), facecolor="#0D1B2A")
ax.set_facecolor("#0D1B2A")

# pcolormesh avec coordonnées des bords de cellules
lon_edges = np.append(LONS - STEP/2, LONS[-1] + STEP/2)
lat_edges = np.append(LATS - STEP/2, LATS[-1] + STEP/2)
LON_E, LAT_E = np.meshgrid(lon_edges, lat_edges)

ax.pcolormesh(LON_E, LAT_E, data_plot,
    cmap=CMAP, vmin=0, vmax=100,
    shading='flat', zorder=2, alpha=0.93)

ax.set_xlim(-180,180); ax.set_ylim(-90,90)
ax.set_aspect('equal'); ax.axis('off')

fig.text(0.5,0.97, f"TVR — {dim_name} · {sc} · {hz}",
    ha='center', color='white', fontsize=13, fontweight='bold')
fig.text(0.5,0.02,
    "Sources : IPCC AR6 · WRI Aqueduct 4.0 · NASA Sea Level Tool  |  "
    "Le TVR note le sol, pas les États — aucune frontière politique",
    ha='center', color='#555', fontsize=8)

for k,(g,r,c2,l) in enumerate([
    ("A+","85–100","#1B4332","Exceptionnel"),
    ("A", "70–84", "#2D6A4F","Haute capacité"),
    ("B", "55–69", "#95D5B2","Modéré"),
    ("C", "40–54", "#F4A261","Dégradé"),
    ("D", "25–39", "#E76F51","Sévère"),
    ("E", "0–24",  "#9B2226","Non-viable")]):
    y = 0.82 - k*0.07
    fig.add_artist(mpatches.FancyBboxPatch(
        (0.01, y-0.015), 0.016, 0.03,
        boxstyle="round,pad=0.002",
        facecolor=c2, edgecolor='none',
        transform=fig.transFigure, zorder=10))
    fig.text(0.03, y, f"{g}  {r}  {l}",
        transform=fig.transFigure,
        color='white', fontsize=7.5, va='center')

plt.tight_layout(rect=[0,0.03,1,0.95])
st.pyplot(fig, use_container_width=True)
