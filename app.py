"""
CineMatch - Netflix-inspired movie recommendation platform.

ML engine (unchanged): CountVectorizer (max_features=5000, english stop words)
+ cosine similarity over the `tags` column, top-5 neighbours.
The TMDB API is used ONLY for posters / metadata. The recommender never
depends on it, so the app works fully offline with placeholder artwork.

Run:  python -m streamlit run app.py
"""

from __future__ import annotations

import html
import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import joblib
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Discover Your Next Favourite Movie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
TMDB_API = "https://api.themoviedb.org/3"
IMG_POSTER = "https://image.tmdb.org/t/p/w500"
IMG_BACKDROP = "https://image.tmdb.org/t/p/w1280"


# --------------------------------------------------------------------------
# ML ENGINE  (your original loader + recommend() - unchanged)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    movies = joblib.load("movies.pkl")
    vectors = joblib.load("movie_vectors.pkl")
    return movies, vectors


movies, vectors = load_model()


def recommend(movie):
    movie_index = movies[
        movies["title"] == movie
    ].index[0]

    distances = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    movie_list = distances.argsort()[::-1]

    recommendations = []

    for i in movie_list:
        if i != movie_index:
            recommendations.append(
                movies.iloc[i]["title"]
            )

        if len(recommendations) == 5:
            break

    return recommendations


# --------------------------------------------------------------------------
# TMDB (optional, fully fault tolerant)
# --------------------------------------------------------------------------
def get_tmdb_credential() -> str:
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("TMDB_API_KEY", "")).strip()
    except Exception:
        return ""


def _tmdb_get(path: str, params: dict, cred: str):
    headers, params = {}, dict(params)
    if cred.startswith("eyJ"):  # v4 read-access token
        headers["Authorization"] = f"Bearer {cred}"
    else:
        params["api_key"] = cred
    r = requests.get(f"{TMDB_API}{path}", params=params, headers=headers, timeout=6)
    r.raise_for_status()
    return r.json()


def _fetch_details(movie_id, title: str, cred: str) -> dict:
    empty = {"ok": False, "poster": None, "backdrop": None, "overview": "",
             "year": "", "rating": None, "genres": []}
    if not cred:
        return empty
    try:
        data = None
        if movie_id:
            try:
                data = _tmdb_get(f"/movie/{int(movie_id)}", {"language": "en-US"}, cred)
            except Exception:
                data = None
        if data is None:  # fall back to title search
            res = _tmdb_get("/search/movie", {"query": title}, cred).get("results", [])
            if not res:
                return empty
            data = _tmdb_get(f"/movie/{res[0]['id']}", {"language": "en-US"}, cred)
        return {
            "ok": True,
            "poster": f"{IMG_POSTER}{data['poster_path']}" if data.get("poster_path") else None,
            "backdrop": f"{IMG_BACKDROP}{data['backdrop_path']}" if data.get("backdrop_path") else None,
            "overview": data.get("overview") or "",
            "year": (data.get("release_date") or "")[:4],
            "rating": round(float(data["vote_average"]), 1) if data.get("vote_average") else None,
            "genres": [g["name"] for g in data.get("genres", [])],
        }
    except Exception:
        return empty


@st.cache_data(ttl=3600, show_spinner=False)
def get_details_batch(items: tuple, cred: str) -> list[dict]:
    """items = ((movie_id, title), ...). Fetched in parallel, cached for an hour."""
    with ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(lambda it: _fetch_details(it[0], it[1], cred), items))


def placeholder_poster(title: str) -> str:
    lines = textwrap.wrap(title, 14)[:3] or [""]
    tspans = "".join(
        f'<tspan x="150" dy="{0 if i == 0 else 28}">{html.escape(l)}</tspan>'
        for i, l in enumerate(lines)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#2a0a0d"/><stop offset="1" stop-color="#111"/></linearGradient></defs>'
        '<rect width="300" height="450" fill="url(#g)"/>'
        '<rect x="14" y="14" width="272" height="422" rx="6" fill="none" stroke="#e50914" stroke-opacity=".35" stroke-width="2"/>'
        '<text x="150" y="190" text-anchor="middle" font-size="56">🎬</text>'
        f'<text x="150" y="262" text-anchor="middle" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#f2f2f2">{tspans}</text>'
        '</svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg)


def movie_id_of(row) -> int | None:
    """Don't assume columns exist - use whichever id column is present."""
    for col in ("movie_id", "id"):
        if col in row.index:
            try:
                return int(row[col])
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap');

:root{
  --bg:#070708; --panel:#141416; --panel-2:#1c1c20; --line:rgba(255,255,255,.08);
  --red:#e50914; --red-deep:#9b0610; --text:#f4f4f5; --muted:#a1a1aa;
}
html, body, [class*="css"], .stApp{ font-family:'DM Sans',system-ui,sans-serif; color:var(--text); }
.stApp{
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(229,9,20,.22), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(155,6,16,.18), transparent 55%),
    linear-gradient(180deg,#0b0b0d 0%, #070708 40%, #050505 100%);
}
#MainMenu, footer, header[data-testid="stHeader"]{ visibility:hidden; height:0; }
.block-container{ padding:0 2rem 3rem !important; max-width:1240px !important; }

/* ---------- navbar ---------- */
.nav{ display:flex; align-items:center; justify-content:space-between; padding:22px 4px; }
.logo{ font-family:'Bebas Neue',sans-serif; font-size:2.1rem; letter-spacing:.06em; line-height:1;
  background:linear-gradient(90deg,#ff2a36,#b20710); -webkit-background-clip:text; background-clip:text; color:transparent; }
.logo span{ color:var(--text); -webkit-text-fill-color:var(--text); }
.nav-links{ display:flex; gap:28px; }
.nav-links a{ color:var(--muted); text-decoration:none; font-weight:500; font-size:.95rem; transition:color .2s; }
.nav-links a:hover{ color:#fff; }

/* ---------- hero ---------- */
.hero{ text-align:center; padding:54px 12px 26px; position:relative; }
.hero:before{ content:""; position:absolute; inset:0 -2rem; z-index:-1; opacity:.5;
  background:repeating-linear-gradient(90deg, transparent 0 46px, rgba(255,255,255,.025) 46px 48px);
  -webkit-mask-image:linear-gradient(180deg,#000 30%,transparent); mask-image:linear-gradient(180deg,#000 30%,transparent); }
.hero h1{ font-family:'Bebas Neue',sans-serif; font-weight:400; font-size:clamp(2.8rem,8vw,6rem);
  line-height:.95; letter-spacing:.02em; margin:0 0 18px; color:#fff; text-shadow:0 6px 40px rgba(229,9,20,.35); }
.hero p{ max-width:640px; margin:0 auto; color:var(--muted); font-size:1.1rem; line-height:1.65; }

/* ---------- search panel ---------- */
.st-key-search_panel{ max-width:760px; margin:26px auto 0; padding:22px 22px 10px;
  background:rgba(20,20,22,.78); backdrop-filter:blur(10px); border:1px solid var(--line);
  border-radius:16px; box-shadow:0 30px 80px rgba(0,0,0,.55), 0 0 0 1px rgba(229,9,20,.08); }
.st-key-search_panel label p{ color:var(--muted) !important; font-size:.9rem !important; }
div[data-baseweb="select"] > div{ background:#0e0e10 !important; border:1px solid rgba(255,255,255,.12) !important;
  border-radius:10px !important; min-height:52px; transition:border-color .2s, box-shadow .2s; }
div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within{
  border-color:var(--red) !important; box-shadow:0 0 0 3px rgba(229,9,20,.18); }
div[data-baseweb="select"] *{ color:var(--text) !important; }
ul[role="listbox"], div[data-baseweb="popover"] > div{ background:#141416 !important; }
li[role="option"]:hover, li[aria-selected="true"]{ background:rgba(229,9,20,.22) !important; }

.st-key-go button{ width:100%; height:54px; border:0; border-radius:10px; font-weight:700; font-size:1.05rem;
  letter-spacing:.02em; color:#fff; background:linear-gradient(135deg,#ff2a36 0%,#e50914 45%,#9b0610 100%);
  box-shadow:0 10px 30px rgba(229,9,20,.35); transition:transform .18s, box-shadow .18s, filter .18s; }
.st-key-go button:hover{ transform:translateY(-2px); box-shadow:0 16px 40px rgba(229,9,20,.5); filter:brightness(1.08); color:#fff; border:0; }
.st-key-go button:active{ transform:translateY(0); }
.st-key-go button:focus-visible, .qp button:focus-visible{ outline:2px solid #fff; outline-offset:2px; }

/* quick picks */
.st-key-quick{ max-width:900px; margin:14px auto 0; }
.qp-label{ text-align:center; color:var(--muted); font-size:.85rem; margin:10px 0 6px; }
.st-key-quick button{ border-radius:999px; background:var(--panel); border:1px solid var(--line); color:var(--text);
  font-size:.85rem; padding:.25rem .9rem; transition:all .18s; }
.st-key-quick button:hover{ border-color:var(--red); color:#fff; background:rgba(229,9,20,.14); }

.note{ text-align:center; color:#7a7a84; font-size:.8rem; margin:14px auto 0; max-width:620px; }
.note b{ color:#c9c9d1; font-weight:500; }

/* ---------- section titles ---------- */
.sec{ margin:56px 0 6px; display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
.sec h2{ font-family:'Bebas Neue',sans-serif; font-weight:400; font-size:2rem; letter-spacing:.05em; margin:0;
  padding-left:14px; border-left:4px solid var(--red); line-height:1.1; }
.sec small{ color:var(--muted); font-size:.95rem; }

/* ---------- selected movie ---------- */
.feature{ position:relative; overflow:hidden; border-radius:18px; border:1px solid var(--line);
  background:var(--panel); margin-top:14px; box-shadow:0 24px 70px rgba(0,0,0,.5); }
.feature .bd{ position:absolute; inset:0; background-size:cover; background-position:center 20%; opacity:.42; filter:saturate(1.05); }
.feature .shade{ position:absolute; inset:0; background:linear-gradient(90deg,rgba(7,7,8,.97) 25%,rgba(7,7,8,.55) 70%,rgba(7,7,8,.3)); }
.feature .in{ position:relative; display:flex; gap:28px; padding:28px; align-items:flex-start; }
.feature img.poster{ width:170px; border-radius:10px; box-shadow:0 14px 40px rgba(0,0,0,.65); flex:none; }
.feature h3{ font-family:'Bebas Neue',sans-serif; font-weight:400; font-size:2.6rem; letter-spacing:.03em; margin:0 0 8px; line-height:1; }
.feature p{ color:#d4d4d8; line-height:1.65; max-width:680px; margin:12px 0 0; }
.chips{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.chip{ border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.06); border-radius:999px; padding:3px 12px; font-size:.82rem; color:#e4e4e7; }
.chip.rate{ background:rgba(229,9,20,.18); border-color:rgba(229,9,20,.55); color:#fff; font-weight:700; }

/* ---------- movie rail ---------- */
.rail{ display:grid; grid-auto-flow:column; grid-auto-columns:calc((100% - 64px)/5); gap:16px;
  overflow-x:auto; padding:18px 6px 26px; margin:0 -6px; scroll-snap-type:x mandatory; scrollbar-width:thin;
  scrollbar-color:#333 transparent; }
.rail::-webkit-scrollbar{ height:8px; } .rail::-webkit-scrollbar-thumb{ background:#2c2c31; border-radius:8px; }
.card{ scroll-snap-align:start; transition:transform .28s cubic-bezier(.2,.8,.2,1); }
.card .poster-wrap{ position:relative; border-radius:10px; overflow:hidden; aspect-ratio:2/3; background:#161618;
  box-shadow:0 10px 26px rgba(0,0,0,.5); transition:box-shadow .28s, outline-color .28s; outline:2px solid transparent; }
.card .poster-wrap img{ width:100%; height:100%; object-fit:cover; display:block; transition:transform .5s; }
.card:hover{ transform:translateY(-8px) scale(1.035); z-index:3; }
.card:hover .poster-wrap{ box-shadow:0 22px 50px rgba(229,9,20,.28), 0 18px 40px rgba(0,0,0,.7); outline-color:rgba(229,9,20,.7); }
.card:hover .poster-wrap img{ transform:scale(1.07); }
.badge{ position:absolute; top:10px; left:10px; background:rgba(0,0,0,.72); backdrop-filter:blur(4px);
  border:1px solid rgba(255,255,255,.14); border-radius:6px; padding:2px 8px; font-size:.78rem; font-weight:700; }
.badge.rank{ left:auto; right:10px; background:linear-gradient(135deg,#ff2a36,#9b0610); border:0; }
.overlay{ position:absolute; inset:0; padding:14px; display:flex; flex-direction:column; justify-content:flex-end;
  background:linear-gradient(180deg,transparent 30%,rgba(0,0,0,.92) 85%); opacity:0; transition:opacity .28s; }
.card:hover .overlay{ opacity:1; }
.overlay p{ margin:6px 0 0; font-size:.78rem; line-height:1.45; color:#d4d4d8;
  display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical; overflow:hidden; }
.overlay .g{ font-size:.75rem; color:#ff6a73; font-weight:500; }
.c-title{ margin-top:12px; font-weight:700; font-size:.98rem; line-height:1.3; }
.c-sub{ color:var(--muted); font-size:.82rem; margin-top:2px; }

/* ---------- loader ---------- */
.loader{ display:flex; flex-direction:column; align-items:center; gap:18px; padding:60px 0; }
.ring{ width:54px; height:54px; border-radius:50%; border:4px solid rgba(255,255,255,.08); border-top-color:var(--red);
  animation:spin .9s linear infinite; box-shadow:0 0 30px rgba(229,9,20,.25); }
.loader span{ color:var(--muted); letter-spacing:.04em; }
@keyframes spin{ to{ transform:rotate(360deg); } }

/* ---------- footer ---------- */
.footer{ margin-top:80px; padding:34px 0 10px; border-top:1px solid var(--line); text-align:center; color:var(--muted); font-size:.9rem; line-height:1.8; }
.footer .logo{ font-size:1.6rem; display:inline-block; margin-bottom:6px; }
.footer a{ color:#d4d4d8; }
.tech{ display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin:12px 0; }

@media (max-width:820px){
  .block-container{ padding:0 1rem 2rem !important; }
  .nav-links{ display:none; }
  .rail{ grid-auto-columns:62%; }
  .feature .in{ flex-direction:column; padding:20px; }
  .feature img.poster{ width:130px; }
  .feature h3{ font-size:2.1rem; }
  .hero{ padding-top:30px; }
}
@media (prefers-reduced-motion:reduce){
  *{ animation:none !important; transition:none !important; }
}
</style>
"""


def H(s: str) -> str:
    """Strip indentation / blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


# --------------------------------------------------------------------------
# HTML builders
# --------------------------------------------------------------------------
def card_html(title: str, d: dict, rank: int) -> str:
    img = d["poster"] or placeholder_poster(title)
    t = html.escape(title)
    genres = ", ".join(d["genres"][:3])
    sub = " • ".join(x for x in [d["year"], genres] if x)
    rating = f'<div class="badge">★ {d["rating"]}</div>' if d["rating"] else ""
    overview = html.escape(d["overview"]) if d["overview"] else "Similar in themes, cast, genre and storyline."
    return H(f"""
    <div class="card">
      <div class="poster-wrap">
        <img src="{html.escape(img, quote=True)}" alt="{t} poster" loading="lazy">
        {rating}
        <div class="badge rank">#{rank}</div>
        <div class="overlay">
          <div class="g">{html.escape(genres)}</div>
          <p>{overview}</p>
        </div>
      </div>
      <div class="c-title">{t}</div>
      <div class="c-sub">{html.escape(sub) if sub else "Recommended for you"}</div>
    </div>
    """)


def feature_html(title: str, d: dict) -> str:
    img = d["poster"] or placeholder_poster(title)
    backdrop = (f'<div class="bd" style="background-image:url({html.escape(d["backdrop"], quote=True)})"></div>'
                if d["backdrop"] else '<div class="bd" style="background:radial-gradient(600px 300px at 80% 0%,#4a0b10,transparent)"></div>')
    chips = ""
    if d["rating"]:
        chips += f'<span class="chip rate">★ {d["rating"]} / 10</span>'
    if d["year"]:
        chips += f'<span class="chip">{html.escape(d["year"])}</span>'
    for g in d["genres"][:4]:
        chips += f'<span class="chip">{html.escape(g)}</span>'
    overview = (html.escape(d["overview"]) if d["overview"]
                else "Add a free TMDB API key to see the synopsis, rating and genres for every title.")
    return H(f"""
    <div class="feature">
      {backdrop}<div class="shade"></div>
      <div class="in">
        <img class="poster" src="{html.escape(img, quote=True)}" alt="{html.escape(title)} poster">
        <div>
          <h3>{html.escape(title)}</h3>
          <div class="chips">{chips}</div>
          <p>{overview}</p>
        </div>
      </div>
    </div>
    """)


def section(title: str, sub: str = "", anchor: str = "") -> str:
    small = f"<small>{html.escape(sub)}</small>" if sub else ""
    return f'<div class="sec" id="{anchor}"><h2>{html.escape(title)}</h2>{small}</div>'


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

cred = get_tmdb_credential()
titles = movies["title"].tolist()

# Navigation + hero ---------------------------------------------------------
st.markdown(H("""
<div class="nav">
  <div class="logo">CINE<span>MATCH</span></div>
  <div class="nav-links">
    <a href="#discover">Discover</a><a href="#recs">Top Recommendations</a><a href="#about">About</a>
  </div>
</div>
<div class="hero" id="discover">
  <h1>Discover Your Next<br>Favourite Movie</h1>
  <p>Pick a film you love. CineMatch reads its story, genres, keywords, cast and director,
  then uses machine learning to find the five movies that feel most like it.</p>
</div>
"""), unsafe_allow_html=True)

# Search panel -------------------------------------------------------------
with st.container(key="search_panel"):
    selected = st.selectbox(
        "Search or choose a movie",
        options=titles,
        index=None,
        placeholder="Start typing a title, e.g. Inception…",
        key="movie_select",
    )
    with st.container(key="go"):
        go = st.button("Get Recommendations", use_container_width=True)

# Quick picks (only titles that exist in the user's data) --------------------
wanted = ["Avatar", "Inception", "The Dark Knight Rises", "Interstellar", "Titanic", "The Avengers", "Toy Story"]
quick = [t for t in wanted if t in set(titles)][:5]
if quick:
    st.markdown('<div class="qp-label">Not sure? Try one of these</div>', unsafe_allow_html=True)
    with st.container(key="quick"):
        cols = st.columns(len(quick))
        for col, t in zip(cols, quick):
            col.button(t, key=f"qp_{t}", use_container_width=True,
                       on_click=lambda t=t: st.session_state.update(movie_select=t))

if not cred:
    st.markdown('<div class="note"><b>Offline mode:</b> add a free TMDB API key to show real posters, '
                'ratings and synopses. Recommendations work either way.</div>', unsafe_allow_html=True)

# Selected movie info -------------------------------------------------------
if selected:
    row = movies[movies["title"] == selected].iloc[0]
    sel_details = get_details_batch(((movie_id_of(row), selected),), cred)[0]
    st.markdown(section("You Selected", "Here is what we know about it"), unsafe_allow_html=True)
    st.markdown(feature_html(selected, sel_details), unsafe_allow_html=True)

# Recommendations -----------------------------------------------------------
if go:
    if not selected:
        st.warning("Please choose a movie first.")
    else:
        holder = st.empty()
        holder.markdown(H("""
        <div class="loader"><div class="ring"></div><span>Analysing the plot, cast and genres…</span></div>
        """), unsafe_allow_html=True)
        t0 = time.time()
        try:
            rec_titles = recommend(selected)
            items = tuple(
                (movie_id_of(movies[movies["title"] == t].iloc[0]), t) for t in rec_titles
            )
            details = get_details_batch(items, cred)
            st.session_state["result"] = {
                "source": selected,
                "titles": [i[1] for i in items],
                "details": details,
            }
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("result", None)
            st.error(f"Sorry, something went wrong while generating recommendations: {e}")
        remaining = 0.7 - (time.time() - t0)  # keep the loader visible long enough to read
        if remaining > 0:
            time.sleep(remaining)
        holder.empty()

res = st.session_state.get("result")
if res:
    cards = "".join(card_html(t, d, i + 1) for i, (t, d) in enumerate(zip(res["titles"], res["details"])))
    st.markdown(section("Top Recommendations", f"Because you liked {res['source']}", "recs"), unsafe_allow_html=True)
    st.markdown(f'<div class="rail">{cards}</div>', unsafe_allow_html=True)

# Footer ---------------------------------------------------------------------
st.markdown(H("""
<div class="footer" id="about">
  <div class="logo">CINE<span>MATCH</span></div>
  <div>A content-based movie recommendation system built on the TMDB 5000 Movies dataset.</div>
  <div class="tech">
    <span class="chip">Python</span><span class="chip">Streamlit</span><span class="chip">Pandas</span>
    <span class="chip">Scikit-learn</span><span class="chip">CountVectorizer</span><span class="chip">Cosine Similarity</span>
  </div>
  <div>Movie data and imagery provided by TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.</div>
</div>
"""), unsafe_allow_html=True)