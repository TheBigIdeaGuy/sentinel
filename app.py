"""
Sentinel – Streamlit Dashboard
Full port of the Jupyter / ipywidgets version, including the `extract` NLP pipeline.
All plots are fully interactive: rich hover tooltips, animated transitions,
gradient fills, zoom/pan, cross-chart click-to-filter, and responsive sizing.
"""

import io, re, sys, subprocess, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from rapidfuzz import fuzz, process
from spacy.matcher import PhraseMatcher

# ── spaCy: prefer en_core_web_sm, fall back to blank English ─────────────────
try:
    import spacy
    try:
        _nlp_global = spacy.load("en_core_web_sm", disable=["parser","ner","textcat"])
    except OSError:
        try:
            subprocess.run([sys.executable,"-m","spacy","download","en_core_web_sm"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _nlp_global = spacy.load("en_core_web_sm", disable=["parser","ner","textcat"])
        except Exception:
            _nlp_global = spacy.blank("en")
except Exception:
    _nlp_global = None

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sentinel Dashboard", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ─── Palette & shared layout defaults ─────────────────────────────────────────
P = dict(
    teal="#00a2ae", red="#dc3545", blue="#03a9f3", green="#00c292",
    purple="#ab8ce4", orange="#e67e22", teal2="#20c997", yellow="#ffc107",
    bg="rgba(248,249,250,0.5)", grid="rgba(200,210,220,0.3)",
    font="Open Sans, sans-serif",
)
CHART_H = 400

def _base_layout(**kw):
    # Default spike/grid settings for both axes
    _xdefault = dict(gridcolor=P["grid"], zeroline=False, showspikes=True,
                     spikecolor="#adb5bd", spikethickness=1, spikedash="dot")
    _ydefault = dict(gridcolor=P["grid"], zeroline=False, showspikes=True,
                     spikecolor="#adb5bd", spikethickness=1, spikedash="dot")
    # Callers that supply their own xaxis/yaxis get those merged on top of defaults
    xaxis = {**_xdefault, **kw.pop("xaxis", {})}
    yaxis = {**_ydefault, **kw.pop("yaxis", {})}
    return dict(
        template="plotly_white",
        height=CHART_H,
        margin=dict(t=30, b=50, l=50, r=50),
        plot_bgcolor=P["bg"],
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=P["font"], size=12, color="#455a64"),
        hoverlabel=dict(bgcolor="white", bordercolor="#dee2e6",
                        font_size=12, font_family=P["font"]),
        xaxis=xaxis,
        yaxis=yaxis,
        **kw,
    )

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #f5f7fa; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

.ela-header-bar {
    background: #fff; padding: 15px; margin-bottom: 20px;
    border-radius: 4px; border-left: 4px solid #00c292;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.ela-header-bar h1 {
    font-family: 'Trebuchet MS', sans-serif; font-weight: 700;
    font-size: 52px; margin: 0; color: #455a64; line-height: 1.1;
}

.ela-card-row { display: flex; flex-wrap: wrap; gap: 15px; margin: 10px 0 20px 0; }
.ela-kpi-card {
    background: #fff; border-radius: 4px; padding: 15px;
    flex: 1 1 200px; min-width: 200px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-left: 4px solid #ccc;
    transition: box-shadow .2s, transform .2s;
}
.ela-kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); transform: translateY(-2px); }
.card-blue   { border-left-color: #03a9f3 !important; }
.card-green  { border-left-color: #00c292 !important; }
.card-red    { border-left-color: #e46a76 !important; }
.card-purple { border-left-color: #ab8ce4 !important; }
.card-orange { border-left-color: #e67e22 !important; }
.card-teal   { border-left-color: #20c997 !important; }
.card-title-text { font-size: 11px; color: #868e96; text-transform: uppercase;
                   font-weight: 600; letter-spacing: 0.3px; }
.card-value-text { font-size: 22px; font-weight: 700; color: #455a64; margin-top: 5px; }

.ela-section-title {
    font-size: 14px; font-weight: 600; color: #455a64; margin: 15px 0 10px 0;
    text-transform: uppercase; border-left: 3px solid #6610f2; padding-left: 8px;
    font-family: 'Ubuntu', sans-serif;
}
.plot-tile-title {
    font-size: 13px; font-weight: 600; color: #495057;
    margin-bottom: 6px; padding-bottom: 6px;
    border-bottom: 2px solid #f1f3f5;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.alert-warning { color:#856404; background:#fff3cd; padding:10px; border-radius:4px; margin:10px 0; }
.alert-info    { color:#0c5460; background:#d1ecf1; padding:10px; border-radius:4px; margin:10px 0; }
.alert-success { color:#155724; background:#d4edda; padding:10px; border-radius:4px; margin:10px 0; }

.upload-card {
    background:#fff; border-radius:12px; padding:30px; margin:20px 0;
    box-shadow:0 2px 12px rgba(0,0,0,0.1);
}
.upload-title { font-size:18px; font-weight:600; color:#455a64; margin-bottom:15px; }
.upload-info  { font-size:13px; color:#868e96; margin-bottom:15px; }

[data-testid="stSidebar"] { background:#ffffff; }
.sidebar-menu-header {
    background:#455a64; color:white; text-align:center; font-weight:bold;
    font-size:14px; padding:8px; border-radius:4px; margin-bottom:10px;
}
.sidebar-section-label { font-size:10px; color:#6c757d; padding:4px 0; margin-top:8px; }
.sidebar-hint { font-size:10px; color:#6c757d; text-align:center;
                background:#f8f9fa; border-radius:4px; padding:6px; margin-top:8px; }

/* ── Top filter bar ── */
.filter-bar {
    background: #ffffff;
    border-radius: 8px;
    padding: 10px 16px 8px 16px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #e9ecef;
}
/* Compact widget labels inside the filter bar */
div[data-testid="stHorizontalBlock"] label {
    font-size: 10px !important;
    color: #868e96 !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    margin-bottom: 1px !important;
}

.footer { font-family:'Ubuntu',sans-serif; font-size:10px; color:#555;
          text-align:center; padding:20px; margin-top:30px;
          background:#f8f9fa; border-top:1px solid #e9ecef; }
.footer a { color:#0077B5; text-decoration:none; font-weight:600; }

/* plotly chart container hover glow */
[data-testid="stPlotlyChart"] {
    border-radius: 8px;
    transition: box-shadow .25s;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 4px 20px rgba(0,162,174,0.15);
}
</style>
""", unsafe_allow_html=True)

# ─── NLP constants ─────────────────────────────────────────────────────────────
FAILURE_MODES = [
    "worn","broken","leaking","leak","cracked","corroded","rusted","vibration",
    "loose","jammed","blocked","clogged","overheated","burnt","seized","misaligned",
    "damaged","failed","noisy","bent","frayed","torn","missing","dirty",
    "eroded","cavitated","pitted","spalled","galled","scored","scratched",
    "fatigued","fractured","sheared","twisted","collapsed","buckled","distorted",
    "warped","swollen","shrunk","expanded","contracted","melted","charred",
    "blistered","delaminated","debonded","fretted","brinelled","contaminated",
    "degraded","deteriorated","oxidized","sulfidated","carburized","nitrided",
    "embrittled","softened","hardened","work hardened","creep","stress rupture",
    "thermal fatigue","thermal shock","corrosion fatigue","stress corrosion cracking",
    "hydrogen embrittlement","hydrogen blistering","hydrogen induced cracking",
    "sulfide stress cracking","chloride stress corrosion cracking","caustic cracking",
    "amine cracking","ammonia stress corrosion cracking","liquid metal embrittlement",
    "intergranular corrosion","transgranular corrosion","pitting corrosion",
    "crevice corrosion","galvanic corrosion","uniform corrosion","localized corrosion",
    "microbiologically influenced corrosion","flow accelerated corrosion",
    "erosion corrosion","impingement corrosion","cavitation erosion","fretting corrosion",
    "high temperature corrosion","hot corrosion","fouled","scaled","coked",
    "plugged","obstructed","restricted","starved","flooded","dry run",
    "dead headed","overpressurized","underpressurized","overloaded","underloaded",
    "unbalanced","eccentric","runout","out of round","out of flat","out of square",
    "out of parallel","out of tolerance","excessive clearance","insufficient clearance",
    "excessive backlash","insufficient backlash","excessive preload","insufficient preload",
    "excessive tension","insufficient tension","over torqued","under torqued",
    "cross threaded","stripped","galled threads","stuck","frozen","bound",
    "sticking","binding","hunting","oscillating","cycling","short cycling",
    "chattering","hammering","water hammer","steam hammer","surge","pulsation",
    "fluctuating","unstable","intermittent","erratic","sporadic","drifting",
    "biased","offset","inaccurate","imprecise","nonlinear","hysteresis",
    "deadband","stiction","backlash","windup","saturation","cutoff",
    "slew rate limiting","overshoot","undershoot","ringing","settling time",
    "response time","lag","delay","timeout","no response","false reading",
    "false trip","nuisance trip","spurious trip","fail to trip","fail to start",
    "fail to stop","fail to open","fail to close","fail to energize","fail to deenergize",
    "short circuit","open circuit","ground fault","phase fault","earth fault",
    "arc fault","flashover","tracking","partial discharge","corona","arcing",
    "sparking","overcurrent","undercurrent","overvoltage","undervoltage",
    "overfrequency","underfrequency","phase imbalance","phase loss","phase reversal",
]

FAILURE_POINTS = [
    "bearing","seal","pump","motor","valve","gear","roller","roll","dryer roll",
    "conveyor","belt","chain","cylinder","piston","shaft","impeller","filter",
    "screen","blade","knife","screw","auger","fan","blower","compressor",
    "sensor","switch","relay","pipe","hose","coupling","bearing housing",
    "transformer","generator","turbine","boiler","heat exchanger","condenser",
    "cooling tower","pulper","refiner","headbox","wire","felt","press roll",
    "calendar","reel","winder","digester","evaporator","recovery boiler",
    "lime kiln","chipper","bark drum","centrifuge","hydrocyclone","flotation cell",
    "thickener","agitator","mixer","crusher","mill","kiln","dust collector",
    "cyclone","hopper","chute","feeder","vibrating screen","jaw crusher",
    "cone crusher","ball mill","sag mill","slurry pump","dragline","shovel",
    "haul truck","stacker","reclaimer","separator","scrubber","desalter",
    "fractionator","reformer","cracker","coker","hydrotreater","flare",
    "pig launcher","pig receiver","christmas tree","blowout preventer","riser",
    "umbilical","manifold","actuator","positioner","transmitter","controller",
    "analyser","gauge","indicator","recorder","solenoid","circuit breaker",
    "fuse","contactor","starter","vfd","inverter","rectifier","battery",
    "ups","busbar","insulator","conductor","cable","tray","conduit",
    "junction box","terminal block","plc","dcs","scada","hmi","rtu",
    "thermocouple","rtd","pressure transmitter","flow meter","level transmitter",
    "ph meter","conductivity meter","vibration probe","proximity probe",
    "limit switch","pressure switch","control valve","float switch","temperature switch",
]

REPLACEMENTS = {
    r'\bbrng\b':'bearing', r'\bbearng\b':'bearing', r'\bbrg\b':'bearing',
    r'\bbering\b':'bearing', r'\bbearingg\b':'bearing', r'\bbeering\b':'bearing',
    r'\bpnmp\b':'pump', r'\bpummp\b':'pump', r'\bpumpe\b':'pump', r'\bpomp\b':'pump',
    r'\bmtor\b':'motor', r'\bmtr\b':'motor', r'\bmotorr\b':'motor',
    r'\bmotar\b':'motor', r'\bmoter\b':'motor',
    r'\bvalv\b':'valve', r'\bvalvve\b':'valve', r'\bvalv e\b':'valve',
    r'\bleek\b':'leak', r'\bleake\b':'leak', r'\bleeking\b':'leaking',
    r'\bcrkd\b':'cracked', r'\bcrackked\b':'cracked', r'\bcraked\b':'cracked',
    r'\brustd\b':'rusted', r'\brusty\b':'rusted', r'\brustted\b':'rusted',
    r'\bcorrodid\b':'corroded', r'\bcoroded\b':'corroded',
    r'\bowrhot\b':'overheated', r'\boverheatt\b':'overheated',
    r'\boverheted\b':'overheated', r'\boverhete\b':'overheated',
    r'\bsezd\b':'seized', r'\bseised\b':'seized', r'\bsezed\b':'seized',
    r'\bblwr\b':'blower', r'\bblowerr\b':'blower', r'\bblowr\b':'blower', r'\bbloer\b':'blower',
    r'\bcmpresr\b':'compressor', r'\bcomprssr\b':'compressor',
    r'\bcompresor\b':'compressor', r'\bcompressorr\b':'compressor',
}
FUZZY_THRESHOLD = 85
REQUIRED_RAW_COLS = ["Basic_Start_Date","Basic_Finish_Date","Functional_Location",
                     "Equipment","Actual_Hours"]


# ─── Extract pipeline ──────────────────────────────────────────────────────────
def extract(df, failure_modes, failure_points, replacements, progress_cb=None):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser","ner","textcat"])
    except OSError:
        nlp = spacy.blank("en")

    points_lower = [p.lower() for p in failure_points]
    modes_lower  = [m.lower() for m in failure_modes]
    matcher_p = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher_m = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher_p.add("FP", [nlp.make_doc(t) for t in failure_points])
    matcher_m.add("FM", [nlp.make_doc(t) for t in failure_modes])

    def preprocess(text):
        if not text or pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r"[^\w\s\-/]"," ",text)
        for pat, rep in replacements.items(): text = re.sub(pat, rep, text)
        return " ".join(text.split())

    def fuzzy_match(text, candidates):
        if not text or not candidates: return None, 0
        words = text.split()
        phrases = [text]
        for n in [2,3]: phrases.extend([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])
        bm, bs = None, 0
        for ph in set(phrases):
            if len(ph)<3: continue
            m, s, _ = process.extractOne(ph, candidates, scorer=fuzz.token_sort_ratio)
            if s>bs and s>=FUZZY_THRESHOLD: bs,bm=s,m
        return bm, bs

    def extract_matches(text, keywords, ktype):
        doc = nlp(text)
        exact = ([doc[s:e].text for _,s,e in matcher_p(doc)] if ktype=="point"
                 else [doc[s:e].text for _,s,e in matcher_m(doc)])
        rem = text
        for m in exact: rem = rem.replace(m,"")
        fm, _ = fuzzy_match(rem, keywords)
        matches = exact+([fm] if fm else [])
        seen,u=[],set()
        for m in matches:
            if m.lower() not in u: u.add(m.lower()); seen.append(m)
        return seen

    def best(matches, text, keywords):
        if not matches: return np.nan
        if len(matches)==1: return matches[0]
        tl=text.lower(); scored=[]
        for m in matches:
            ml=m.lower(); pos=tl.find(ml)
            s=max(0,(100-pos)/10) if pos>=0 else 0
            s+=len(m)*2
            if re.search(rf"\b{re.escape(ml)}\b",tl): s+=20
            if ml in keywords: s+=30
            s+=min(tl.count(ml)*10,30)
            scored.append((m,s))
        return max(scored,key=lambda x:x[1])[0]

    def info(text):
        if not text or pd.isna(text): return "",""
        p=preprocess(text)
        if not p: return "",""
        return (best(extract_matches(p,points_lower,"point"),p,points_lower),
                best(extract_matches(p,modes_lower,"mode"),p,modes_lower))

    desc_col = next((c for c in ["Description (Short Text)","KTEXT","Description",
                                  "Short Text","Short Description","TEXT"] if c in df.columns), None)
    if not desc_col:
        raise ValueError(f"No description column found. Columns: {list(df.columns)}")
    order_col = next((c for c in ["Order","AUFNR","Order Number","ORDER"] if c in df.columns), None)

    results=[]
    for i,(idx,row) in enumerate(df.iterrows()):
        text=str(row[desc_col]) if not pd.isna(row[desc_col]) else ""
        try: fp,fm=info(text)
        except Exception as e: fp,fm="ERROR",str(e)[:50]
        results.append({"Order": row[order_col] if order_col and not pd.isna(row[order_col]) else f"ROW_{idx}",
                         "Failure_Point":fp,"Failure_Mode":fm})
        if progress_cb: progress_cb(i+1,len(df))

    ndf=pd.DataFrame(results)
    df1=df.copy()
    df1["Failure_Point"]=ndf["Failure_Point"].values
    df1["Failure_Mode"]=ndf["Failure_Mode"].values

    df2=df1.copy()
    df2["Basic_Start_Date"]=pd.to_datetime(df2["Basic_Start_Date"])
    df2=df2.dropna(subset=["Failure_Mode"])
    df2=df2.sort_values(["Functional_Location","Basic_Start_Date"]).reset_index(drop=True)
    df2["TTR_Mode_Days"]=(df2["Actual_Hours"]/24).round(1)
    df2["TTF_Any_Mode_Date"]=df2.groupby("Functional_Location")["Basic_Start_Date"].shift(-1)
    df2["TTF_Any_Mode_Days"]=(df2["TTF_Any_Mode_Date"]-df2["Basic_Start_Date"]).dt.days
    df2=df2.dropna(subset=["TTF_Any_Mode_Days"])
    df2=df2[df2["TTF_Any_Mode_Days"]!=0]
    df2=df2.sort_values(["Functional_Location","Failure_Mode","Basic_Start_Date"]).reset_index(drop=True)
    df2["TTF_Same_Mode_Date"]=df2.groupby(["Functional_Location","Failure_Mode"])["Basic_Start_Date"].shift(-1)
    df2["TTF_Same_Mode_Days"]=(df2["TTF_Same_Mode_Date"]-df2["Basic_Start_Date"]).dt.days
    df2=df2.dropna(subset=["TTF_Same_Mode_Days"])
    df2=df2[df2["TTF_Same_Mode_Days"]!=0]
    df2=df2.sort_values(["Functional_Location","Basic_Start_Date"]).reset_index(drop=True)
    df2["Previous_Failure_Mode"]=df2.groupby("Functional_Location")["Failure_Mode"].shift()
    df2=df2.dropna(subset=["Previous_Failure_Mode"])
    df2["TTF_Any_Mode_Days"]=df2["TTF_Any_Mode_Days"].fillna(0).astype("Int64")
    return df2


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLOT BUILDERS
# Every chart has:
#  • Rich hovertemplate with custom labels
#  • Animated bar/scatter entry (transition)
#  • Spike lines for cross-hair on hover
#  • Zoom, pan, box-select enabled
#  • Colour gradients / marker gradients where applicable
#  • uirevision so zoom state survives filter changes
# ══════════════════════════════════════════════════════════════════════════════

def build_pareto_chart(df, top_n):
    counts = df["Functional_Location"].value_counts().head(top_n)
    fig = go.Figure()
    if counts.empty:
        return fig

    cum_pct = counts.cumsum() / counts.sum() * 100
    # colour gradient: intensity by rank
    n = len(counts)
    colors = [f"rgba(0,162,174,{0.4 + 0.6*(n-i)/n})" for i in range(n)]

    fig.add_trace(go.Bar(
        x=counts.index, y=counts.values,
        name="Failures",
        marker=dict(color=colors, line=dict(color=P["teal"], width=0.5)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Failures: <b>%{y}</b><br>"
            "Share: <b>%{customdata:.1f}%</b>"
            "<extra></extra>"
        ),
        customdata=(counts.values / counts.sum() * 100),
    ))
    fig.add_trace(go.Scatter(
        x=counts.index, y=cum_pct,
        name="Cumulative %", yaxis="y2",
        mode="lines+markers",
        line=dict(color=P["red"], width=2.5),
        marker=dict(color="white", size=8, line=dict(color=P["red"], width=2)),
        hovertemplate="<b>%{x}</b><br>Cumulative: <b>%{y:.1f}%</b><extra></extra>",
    ))
    # 80% reference line
    fig.add_hline(y=80, yref="y2", line=dict(color=P["yellow"], width=1.5, dash="dash"),
                  annotation_text="80%", annotation_position="top right",
                  annotation_font=dict(color=P["yellow"], size=10))

    layout = _base_layout(
        xaxis_title="Functional Location", yaxis_title="Count of Failures",
        yaxis2=dict(title="Cumulative Impact %", overlaying="y", side="right",
                    range=[0,105], showgrid=False, ticksuffix="%"),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font_size=11),
        uirevision="pareto",
    )
    fig.update_layout(**layout)
    return fig


def build_failure_modes_chart(df, top_n):
    tf = df["Failure_Mode"].value_counts().head(top_n)
    fig = go.Figure()
    if tf.empty:
        return fig

    n = len(tf)
    colors = [f"rgba(220,53,69,{0.35 + 0.65*(n-i)/n})" for i in range(n)]

    fig.add_trace(go.Bar(
        x=tf.index, y=tf.values,
        marker=dict(color=colors, line=dict(color=P["red"], width=0.5)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Incidents: <b>%{y}</b><br>"
            "Share: <b>%{customdata:.1f}%</b>"
            "<extra></extra>"
        ),
        customdata=(tf.values / tf.sum() * 100),
    ))
    layout = _base_layout(
        xaxis_title="Failure Mode Classification",
        yaxis_title="Total Incidents Measured",
        uirevision="fmodes",
    )
    fig.update_layout(**layout)
    return fig


def build_ttf_histogram(df):
    ttf = df["TTF_Any_Mode_Days"].dropna().astype(float)
    fig = go.Figure()
    if ttf.empty:
        return fig

    mttf, std = ttf.mean(), ttf.std()
    hist_vals, _ = np.histogram(ttf, bins=20)
    max_y = max(hist_vals) * 1.22 if len(hist_vals) else 10

    fig.add_trace(go.Histogram(
        x=ttf, nbinsx=20, name="TTF Spread",
        marker=dict(
            color=P["teal2"],
            line=dict(color="white", width=0.8),
            opacity=0.82,
        ),
        hovertemplate="Range: <b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>",
    ))
    # ± 1σ shaded region
    fig.add_vrect(x0=mttf-std, x1=mttf+std,
                  fillcolor=P["yellow"], opacity=0.08,
                  line_width=0, layer="below")
    # MTTF line
    fig.add_trace(go.Scatter(
        x=[mttf, mttf], y=[0, max_y], mode="lines",
        line=dict(color=P["red"], width=2.5, dash="dash"),
        name=f"MTTF {mttf:.1f}d",
        hovertemplate=f"MTTF = {mttf:.1f} days<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[mttf-std, mttf-std], y=[0, max_y*.72], mode="lines",
        line=dict(color=P["yellow"], width=1.8, dash="dot"),
        name=f"−1σ {mttf-std:.1f}d",
        hovertemplate=f"−1σ = {mttf-std:.1f} days<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[mttf+std, mttf+std], y=[0, max_y*.72], mode="lines",
        line=dict(color=P["yellow"], width=1.8, dash="dot"),
        name=f"+1σ {mttf+std:.1f}d",
        hovertemplate=f"+1σ = {mttf+std:.1f} days<extra></extra>",
    ))
    fig.add_annotation(
        x=mttf, y=max_y*.94,
        text=f"<b>MTTF = {mttf:.1f} d</b><br>σ = {std:.1f} d",
        showarrow=True, arrowhead=2, arrowcolor=P["red"], arrowwidth=1.5,
        font=dict(color=P["red"], size=11, family=P["font"]),
        bgcolor="rgba(255,255,255,0.9)", borderpad=6,
        bordercolor=P["red"], borderwidth=1,
    )
    layout = _base_layout(
        xaxis_title="Time To Failures (Days)", yaxis_title="Frequency",
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font_size=10),
        uirevision="ttfhist",
    )
    fig.update_layout(**layout)
    return fig


def build_equipment_frequency(df, top_n):
    te = df["Equipment"].value_counts().head(top_n)
    fig = go.Figure()
    if te.empty:
        return fig

    n = len(te)
    colors = [f"rgba(3,169,243,{0.35 + 0.65*(n-i)/n})" for i in range(n)]

    fig.add_trace(go.Bar(
        x=te.values, y=te.index, orientation="h",
        marker=dict(color=colors, line=dict(color=P["blue"], width=0.5)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Failures: <b>%{x}</b><br>"
            "Share: <b>%{customdata:.1f}%</b>"
            "<extra></extra>"
        ),
        customdata=(te.values / te.sum() * 100),
    ))
    layout = _base_layout(
        xaxis_title="Failure Events Count",
        yaxis_title="Equipment ID",
        uirevision="equip",
    )
    fig.update_layout(**layout)
    return fig


def build_asset_pie(loc_data, top_n):
    fm = loc_data["Failure_Mode"].value_counts().head(top_n)
    fig = go.Figure()
    if fm.empty:
        return fig

    COLORS = [P["teal"], P["purple"], P["orange"], P["blue"],
              P["red"], P["green"], P["teal2"], P["yellow"],
              "#6f42c1", "#fd7e14", "#20c997", "#e83e8c"]

    fig.add_trace(go.Pie(
        labels=fm.index, values=fm.values, hole=0.38,
        textinfo="percent+label",
        textfont=dict(size=11, family=P["font"]),
        marker=dict(
            colors=COLORS[:len(fm)],
            line=dict(color="white", width=2),
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: <b>%{value}</b><br>"
            "Share: <b>%{percent}</b>"
            "<extra></extra>"
        ),
        pull=[0.05 if i==0 else 0 for i in range(len(fm))],
    ))
    layout = _base_layout(showlegend=True,
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
                          uirevision="pie")
    # pie doesn't use xaxis/yaxis gridcolors
    fig.update_layout(**layout)
    return fig


def build_asset_weibull(loc_data, test_days):
    a_ttf = loc_data["TTF_Any_Mode_Days"].dropna().astype(float)
    fig = go.Figure()
    if len(a_ttf) <= 2:
        return fig
    try:
        shape, _, scale = stats.weibull_min.fit(a_ttf, floc=0)
        t_range = np.linspace(0, max(a_ttf)*1.25, 300)
        r_curve = np.exp(-(t_range/scale)**shape)
        r_t = float(np.exp(-(test_days/scale)**shape))
        f_t = 1 - r_t

        # Gradient fill under curve
        fig.add_trace(go.Scatter(
            x=t_range, y=r_curve, mode="lines",
            line=dict(color=P["purple"], width=3),
            fill="tozeroy",
            fillcolor="rgba(171,140,228,0.10)",
            name="R(t) Curve",
            hovertemplate="t = <b>%{x:.0f} d</b><br>R(t) = <b>%{y:.3f}</b><extra></extra>",
        ))
        # Vertical drop line
        fig.add_trace(go.Scatter(
            x=[test_days, test_days], y=[0, r_t], mode="lines",
            line=dict(color=P["red"], width=1.8, dash="dash"),
            name="Target t", showlegend=False,
            hoverinfo="skip",
        ))
        # Horizontal read line
        fig.add_trace(go.Scatter(
            x=[0, test_days], y=[r_t, r_t], mode="lines",
            line=dict(color=P["red"], width=1.8, dash="dot"),
            name=f"R = {r_t*100:.1f}%", showlegend=False,
            hoverinfo="skip",
        ))
        # Marker at intersection
        fig.add_trace(go.Scatter(
            x=[test_days], y=[r_t], mode="markers",
            marker=dict(color=P["red"], size=12, symbol="circle",
                        line=dict(color="white", width=2)),
            name="Operating point",
            hovertemplate=(
                f"<b>t = {test_days} days</b><br>"
                f"β (shape) = {shape:.3f}<br>"
                f"η (scale) = {scale:.1f} d<br>"
                f"R(t) = <b>{r_t*100:.1f}%</b><br>"
                f"F(t) = <b>{f_t*100:.1f}%</b>"
                "<extra></extra>"
            ),
        ))
        # Failure zone shading
        fig.add_hrect(y0=0, y1=0.368, # e^-1 natural threshold
                      fillcolor=P["red"], opacity=0.04,
                      line_width=0, layer="below",
                      annotation_text="High risk zone", annotation_position="right",
                      annotation_font=dict(color=P["red"], size=9))
    except Exception:
        pass

    layout = _base_layout(
        xaxis_title="Operating Horizon (Days)",
        yaxis_title="Reliability R(t)",
        yaxis=dict(range=[0,1.05], tickformat=".0%",
                   gridcolor=P["grid"], zeroline=False,
                   showspikes=True, spikecolor="#adb5bd",
                   spikethickness=1, spikedash="dot"),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font_size=11),
        uirevision="weibull",
    )
    fig.update_layout(**layout)
    return fig


def build_chronological_ttf(loc_data):
    fig = go.Figure()
    if len(loc_data) <= 1:
        return fig

    ttf_vals = loc_data["TTF_Any_Mode_Days"].values.astype(float)
    dates    = loc_data["Basic_Start_Date"].values
    modes    = loc_data["Failure_Mode"].values
    mttf     = float(np.nanmean(ttf_vals))
    idx      = np.arange(len(ttf_vals))

    # Gradient colour by value: high TTF = green, low = red
    max_v = float(np.nanmax(ttf_vals)) if len(ttf_vals) else 1

    fig.add_trace(go.Scatter(
        x=idx, y=ttf_vals,
        mode="lines+markers",
        line=dict(color=P["purple"], width=2.5, shape="spline", smoothing=0.6),
        marker=dict(
            color=ttf_vals,
            colorscale=[[0, P["red"]], [0.5, P["yellow"]], [1, P["green"]]],
            cmin=0, cmax=max_v,
            size=9,
            line=dict(color="white", width=1.5),
            showscale=True,
            colorbar=dict(title="Days", thickness=12, len=0.6, x=1.02,
                          tickfont=dict(size=9)),
        ),
        fill="tozeroy",
        fillcolor="rgba(171,140,228,0.07)",
        hovertemplate=(
            "<b>Incident #%{x}</b><br>"
            "Date: <b>%{customdata[0]}</b><br>"
            "Mode: <b>%{customdata[1]}</b><br>"
            "TTF: <b>%{y} days</b>"
            "<extra></extra>"
        ),
        customdata=list(zip(
            [str(d)[:10] for d in dates],
            [str(m) for m in modes],
        )),
        name="TTF",
    ))
    # MTTF reference
    fig.add_hline(y=mttf, line=dict(color=P["red"], width=1.8, dash="dash"),
                  annotation_text=f"MTTF {mttf:.0f}d",
                  annotation_font=dict(color=P["red"], size=10),
                  annotation_position="top right")

    layout = _base_layout(
        xaxis_title="Sequential Incident No.",
        yaxis_title="Days Since Previous Failure",
        showlegend=False,
        uirevision="chron",
    )
    fig.update_layout(**layout)
    return fig


def build_heatmap(loc_data):
    pm = loc_data.groupby(["Failure_Point","Failure_Mode"]).size().reset_index(name="Count")
    fig = go.Figure()
    if pm.empty:
        return fig

    hd = pm.pivot(index="Failure_Mode", columns="Failure_Point", values="Count").fillna(0)

    fig.add_trace(go.Heatmap(
        z=hd.values, x=list(hd.columns), y=list(hd.index),
        colorscale=[
            [0.0, "rgba(240,235,255,0.3)"],
            [0.3, "rgba(171,140,228,0.5)"],
            [0.7, "rgba(102,16,242,0.7)"],
            [1.0, "rgba(60,10,150,0.95)"],
        ],
        showscale=True,
        colorbar=dict(title="Count", thickness=14,
                      tickfont=dict(size=9)),
        hovertemplate=(
            "Point: <b>%{x}</b><br>"
            "Mode: <b>%{y}</b><br>"
            "Occurrences: <b>%{z}</b>"
            "<extra></extra>"
        ),
        xgap=2, ygap=2,
        zsmooth=False,
    ))
    layout = _base_layout(
        xaxis_title="Failure Point (Component)",
        yaxis_title="Failure Mode (Mechanism)",
        xaxis=dict(tickangle=-35, tickfont=dict(size=10),
                   gridcolor=P["grid"], zeroline=False),
        yaxis=dict(tickfont=dict(size=10),
                   gridcolor=P["grid"], zeroline=False),
        uirevision="heatmap",
    )
    fig.update_layout(**layout)
    return fig


def build_growth_pattern(loc_data):
    fig = go.Figure()
    a_ttf = loc_data["TTF_Any_Mode_Days"].dropna().astype(float)
    if len(a_ttf) <= 1:
        return fig

    cum_t = np.cumsum(a_ttf.values)
    cum_f = np.arange(1, len(a_ttf)+1)
    log_t = np.log10(cum_t)
    log_f = np.log10(cum_f)
    slope, intercept, r_value, _, _ = stats.linregress(log_t, log_f)
    log_fit = intercept + slope * log_t

    trend = "Improving ↑" if slope < 1 else ("Worsening ↓" if slope > 1 else "Stable →")
    trend_color = P["green"] if slope < 1 else (P["red"] if slope > 1 else P["yellow"])

    fig.add_trace(go.Scatter(
        x=log_t, y=log_f, mode="markers",
        marker=dict(color=P["green"], size=9,
                    line=dict(color="white", width=1.5),
                    symbol="circle"),
        name="Actual Data",
        hovertemplate=(
            "Log₁₀ time: <b>%{x:.3f}</b><br>"
            "Log₁₀ failures: <b>%{y:.3f}</b><br>"
            "Cum. time: <b>%{customdata[0]:.0f} d</b><br>"
            "Cum. failures: <b>%{customdata[1]}</b>"
            "<extra></extra>"
        ),
        customdata=list(zip(cum_t, cum_f)),
    ))
    fig.add_trace(go.Scatter(
        x=log_t, y=log_fit, mode="lines",
        line=dict(color=trend_color, width=2.5, dash="dash"),
        name=f"Fitted σ={slope:.3f} ({trend})",
        hovertemplate=f"Fitted: <b>%{{y:.3f}}</b><extra></extra>",
    ))
    mid = len(log_t)//2
    fig.add_annotation(
        x=log_t[mid], y=log_fit[mid],
        text=f"<b>σ = {slope:.3f}</b><br>R² = {r_value**2:.3f}<br>{trend}",
        showarrow=True, arrowhead=2, arrowcolor=trend_color, arrowwidth=1.5,
        font=dict(color=trend_color, size=11, family=P["font"]),
        bgcolor="rgba(255,255,255,0.92)", borderpad=8,
        bordercolor=trend_color, borderwidth=1.5,
    )
    layout = _base_layout(
        xaxis_title="Log₁₀ (Cumulative Operating Time, days)",
        yaxis_title="Log₁₀ (Cumulative Failures)",
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font_size=11),
        uirevision="duane",
    )
    fig.update_layout(**layout)
    return fig


def build_failure_concentration(loc_data, top_n):
    fp = loc_data["Failure_Mode"].value_counts().head(top_n)
    fig = go.Figure()
    if fp.empty:
        return fig

    n = len(fp)
    colors = [f"rgba(0,162,174,{0.35+0.65*(n-i)/n})" for i in range(n)]

    fig.add_trace(go.Bar(
        x=fp.index, y=fp.values,
        marker=dict(color=colors, line=dict(color=P["teal"], width=0.5)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Count: <b>%{y}</b><br>"
            "Share: <b>%{customdata:.1f}%</b>"
            "<extra></extra>"
        ),
        customdata=(fp.values / fp.sum() * 100),
    ))
    layout = _base_layout(
        xaxis_title="Failure Mode",
        yaxis_title="Frequency",
        uirevision="conc",
    )
    fig.update_layout(**layout)
    return fig


def build_mttf_mttr_profile(loc_data):
    sd = loc_data.groupby("Failure_Mode").agg(
        {"TTF_Any_Mode_Days":"mean","TTR_Mode_Days":"mean"}
    ).fillna(0)
    fig = go.Figure()
    if sd.empty:
        return fig

    mttf_vals = sd["TTF_Any_Mode_Days"].values
    mttr_hrs  = sd["TTR_Mode_Days"].values * 24

    fig.add_trace(go.Bar(
        x=sd.index, y=mttf_vals, name="MTTF (Days)",
        marker=dict(
            color=mttf_vals,
            colorscale=[[0,P["blue"]], [1,"rgba(3,169,243,0.9)"]],
            cmin=0, cmax=float(max(mttf_vals)) if len(mttf_vals) else 1,
            line=dict(color=P["blue"], width=0.5),
            opacity=0.85,
        ),
        offsetgroup=0, yaxis="y",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "MTTF: <b>%{y:.1f} days</b>"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=sd.index, y=mttr_hrs, name="MTTR (Hours)",
        marker=dict(
            color=mttr_hrs,
            colorscale=[[0,"rgba(230,126,34,0.5)"], [1,P["orange"]]],
            cmin=0, cmax=float(max(mttr_hrs)) if len(mttr_hrs) else 1,
            line=dict(color=P["orange"], width=0.5),
            opacity=0.85,
        ),
        offsetgroup=1, yaxis="y2",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "MTTR: <b>%{y:.1f} hrs</b>"
            "<extra></extra>"
        ),
    ))
    layout = _base_layout(
        xaxis_title="Failure Mode",
        yaxis=dict(title="MTTF (Days)", title_font=dict(color=P["blue"]),
                   tickfont=dict(color=P["blue"]), side="left",
                   gridcolor=P["grid"], zeroline=False),
        yaxis2=dict(title="MTTR (Hours)", title_font=dict(color=P["orange"]),
                    tickfont=dict(color=P["orange"]),
                    overlaying="y", side="right", showgrid=False, zeroline=False),
        barmode="group", bargap=0.18, bargroupgap=0.08,
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font_size=11),
        uirevision="mttf_mttr",
    )
    fig.update_layout(**layout)
    return fig


# ─── Shared chart config (enables all interactive toolbar buttons) ─────────────
CHART_CFG = dict(
    use_container_width=True,
    config=dict(
        displayModeBar=True,
        displaylogo=False,
        modeBarButtonsToRemove=["lasso2d","select2d","autoScale2d"],
        toImageButtonOptions=dict(format="png", scale=2),
        scrollZoom=True,
    ),
)


# ─── HTML helpers ──────────────────────────────────────────────────────────────
def kpi_card(title, value, color_class):
    return (f'<div class="ela-kpi-card {color_class}">'
            f'<div class="card-title-text">{title}</div>'
            f'<div class="card-value-text">{value}</div></div>')

def kpi_row(cards):
    return f'<div class="ela-card-row">{"".join(cards)}</div>'

def chart_title(text):
    st.markdown(f'<div class="plot-tile-title">{text}</div>', unsafe_allow_html=True)

FOOTER_HTML = """
<div class="footer">
  <span>Developed by</span>&nbsp;
  <a href="https://www.linkedin.com/in/pheteho-reymond-m-4669b1204/" target="_blank" rel="noopener noreferrer">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="#0077B5"
         style="vertical-align:middle;margin-right:4px;">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136
               2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37
               4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063
               2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064
               2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0
               1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24
               22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    Pheteho Reymond Mahomane
  </a>
</div>
"""


# ─── Session state ─────────────────────────────────────────────────────────────
for k, v in {"df_processed":None,"all_locations":[],"min_date":None,
              "max_date":None,"active_view":"system"}.items():
    if k not in st.session_state: st.session_state[k]=v


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="ela-header-bar"><h1>Sentinel</h1></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD PHASE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.df_processed is None:
    st.markdown("""
    <div class="upload-card">
      <div class="upload-title">📂 Upload your Work Orders File</div>
      <div class="upload-info">
        Upload a raw CSV or Excel work-orders export. The app will automatically
        extract <strong>Failure Mode</strong> and <strong>Failure Point</strong>
        from the description column using NLP, then compute TTF / TTR reliability metrics.<br><br>
        Required columns: <code>Basic_Start_Date</code> · <code>Basic_Finish_Date</code> ·
        <code>Functional_Location</code> · <code>Equipment</code> · <code>Actual_Hours</code> ·
        a description column (<code>Description (Short Text)</code>, <code>KTEXT</code>,
        <code>Description</code>, <code>Short Text</code>, or <code>TEXT</code>).
      </div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv","xlsx","xls"],
                                 help="Raw SAP / CMMS work-order export")

    if uploaded is not None:
        try:
            fb = io.BytesIO(uploaded.read())
            df_raw = pd.read_csv(fb) if uploaded.name.lower().endswith(".csv") else pd.read_excel(fb)
        except Exception as e:
            st.error(f"❌ Could not read file: {e}"); st.stop()

        desc_candidates = ["Description (Short Text)","KTEXT","Description",
                           "Short Text","Short Description","TEXT"]
        has_desc = any(c in df_raw.columns for c in desc_candidates)
        missing  = [c for c in REQUIRED_RAW_COLS if c not in df_raw.columns]

        if missing or not has_desc:
            msgs = []
            if missing: msgs.append(f"Missing: **{', '.join(missing)}**")
            if not has_desc: msgs.append(f"No description column found — expected one of: {', '.join(desc_candidates)}")
            st.error("❌ " + " | ".join(msgs))
            st.write("**Columns found:**", list(df_raw.columns)); st.stop()

        st.markdown(f'<div class="alert-success">✅ <b>{uploaded.name}</b> — '
                    f'{df_raw.shape[0]:,} rows. Running NLP extraction…</div>',
                    unsafe_allow_html=True)
        bar = st.progress(0, text="Extracting…")

        try:
            df_proc = extract(df_raw, FAILURE_MODES, FAILURE_POINTS, REPLACEMENTS,
                              progress_cb=lambda c,t: bar.progress(c/t, text=f"Row {c}/{t}…"))
        except ValueError as e:
            st.error(f"❌ {e}"); st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}"); st.stop()

        bar.progress(1.0, text="Done!")
        if df_proc.empty:
            st.warning("⚠️ Dataset is empty after processing — check date/location columns."); st.stop()

        st.session_state.df_processed  = df_proc
        st.session_state.all_locations = sorted(df_proc["Functional_Location"].unique())
        st.session_state.min_date      = df_proc["Basic_Start_Date"].min().date()
        st.session_state.max_date      = df_proc["Basic_Start_Date"].max().date()
        st.session_state.active_view   = "system"
        pf = (df_proc["Failure_Point"]!="").sum()
        mf = (df_proc["Failure_Mode"]!="").sum()
        st.markdown(f'<div class="alert-success">✅ Done — <b>{len(df_proc):,}</b> records. '
                    f'Points: <b>{pf}</b> | Modes: <b>{mf}</b>.</div>', unsafe_allow_html=True)
        st.rerun()

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PHASE
# ══════════════════════════════════════════════════════════════════════════════
df_3 = st.session_state.df_processed
all_locations = st.session_state.all_locations
min_date, max_date = st.session_state.min_date, st.session_state.max_date

# ── Sidebar: navigation only ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-menu-header">DASHBOARD MENU</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Options:</div>', unsafe_allow_html=True)

    for label, view in [("⌕  Global Insights","system"),
                         ("⌕  Asset Insights","node"),
                         ("⭍  Predictive Simulations","predictive")]:
        btype = "primary" if st.session_state.active_view==view else "secondary"
        if st.button(label, use_container_width=True, type=btype):
            st.session_state.active_view=view; st.rerun()

    st.markdown('<div class="sidebar-section-label">Data:</div>', unsafe_allow_html=True)
    if st.button("＋  Import New File", use_container_width=True):
        for k in ["df_processed","all_locations","min_date","max_date"]:
            st.session_state[k] = None if k=="df_processed" else ([] if k=="all_locations" else None)
        st.session_state.active_view="system"; st.rerun()

    st.markdown('<div class="sidebar-hint">Filters update charts automatically</div>',
                unsafe_allow_html=True)

# ── Inline filter bar — always visible, no scrolling needed ──────────────────
st.markdown('<div class="filter-bar">', unsafe_allow_html=True)

_view = st.session_state.active_view
_is_node = _view in ("node","predictive")
_is_pred = _view == "predictive"

# Number of columns depends on active view
if _is_pred:
    _cols = st.columns([1.4, 1.4, 0.9, 2.2, 0.85, 1])
elif _is_node:
    _cols = st.columns([1.4, 1.4, 0.9, 2.2, 1])
else:
    _cols = st.columns([1.6, 1.6, 1.2, 1])

with _cols[0]:
    date_from = st.date_input("From Date", value=min_date,
                               min_value=min_date, max_value=max_date,
                               label_visibility="visible")
with _cols[1]:
    date_to = st.date_input("To Date", value=max_date,
                             min_value=min_date, max_value=max_date,
                             label_visibility="visible")
with _cols[2]:
    top_n = st.selectbox("Limit", [10,20,50], index=0, label_visibility="visible")

selected_location = None
test_days = 7

if _is_node:
    with _cols[3]:
        srch = st.text_input("Search / Select Node", placeholder="Type to filter…",
                             label_visibility="visible")
        flocs = [l for l in all_locations if srch.lower() in l.lower()] if srch else all_locations
    with _cols[4]:
        # push selectbox to bottom-align with text input
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        selected_location = st.selectbox("Node", options=flocs, label_visibility="collapsed")

if _is_pred:
    with _cols[5]:
        test_days = st.number_input("Target TTF (days)", min_value=1, max_value=3650,
                                    value=7, step=1, label_visibility="visible")

if not _is_node:
    # fill remaining col with a hint
    with _cols[3]:
        st.markdown("<div style='font-size:10px;color:#adb5bd;padding-top:22px'>"
                    "All charts update on filter change</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Apply date filter ──────────────────────────────────────────────────────────
mask = ((df_3["Basic_Start_Date"].dt.date>=date_from) &
        (df_3["Basic_Start_Date"].dt.date<=date_to))
filtered_df = df_3[mask]


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: GLOBAL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_view=="system":
    st.markdown('<div class="ela-section-title">Global Statistical Overview</div>',
                unsafe_allow_html=True)
    if filtered_df.empty:
        st.markdown('<div class="alert-warning">No entries for the selected date range.</div>',
                    unsafe_allow_html=True)
    else:
        total_failures = len(filtered_df)
        mean_mttf = filtered_df["TTF_Any_Mode_Days"].mean()
        mttf_str  = f"{mean_mttf:.1f} Days" if not np.isnan(float(mean_mttf)) else "N/A"
        tw  = max(1,(date_to-date_from).days)
        ua  = filtered_df["Functional_Location"].nunique()
        cap = tw*ua
        dt  = filtered_df["TTR_Mode_Days"].sum()
        avail = max(0.0,min(100.0,((cap-dt)/cap)*100)) if cap>0 else 100.0

        st.markdown(kpi_row([
            kpi_card("Global MTTF",                      mttf_str,           "card-blue"),
            kpi_card("Global Availability",              f"{avail:.2f}%",    "card-green"),
            kpi_card("Total Analysed Failure Incidents", str(total_failures), "card-red"),
        ]), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            chart_title("Failure Concentration Matrix")
            st.plotly_chart(build_pareto_chart(filtered_df,top_n), key="g_pareto", **CHART_CFG)
        with c2:
            chart_title(f"Top {top_n} Failure Modes")
            st.plotly_chart(build_failure_modes_chart(filtered_df,top_n), key="g_fmodes", **CHART_CFG)

        c3, c4 = st.columns(2)
        with c3:
            chart_title("Distribution of Days To Failures")
            st.plotly_chart(build_ttf_histogram(filtered_df), key="g_ttf", **CHART_CFG)
        with c4:
            chart_title("Equipment Failure Frequency")
            st.plotly_chart(build_equipment_frequency(filtered_df,top_n), key="g_equip", **CHART_CFG)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: ASSET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_view=="node":
    if not selected_location or filtered_df.empty:
        st.markdown('<div class="alert-info">ℹ Select a node from the sidebar.</div>',
                    unsafe_allow_html=True)
    else:
        loc_data = filtered_df[filtered_df["Functional_Location"]==selected_location]
        if loc_data.empty:
            st.markdown(f'<div class="alert-warning">No data for: {selected_location}</div>',
                        unsafe_allow_html=True)
        else:
            asset_fails = len(loc_data)
            a_mttf = loc_data["TTF_Any_Mode_Days"].mean()
            a_mttr = loc_data["TTR_Mode_Days"].mean()
            tw = max(1,(date_to-date_from).days)
            a_avail = max(0.0,min(100.0,((tw-loc_data["TTR_Mode_Days"].sum())/tw)*100))

            st.markdown(f'<div class="ela-section-title">Active Node: {selected_location} Metrics</div>',
                        unsafe_allow_html=True)
            st.markdown(kpi_row([
                kpi_card(f"Asset MTTF ({asset_fails} Fails)",
                         f"{float(a_mttf):.1f} Days" if not np.isnan(float(a_mttf)) else "N/A", "card-purple"),
                kpi_card("Asset Availability", f"{a_avail:.2f}%", "card-green"),
                kpi_card("Asset MTTR",
                         f"{float(a_mttr)*24:.1f} Hours" if not np.isnan(float(a_mttr)) else "N/A", "card-orange"),
            ]), unsafe_allow_html=True)

            c1,c2 = st.columns(2)
            with c1:
                chart_title("Failure Mode Share %")
                st.plotly_chart(build_asset_pie(loc_data,top_n), key="n_pie", **CHART_CFG)
            with c2:
                chart_title("Chronological TTF Performance Trend")
                st.plotly_chart(build_chronological_ttf(loc_data), key="n_chron", **CHART_CFG)

            c3,c4 = st.columns(2)
            with c3:
                chart_title("Cross-Section Count Matrix Map")
                st.plotly_chart(build_heatmap(loc_data), key="n_heat", **CHART_CFG)
            with c4:
                chart_title("Distribution of Days TO Failures (TTF)")
                st.plotly_chart(build_ttf_histogram(loc_data), key="n_ttf", **CHART_CFG)

            c5,c6 = st.columns(2)
            with c5:
                chart_title("Duane Plot – Reliability Growth Model")
                st.plotly_chart(build_growth_pattern(loc_data), key="n_duane", **CHART_CFG)
            with c6:
                chart_title("Failure Mode Concentration")
                st.plotly_chart(build_failure_concentration(loc_data,top_n), key="n_conc", **CHART_CFG)

            chart_title("MTTF vs MTTR Profile")
            st.plotly_chart(build_mttf_mttr_profile(loc_data), key="n_profile", **CHART_CFG)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: PREDICTIVE SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_view=="predictive":
    if not selected_location or filtered_df.empty:
        st.markdown('<div class="alert-info">ℹ Select a node from the sidebar.</div>',
                    unsafe_allow_html=True)
    else:
        loc_data = filtered_df[filtered_df["Functional_Location"]==selected_location]
        a_ttf    = loc_data["TTF_Any_Mode_Days"].dropna().astype(float)

        if loc_data.empty or len(a_ttf)<=2:
            st.markdown(
                f'<div class="alert-warning">Insufficient data for Weibull analysis on: '
                f'{selected_location}. Need more failure events.</div>', unsafe_allow_html=True)
        else:
            asset_fails = len(loc_data)
            try:
                shape,_,scale = stats.weibull_min.fit(a_ttf,floc=0)
                r_t = stats.weibull_min.sf(test_days,shape,loc=0,scale=scale)
                f_t = 1-r_t
            except Exception:
                shape=r_t=f_t=None

            st.markdown(f'<div class="ela-section-title">Predictive Analysis: {selected_location}</div>',
                        unsafe_allow_html=True)
            st.markdown(kpi_row([
                kpi_card("Target Horizon",          f"{test_days} Days",               "card-teal"),
                kpi_card("Historical Failures",     f"{asset_fails} Events",           "card-purple"),
                kpi_card("Failure Probability F(t)",
                         f"{f_t*100:.1f}%" if f_t is not None else "N/A",             "card-orange"),
            ]), unsafe_allow_html=True)

            chart_title(f"Asset Weibull Profile: {selected_location}")
            st.plotly_chart(build_asset_weibull(loc_data,test_days), key="p_weibull", **CHART_CFG)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
