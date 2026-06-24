"""
Sentinel – Streamlit Dashboard
Full port of the Jupyter / ipywidgets version, including the `extract` NLP pipeline.
"""

import io
import re
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from rapidfuzz import fuzz, process
from spacy.matcher import PhraseMatcher

# ── spaCy: prefer en_core_web_sm, fall back to blank English ─────────────────
try:
    import spacy
    try:
        _nlp_global = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except OSError:
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _nlp_global = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        except Exception:
            _nlp_global = spacy.blank("en")
except Exception:
    _nlp_global = None

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
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
}
.card-blue   { border-left-color: #03a9f3 !important; }
.card-green  { border-left-color: #00c292 !important; }
.card-red    { border-left-color: #e46a76 !important; }
.card-purple { border-left-color: #ab8ce4 !important; }
.card-orange { border-left-color: #e67e22 !important; }
.card-teal   { border-left-color: #20c997 !important; }
.card-title-text  { font-size: 11px; color: #868e96; text-transform: uppercase; font-weight: 600; letter-spacing: 0.3px; }
.card-value-text  { font-size: 22px; font-weight: 700; color: #455a64; margin-top: 5px; }

.ela-section-title {
    font-size: 14px; font-weight: 600; color: #455a64; margin: 15px 0 10px 0;
    text-transform: uppercase; border-left: 3px solid #6610f2; padding-left: 8px;
    font-family: 'Ubuntu', sans-serif;
}
.plot-tile-title {
    font-size: 13px; font-weight: 600; color: #495057;
    margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 2px solid #f1f3f5; text-transform: uppercase; letter-spacing: 0.5px;
}
.alert-warning { color: #856404; background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }
.alert-info    { color: #0c5460; background: #d1ecf1; padding: 10px; border-radius: 4px; margin: 10px 0; }
.alert-success { color: #155724; background: #d4edda; padding: 10px; border-radius: 4px; margin: 10px 0; }

.upload-card {
    background: #fff; border-radius: 12px; padding: 30px; margin: 20px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}
.upload-title { font-size: 18px; font-weight: 600; color: #455a64; margin-bottom: 15px; }
.upload-info  { font-size: 13px; color: #868e96; margin-bottom: 15px; }

[data-testid="stSidebar"] { background: #ffffff; }
.sidebar-menu-header {
    background: #455a64; color: white; text-align: center; font-weight: bold;
    font-size: 16px; padding: 10px 8px; border-radius: 4px 4px 0 0; margin-bottom: 10px;
}
.sidebar-section-label { font-size: 10px; color: #6c757d; padding: 8px; margin-top: 15px; }
.sidebar-hint { font-size: 10px; color: #6c757d; text-align: center; background: #f8f9fa; border-radius: 4px; padding: 8px; margin-top: 15px; }

.footer {
    font-family: 'Ubuntu', sans-serif; font-size: 10px; color: #555;
    text-align: center; padding: 20px; margin-top: 30px;
    background: #f8f9fa; border-top: 1px solid #e9ecef;
}
.footer a { color: #0077B5; text-decoration: none; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── NLP / extraction constants ───────────────────────────────────────────────
FAILURE_MODES = [
    "worn", "broken", "leaking", "leak", "cracked", "corroded", "rusted", "vibration",
    "loose", "jammed", "blocked", "clogged", "overheated", "burnt", "seized", "misaligned",
    "damaged", "failed", "noisy", "bent", "frayed", "torn", "missing", "dirty",
    "eroded", "cavitated", "pitted", "spalled", "galled", "scored", "scratched",
    "fatigued", "fractured", "sheared", "twisted", "collapsed", "buckled", "distorted",
    "warped", "swollen", "shrunk", "expanded", "contracted", "melted", "charred",
    "blistered", "delaminated", "debonded", "fretted", "brinelled", "contaminated",
    "degraded", "deteriorated", "oxidized", "sulfidated", "carburized", "nitrided",
    "embrittled", "softened", "hardened", "work hardened", "creep", "stress rupture",
    "thermal fatigue", "thermal shock", "corrosion fatigue", "stress corrosion cracking",
    "hydrogen embrittlement", "hydrogen blistering", "hydrogen induced cracking",
    "sulfide stress cracking", "chloride stress corrosion cracking", "caustic cracking",
    "amine cracking", "ammonia stress corrosion cracking", "liquid metal embrittlement",
    "intergranular corrosion", "transgranular corrosion", "pitting corrosion",
    "crevice corrosion", "galvanic corrosion", "uniform corrosion", "localized corrosion",
    "microbiologically influenced corrosion", "flow accelerated corrosion",
    "erosion corrosion", "impingement corrosion", "cavitation erosion", "fretting corrosion",
    "high temperature corrosion", "hot corrosion", "fouled", "scaled", "coked",
    "plugged", "obstructed", "restricted", "starved", "flooded", "dry run",
    "dead headed", "overpressurized", "underpressurized", "overloaded", "underloaded",
    "unbalanced", "eccentric", "runout", "out of round", "out of flat", "out of square",
    "out of parallel", "out of tolerance", "excessive clearance", "insufficient clearance",
    "excessive backlash", "insufficient backlash", "excessive preload", "insufficient preload",
    "excessive tension", "insufficient tension", "over torqued", "under torqued",
    "cross threaded", "stripped", "galled threads", "stuck", "frozen", "bound",
    "sticking", "binding", "hunting", "oscillating", "cycling", "short cycling",
    "chattering", "hammering", "water hammer", "steam hammer", "surge", "pulsation",
    "fluctuating", "unstable", "intermittent", "erratic", "sporadic", "drifting",
    "biased", "offset", "inaccurate", "imprecise", "nonlinear", "hysteresis",
    "deadband", "stiction", "backlash", "windup", "saturation", "cutoff",
    "slew rate limiting", "overshoot", "undershoot", "ringing", "settling time",
    "response time", "lag", "delay", "timeout", "no response", "false reading",
    "false trip", "nuisance trip", "spurious trip", "fail to trip", "fail to start",
    "fail to stop", "fail to open", "fail to close", "fail to energize", "fail to deenergize",
    "short circuit", "open circuit", "ground fault", "phase fault", "earth fault",
    "arc fault", "flashover", "tracking", "partial discharge", "corona", "arcing",
    "sparking", "overcurrent", "undercurrent", "overvoltage", "undervoltage",
    "overfrequency", "underfrequency", "phase imbalance", "phase loss", "phase reversal",
]

FAILURE_POINTS = [
    "bearing", "seal", "pump", "motor", "valve", "gear", "roller", "roll", "dryer roll",
    "conveyor", "belt", "chain", "cylinder", "piston", "shaft", "impeller", "filter",
    "screen", "blade", "knife", "screw", "auger", "fan", "blower", "compressor",
    "sensor", "switch", "relay", "pipe", "hose", "coupling", "bearing housing",
    "transformer", "generator", "turbine", "boiler", "heat exchanger", "condenser",
    "cooling tower", "pulper", "refiner", "headbox", "wire", "felt", "press roll",
    "calendar", "reel", "winder", "digester", "evaporator", "recovery boiler",
    "lime kiln", "chipper", "bark drum", "centrifuge", "hydrocyclone", "flotation cell",
    "thickener", "agitator", "mixer", "crusher", "mill", "kiln", "dust collector",
    "cyclone", "hopper", "chute", "feeder", "vibrating screen", "jaw crusher",
    "cone crusher", "ball mill", "sag mill", "slurry pump", "dragline", "shovel",
    "haul truck", "stacker", "reclaimer", "separator", "scrubber", "desalter",
    "fractionator", "reformer", "cracker", "coker", "hydrotreater", "flare",
    "pig launcher", "pig receiver", "christmas tree", "blowout preventer", "riser",
    "umbilical", "manifold", "actuator", "positioner", "transmitter", "controller",
    "analyser", "gauge", "indicator", "recorder", "solenoid", "circuit breaker",
    "fuse", "contactor", "starter", "vfd", "inverter", "rectifier", "battery",
    "ups", "busbar", "insulator", "conductor", "cable", "tray", "conduit",
    "junction box", "terminal block", "plc", "dcs", "scada", "hmi", "rtu",
    "thermocouple", "rtd", "pressure transmitter", "flow meter", "level transmitter",
    "ph meter", "conductivity meter", "vibration probe", "proximity probe",
    "limit switch", "pressure switch", "control valve", "float switch", "temperature switch",
]

REPLACEMENTS = {
    r'\bbrng\b': 'bearing', r'\bbearng\b': 'bearing', r'\bbrg\b': 'bearing',
    r'\bbering\b': 'bearing', r'\bbearingg\b': 'bearing', r'\bbeering\b': 'bearing',
    r'\bpnmp\b': 'pump', r'\bpummp\b': 'pump', r'\bpumpe\b': 'pump', r'\bpomp\b': 'pump',
    r'\bmtor\b': 'motor', r'\bmtr\b': 'motor', r'\bmotorr\b': 'motor',
    r'\bmotar\b': 'motor', r'\bmoter\b': 'motor',
    r'\bvalv\b': 'valve', r'\bvalvve\b': 'valve', r'\bvalv e\b': 'valve',
    r'\bleek\b': 'leak', r'\bleake\b': 'leak', r'\bleeking\b': 'leaking',
    r'\bcrkd\b': 'cracked', r'\bcrackked\b': 'cracked', r'\bcraked\b': 'cracked',
    r'\brustd\b': 'rusted', r'\brusty\b': 'rusted', r'\brustted\b': 'rusted',
    r'\bcorrodid\b': 'corroded', r'\bcoroded\b': 'corroded',
    r'\bowrhot\b': 'overheated', r'\boverheatt\b': 'overheated',
    r'\boverheted\b': 'overheated', r'\boverhete\b': 'overheated',
    r'\bsezd\b': 'seized', r'\bseised\b': 'seized', r'\bsezed\b': 'seized',
    r'\bblwr\b': 'blower', r'\bblowerr\b': 'blower', r'\bblowr\b': 'blower', r'\bbloer\b': 'blower',
    r'\bcmpresr\b': 'compressor', r'\bcomprssr\b': 'compressor',
    r'\bcompresor\b': 'compressor', r'\bcompressorr\b': 'compressor',
}

FUZZY_THRESHOLD = 85


# ─── Extract pipeline ─────────────────────────────────────────────────────────
def extract(df, failure_modes, failure_points, replacements, progress_cb=None):
    """
    Full two-step extract pipeline matching the Jupyter version exactly.
    progress_cb(current, total) is called after each row if provided.
    """
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except OSError:
        nlp = spacy.blank("en")

    points_lower = [p.lower() for p in failure_points]
    modes_lower  = [m.lower() for m in failure_modes]

    matcher_points = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher_modes  = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher_points.add("FAILURE_POINTS", [nlp.make_doc(t) for t in failure_points])
    matcher_modes.add("FAILURE_MODES",  [nlp.make_doc(t) for t in failure_modes])

    def preprocess_text(text):
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^\w\s\-/]", " ", text)
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return " ".join(text.split())

    def get_best_fuzzy_match(text, candidates, threshold=FUZZY_THRESHOLD):
        if not text or not candidates:
            return None, 0
        words = text.split()
        phrases = [text]
        for n in [2, 3]:
            phrases.extend([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])
        best_match, best_score = None, 0
        for phrase in set(phrases):
            if len(phrase) < 3:
                continue
            match, score, _ = process.extractOne(phrase, candidates, scorer=fuzz.token_sort_ratio)
            if score > best_score and score >= threshold:
                best_score, best_match = score, match
        return best_match, best_score

    def extract_with_fuzzy(text, keywords, ktype):
        doc = nlp(text)
        exact = (
            [doc[s:e].text for _, s, e in matcher_points(doc)] if ktype == "point"
            else [doc[s:e].text for _, s, e in matcher_modes(doc)]
        )
        remaining = text
        for m in exact:
            remaining = remaining.replace(m, "")
        fuzzy_match, _ = get_best_fuzzy_match(remaining, keywords)
        matches = exact + ([fuzzy_match] if fuzzy_match else [])
        seen, unique = set(), []
        for m in matches:
            if m.lower() not in seen:
                seen.add(m.lower()); unique.append(m)
        return unique

    def select_best(matches, text, keywords):
        if not matches:
            return np.nan
        if len(matches) == 1:
            return matches[0]
        text_lower = text.lower()
        scored = []
        for match in matches:
            ml = match.lower()
            pos = text_lower.find(ml)
            score = max(0, (100 - pos) / 10) if pos >= 0 else 0
            score += len(match) * 2
            if re.search(rf"\b{re.escape(ml)}\b", text_lower):
                score += 20
            if ml in keywords:
                score += 30
            score += min(text_lower.count(ml) * 10, 30)
            for ctx in ["failed","failure","broken","damaged","issue","problem"]:
                if ctx in text_lower and pos >= 0 and abs(pos - text_lower.find(ctx)) < 50:
                    score += 25; break
            scored.append((match, score))
        return max(scored, key=lambda x: x[1])[0]

    def extract_failure_info(short_text):
        if not short_text or pd.isna(short_text):
            return "", ""
        processed = preprocess_text(short_text)
        if not processed:
            return "", ""
        pt_candidates = extract_with_fuzzy(processed, points_lower, "point")
        md_candidates = extract_with_fuzzy(processed, modes_lower,  "mode")
        return (
            select_best(pt_candidates, processed, points_lower),
            select_best(md_candidates, processed, modes_lower),
        )

    # ── Identify columns ──────────────────────────────────────────────────────
    desc_col = next(
        (c for c in ["Description (Short Text)", "KTEXT", "Description",
                     "Short Text", "Short Description", "TEXT"] if c in df.columns),
        None,
    )
    if not desc_col:
        raise ValueError(
            f"Could not find a description column. Available columns: {list(df.columns)}"
        )

    order_col = next(
        (c for c in ["Order", "AUFNR", "Order Number", "ORDER"] if c in df.columns),
        None,
    )

    # ── Step 1: NLP extraction ────────────────────────────────────────────────
    results = []
    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        text = str(row[desc_col]) if not pd.isna(row[desc_col]) else ""
        try:
            fp, fm = extract_failure_info(text)
        except Exception as e:
            fp, fm = "ERROR", str(e)[:50]
        results.append({
            "Order": row[order_col] if order_col and not pd.isna(row[order_col]) else f"ROW_{idx}",
            "Failure_Point": fp,
            "Failure_Mode":  fm,
        })
        if progress_cb:
            progress_cb(i + 1, total)

    new_df = pd.DataFrame(results)
    df_1 = df.copy()
    df_1["Failure_Point"] = new_df["Failure_Point"].values
    df_1["Failure_Mode"]  = new_df["Failure_Mode"].values

    # ── Step 2: Reliability calculations ─────────────────────────────────────
    df_2 = df_1.copy()
    df_2["Basic_Start_Date"] = pd.to_datetime(df_2["Basic_Start_Date"])
    df_2 = df_2.dropna(subset=["Failure_Mode"])
    df_2 = df_2.sort_values(
        ["Functional_Location", "Basic_Start_Date"], ascending=[True, True]
    ).reset_index(drop=True)

    df_2["TTR_Mode_Days"]       = (df_2["Actual_Hours"] / 24).round(1)
    df_2["TTF_Any_Mode_Date"]   = df_2.groupby("Functional_Location")["Basic_Start_Date"].shift(-1)
    df_2["TTF_Any_Mode_Days"]   = (df_2["TTF_Any_Mode_Date"] - df_2["Basic_Start_Date"]).dt.days

    df_2 = df_2.dropna(subset=["TTF_Any_Mode_Days"])
    df_2 = df_2[df_2["TTF_Any_Mode_Days"] != 0]

    df_2 = df_2.sort_values(
        ["Functional_Location", "Failure_Mode", "Basic_Start_Date"], ascending=[True, True, True]
    ).reset_index(drop=True)

    df_2["TTF_Same_Mode_Date"] = df_2.groupby(
        ["Functional_Location", "Failure_Mode"])["Basic_Start_Date"].shift(-1)
    df_2["TTF_Same_Mode_Days"] = (df_2["TTF_Same_Mode_Date"] - df_2["Basic_Start_Date"]).dt.days

    df_2 = df_2.dropna(subset=["TTF_Same_Mode_Days"])
    df_2 = df_2[df_2["TTF_Same_Mode_Days"] != 0]

    df_2 = df_2.sort_values(
        ["Functional_Location", "Basic_Start_Date"], ascending=[True, True]
    ).reset_index(drop=True)

    df_2["Previous_Failure_Mode"] = df_2.groupby("Functional_Location")["Failure_Mode"].shift()
    df_2 = df_2.dropna(subset=["Previous_Failure_Mode"])
    df_2["TTF_Any_Mode_Days"] = df_2["TTF_Any_Mode_Days"].fillna(0).astype("Int64")

    return df_2


# ─── Required raw columns ─────────────────────────────────────────────────────
REQUIRED_RAW_COLS = [
    "Basic_Start_Date",
    "Basic_Finish_Date",
    "Functional_Location",
    "Equipment",
    "Actual_Hours",
    # description col is detected automatically
]


# ─── Plot builders (identical logic to Jupyter version) ───────────────────────
def build_pareto_chart(filtered_df, top_n):
    fig = go.Figure()
    counts = filtered_df["Functional_Location"].value_counts().head(top_n)
    if not counts.empty:
        cum_pct = counts.cumsum() / counts.sum() * 100
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, name="Failures", marker_color="#00a2ae"))
        fig.add_trace(go.Scatter(x=counts.index, y=cum_pct, name="Cumulative %",
                                  marker_color="#dc3545", mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        xaxis_title="Functional Location", yaxis_title="Count of Failures",
        yaxis2=dict(title="Cumulative Impact %", overlaying="y", side="right"),
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=False, plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_failure_modes_chart(filtered_df, top_n):
    fig = go.Figure()
    tf = filtered_df["Failure_Mode"].value_counts().head(top_n)
    if not tf.empty:
        fig.add_trace(go.Bar(x=tf.index, y=tf.values, marker_color="#dc3545"))
    fig.update_layout(
        xaxis_title="Failure Mode Classification", yaxis_title="Total Incidents Measured",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=False, plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_ttf_histogram(filtered_df):
    fig = go.Figure()
    ttf = filtered_df["TTF_Any_Mode_Days"].dropna().astype(float)
    if not ttf.empty:
        mttf, std = ttf.mean(), ttf.std()
        hist_vals, _ = np.histogram(ttf, bins=20)
        max_y = max(hist_vals) * 1.15 if len(hist_vals) else 10
        fig.add_trace(go.Histogram(x=ttf, nbinsx=20, marker_color="#20c997"))
        fig.add_trace(go.Scatter(x=[mttf,mttf], y=[0,max_y], mode="lines",
                                  line=dict(color="#dc3545",width=2,dash="dash"),
                                  name=f"MTTF: {mttf:.1f}d"))
        fig.add_trace(go.Scatter(x=[mttf-std,mttf-std], y=[0,max_y*.7], mode="lines",
                                  line=dict(color="#ffc107",width=1.5,dash="dot"),
                                  name=f"-1σ: {mttf-std:.1f}d"))
        fig.add_trace(go.Scatter(x=[mttf+std,mttf+std], y=[0,max_y*.7], mode="lines",
                                  line=dict(color="#ffc107",width=1.5,dash="dot"),
                                  name=f"+1σ: {mttf+std:.1f}d"))
        fig.add_annotation(x=mttf, y=max_y*.95,
                            text=f"MTTF = {mttf:.1f} days<br>σ = {std:.1f} days",
                            showarrow=True, arrowhead=2, arrowcolor="#dc3545",
                            font=dict(color="#dc3545",size=11,family="Open Sans"),
                            bgcolor="rgba(255,255,255,0.85)", borderpad=6)
    fig.update_layout(
        xaxis_title="Time To Failures (Days)", yaxis_title="Frequency",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=False, plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_equipment_frequency(filtered_df, top_n):
    fig = go.Figure()
    te = filtered_df["Equipment"].value_counts().head(top_n)
    if not te.empty:
        fig.add_trace(go.Bar(x=te.values, y=te.index, orientation="h", marker_color="#03a9f3"))
    fig.update_layout(
        xaxis_title="Failure Events Count", yaxis_title="Equipment ID Reference",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=False, plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_asset_pie(loc_data, top_n):
    fig = go.Figure()
    fm = loc_data["Failure_Mode"].value_counts().head(top_n)
    if not fm.empty:
        fig.add_trace(go.Pie(labels=fm.index, values=fm.values, hole=.3, textinfo="percent+label"))
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(t=20,b=40,l=40,r=40), showlegend=False)
    return fig

def build_asset_weibull(loc_data, test_days):
    fig = go.Figure()
    a_ttf = loc_data["TTF_Any_Mode_Days"].dropna().astype(float)
    if len(a_ttf) > 2:
        try:
            shape, _, scale = stats.weibull_min.fit(a_ttf, floc=0)
            r_t = stats.weibull_min.sf(test_days, shape, loc=0, scale=scale)
            f_t = 1 - r_t
            t_range = np.linspace(0, max(a_ttf)*1.2, 100)
            reliability = np.exp(-(t_range/scale)**shape)
            fig.add_trace(go.Scatter(x=t_range, y=reliability, line=dict(color="#ab8ce4",width=2.5)))
            fig.add_trace(go.Scatter(x=[test_days,test_days], y=[0,r_t], mode="lines",
                                      line=dict(color="#dc3545",width=1.5,dash="dash")))
            fig.add_trace(go.Scatter(x=[0,test_days], y=[r_t,r_t], mode="lines",
                                      line=dict(color="#dc3545",width=1.5,dash="dot")))
            fig.add_trace(go.Scatter(
                x=[test_days], y=[r_t], mode="markers+text",
                marker=dict(color="#dc3545",size=10),
                text=[f" β={shape:.2f}<br> R({test_days}d)={r_t*100:.1f}%<br> F({test_days}d)={f_t*100:.1f}%"],
                textposition="top right",
                textfont=dict(color="#dc3545",size=11,family="Open Sans"),
                showlegend=False,
            ))
        except Exception:
            pass
    fig.update_layout(
        xaxis_title="Operating Horizon (Days)", yaxis_title="Reliability Fraction R(t)",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=False, plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_chronological_ttf(loc_data):
    fig = go.Figure()
    if len(loc_data) > 1:
        fig.add_trace(go.Scatter(y=loc_data["TTF_Any_Mode_Days"].values,
                                  mode="lines+markers", line=dict(color="#ab8ce4")))
    fig.update_layout(
        xaxis_title="Sequential Incident Sequence No.",
        yaxis_title="Days Since Previous Failure",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=False, plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_heatmap(loc_data):
    fig = go.Figure()
    pm = loc_data.groupby(["Failure_Point","Failure_Mode"]).size().reset_index(name="Count")
    if not pm.empty:
        hd = pm.pivot(index="Failure_Mode", columns="Failure_Point", values="Count").fillna(0)
        fig.add_trace(go.Heatmap(z=hd.values, x=hd.columns, y=hd.index,
                                  colorscale="Purples", showscale=True,
                                  colorbar=dict(title="Count")))
    fig.update_layout(
        xaxis_title="Identified Failure Mode Location Component",
        yaxis_title="Observed Failure Mechanism / Mode",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
    )
    return fig

def build_growth_pattern(loc_data):
    fig = go.Figure()
    a_ttf = loc_data["TTF_Any_Mode_Days"].dropna().astype(float)
    if len(a_ttf) > 1:
        cum_time     = np.cumsum(a_ttf.values)
        cum_failures = np.arange(1, len(a_ttf)+1)
        log_t = np.log10(cum_time)
        log_f = np.log10(cum_failures)
        slope, intercept, r_value, _, _ = stats.linregress(log_t, log_f)
        log_fit = intercept + slope * log_t
        fig.add_trace(go.Scatter(x=log_t, y=log_f, mode="markers",
                                  marker=dict(color="#00c292",size=8), name="Actual Data"))
        fig.add_trace(go.Scatter(x=log_t, y=log_fit, mode="lines",
                                  line=dict(color="#dc3545",width=2,dash="dash"),
                                  name=f"Fitted (σ={slope:.3f})"))
        mid = len(log_t)//2
        fig.add_annotation(
            x=log_t[mid], y=log_fit[mid],
            text=f"σ (Growth Rate) = {slope:.3f}<br>R² = {r_value**2:.3f}",
            showarrow=True, arrowhead=2, arrowcolor="#dc3545",
            font=dict(color="#dc3545",size=12,family="Open Sans"),
            bgcolor="rgba(255,255,255,0.9)", borderpad=8,
            bordercolor="#dc3545", borderwidth=1,
        )
    fig.update_layout(
        xaxis_title="Log₁₀ (Cumulative Operating Time in Days)",
        yaxis_title="Log₁₀ (Cumulative Number of Failures)",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5,font=dict(size=10)),
        plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_failure_concentration(loc_data, top_n):
    fig = go.Figure()
    fp = loc_data["Failure_Mode"].value_counts().head(top_n)
    if not fp.empty:
        fig.add_trace(go.Bar(x=fp.index, y=fp.values, marker_color="#00a2ae"))
    fig.update_layout(
        xaxis_title="Failure Mode Code Reference",
        yaxis_title="Observed Failure Event Frequency Count",
        template="plotly_white", height=380, margin=dict(t=20,b=40,l=40,r=40),
        showlegend=False, plot_bgcolor="rgba(248,249,250,0.5)",
    )
    return fig

def build_mttf_mttr_profile(loc_data):
    fig = go.Figure()
    sd = loc_data.groupby("Failure_Mode").agg(
        {"TTF_Any_Mode_Days": "mean", "TTR_Mode_Days": "mean"}
    ).fillna(0)
    if not sd.empty:
        fig.add_trace(go.Bar(x=sd.index, y=sd["TTF_Any_Mode_Days"],
                              name="MTTF (Days)", marker_color="#03a9f3",
                              offsetgroup=0, yaxis="y"))
        fig.add_trace(go.Bar(x=sd.index, y=sd["TTR_Mode_Days"]*24,
                              name="MTTR (Hours)", marker_color="#e67e22",
                              offsetgroup=1, yaxis="y2"))
    fig.update_layout(
        xaxis_title="Failure Mode", template="plotly_white",
        height=380, margin=dict(t=20,b=40,l=40,r=40),
        plot_bgcolor="rgba(248,249,250,0.5)", barmode="group",
        yaxis=dict(title="MTTF (Days)", title_font=dict(color="#03a9f3"),
                   tickfont=dict(color="#03a9f3"), side="left"),
        yaxis2=dict(title="MTTR (Hours)", title_font=dict(color="#e67e22"),
                    tickfont=dict(color="#e67e22"), overlaying="y", side="right"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,
                    xanchor="center",x=0.5,font=dict(size=10)),
        bargap=0.15, bargroupgap=0.1,
    )
    return fig


# ─── HTML helpers ──────────────────────────────────────────────────────────────
def kpi_card(title, value, color_class):
    return (f'<div class="ela-kpi-card {color_class}">'
            f'<div class="card-title-text">{title}</div>'
            f'<div class="card-value-text">{value}</div></div>')

def kpi_row(cards):
    return f'<div class="ela-card-row">{"".join(cards)}</div>'

FOOTER_HTML = """
<div class="footer">
  <span>Developed by</span>
  <a href="https://www.linkedin.com/in/pheteho-reymond-m-4669b1204/" target="_blank" rel="noopener noreferrer">
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="#0077B5"
         style="vertical-align:middle;margin-right:4px;">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    Pheteho Reymond Mahomane
  </a>
</div>
"""


# ─── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "df_processed": None,
    "all_locations": [],
    "min_date": None,
    "max_date": None,
    "active_view": "system",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


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
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Raw SAP / CMMS work-order export",
    )

    if uploaded is not None:
        # ── Read file ────────────────────────────────────────────────────────
        try:
            file_bytes = io.BytesIO(uploaded.read())
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(file_bytes)
            else:
                df_raw = pd.read_excel(file_bytes)
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            st.stop()

        # ── Validate mandatory columns ────────────────────────────────────────
        desc_candidates = ["Description (Short Text)", "KTEXT", "Description",
                           "Short Text", "Short Description", "TEXT"]
        has_desc = any(c in df_raw.columns for c in desc_candidates)
        missing = [c for c in REQUIRED_RAW_COLS if c not in df_raw.columns]

        if missing or not has_desc:
            msgs = []
            if missing:
                msgs.append(f"Missing required columns: **{', '.join(missing)}**")
            if not has_desc:
                msgs.append(
                    f"No description column found. Expected one of: "
                    f"{', '.join(desc_candidates)}"
                )
            st.error("❌ " + " | ".join(msgs))
            st.write("**Columns found in your file:**", list(df_raw.columns))
            st.stop()

        st.markdown(
            f'<div class="alert-success">✅ File loaded: <b>{uploaded.name}</b> — '
            f'{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns. '
            f'Running NLP extraction…</div>',
            unsafe_allow_html=True,
        )

        # ── Run extract with live progress bar ────────────────────────────────
        prog_bar  = st.progress(0, text="Extracting failure modes and points…")
        prog_text = st.empty()
        total_rows = len(df_raw)

        def progress_cb(current, total):
            frac = current / total
            prog_bar.progress(frac, text=f"Processing row {current} of {total}…")

        try:
            df_processed = extract(
                df_raw,
                FAILURE_MODES,
                FAILURE_POINTS,
                REPLACEMENTS,
                progress_cb=progress_cb,
            )
        except ValueError as e:
            st.error(f"❌ Extraction error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error during extraction: {e}")
            st.stop()

        prog_bar.progress(1.0, text="Done!")
        prog_text.empty()

        if df_processed.empty:
            st.warning(
                "⚠️ Processing completed but the resulting dataset is empty. "
                "This usually means no rows had both a recognised failure mode and "
                "at least two consecutive failures at the same location. "
                "Check that your date and location columns are correct."
            )
            st.stop()

        # ── Persist to session state ──────────────────────────────────────────
        st.session_state.df_processed  = df_processed
        st.session_state.all_locations = sorted(df_processed["Functional_Location"].unique())
        st.session_state.min_date      = df_processed["Basic_Start_Date"].min().date()
        st.session_state.max_date      = df_processed["Basic_Start_Date"].max().date()
        st.session_state.active_view   = "system"

        points_found = (df_processed["Failure_Point"] != "").sum()
        modes_found  = (df_processed["Failure_Mode"]  != "").sum()
        st.markdown(
            f'<div class="alert-success">'
            f'✅ Extraction complete — <b>{len(df_processed):,}</b> analysable records. '
            f'Failure points: <b>{points_found}</b> | Failure modes: <b>{modes_found}</b>.'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.rerun()

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PHASE
# ══════════════════════════════════════════════════════════════════════════════
df_3          = st.session_state.df_processed
all_locations = st.session_state.all_locations
min_date      = st.session_state.min_date
max_date      = st.session_state.max_date

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-menu-header">DASHBOARD MENU</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Options:</div>', unsafe_allow_html=True)
    for label, view, btn_type in [
        ("⌕  Global Insights",        "system",     "primary"   if st.session_state.active_view == "system"     else "secondary"),
        ("⌕  Asset Insights",         "node",       "primary"   if st.session_state.active_view == "node"       else "secondary"),
        ("⭍  Predictive Simulations", "predictive", "primary"   if st.session_state.active_view == "predictive" else "secondary"),
    ]:
        if st.button(label, use_container_width=True, type=btn_type):
            st.session_state.active_view = view
            st.rerun()

    st.markdown('<div class="sidebar-section-label">Data:</div>', unsafe_allow_html=True)
    if st.button("＋  Import New File", use_container_width=True):
        for k in ["df_processed","all_locations","min_date","max_date"]:
            st.session_state[k] = None if k == "df_processed" else ([] if k == "all_locations" else None)
        st.session_state.active_view = "system"
        st.rerun()

    st.divider()
    st.subheader("Filters")
    date_from = st.date_input("From Date", value=min_date, min_value=min_date, max_value=max_date)
    date_to   = st.date_input("To Date",   value=max_date, min_value=min_date, max_value=max_date)
    top_n     = st.selectbox("Display Limit", [10, 20, 50], index=0)

    selected_location = None
    test_days = 7
    if st.session_state.active_view in ("node", "predictive"):
        st.divider()
        search_text = st.text_input("Search Node", placeholder="Filter nodes…")
        filtered_locs = (
            [l for l in all_locations if search_text.lower() in l.lower()]
            if search_text else all_locations
        )
        selected_location = st.selectbox("Select Node", options=filtered_locs)

    if st.session_state.active_view == "predictive":
        st.divider()
        test_days = st.number_input("Target TTF (Days)", min_value=1, max_value=3650, value=7, step=1)
        st.button("▶  Simulate", use_container_width=True)   # triggers rerun naturally

    st.markdown('<div class="sidebar-hint">Filters update content automatically</div>',
                unsafe_allow_html=True)

# ── Apply date filter ──────────────────────────────────────────────────────────
mask = (
    (df_3["Basic_Start_Date"].dt.date >= date_from) &
    (df_3["Basic_Start_Date"].dt.date <= date_to)
)
filtered_df = df_3[mask]


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: GLOBAL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_view == "system":
    st.markdown('<div class="ela-section-title">Global Statistical Overview</div>',
                unsafe_allow_html=True)

    if filtered_df.empty:
        st.markdown('<div class="alert-warning">Operational slice containing 0 entries for current dates selected.</div>',
                    unsafe_allow_html=True)
    else:
        total_failures = len(filtered_df)
        mean_mttf = filtered_df["TTF_Any_Mode_Days"].mean()
        mttf_str  = f"{mean_mttf:.1f} Days" if not np.isnan(float(mean_mttf)) else "N/A"
        total_window   = max(1, (date_to - date_from).days)
        unique_assets  = filtered_df["Functional_Location"].nunique()
        capacity       = total_window * unique_assets
        downtime       = filtered_df["TTR_Mode_Days"].sum()
        avail = max(0.0, min(100.0, ((capacity - downtime) / capacity) * 100)) if capacity > 0 else 100.0

        st.markdown(kpi_row([
            kpi_card("Global MTTF",                       mttf_str,            "card-blue"),
            kpi_card("Global Availability",               f"{avail:.2f}%",     "card-green"),
            kpi_card("Total Analysed Failure Incidents",  str(total_failures),  "card-red"),
        ]), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="plot-tile-title">Failure Concentration Matrix</div>', unsafe_allow_html=True)
            st.plotly_chart(build_pareto_chart(filtered_df, top_n), use_container_width=True, key="g_pareto")
        with c2:
            st.markdown(f'<div class="plot-tile-title">Top {top_n} Failure Modes</div>', unsafe_allow_html=True)
            st.plotly_chart(build_failure_modes_chart(filtered_df, top_n), use_container_width=True, key="g_fmodes")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="plot-tile-title">Distribution of Days To Failures</div>', unsafe_allow_html=True)
            st.plotly_chart(build_ttf_histogram(filtered_df), use_container_width=True, key="g_ttf")
        with c4:
            st.markdown('<div class="plot-tile-title">Equipment Failure Frequency</div>', unsafe_allow_html=True)
            st.plotly_chart(build_equipment_frequency(filtered_df, top_n), use_container_width=True, key="g_equip")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: ASSET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_view == "node":
    if not selected_location or filtered_df.empty:
        st.markdown('<div class="alert-info">ℹ Select a node from the sidebar to view granular metrics.</div>',
                    unsafe_allow_html=True)
    else:
        loc_data = filtered_df[filtered_df["Functional_Location"] == selected_location]
        if loc_data.empty:
            st.markdown(f'<div class="alert-warning">No data for node: {selected_location}</div>',
                        unsafe_allow_html=True)
        else:
            asset_fails = len(loc_data)
            a_mttf = loc_data["TTF_Any_Mode_Days"].mean()
            a_mttr = loc_data["TTR_Mode_Days"].mean()
            total_window = max(1, (date_to - date_from).days)
            a_avail = max(0.0, min(100.0,
                ((total_window - loc_data["TTR_Mode_Days"].sum()) / total_window) * 100))

            st.markdown(f'<div class="ela-section-title">Active Node: {selected_location} Metrics</div>',
                        unsafe_allow_html=True)
            st.markdown(kpi_row([
                kpi_card(f"Asset MTTF ({asset_fails} Fails)",
                         f"{float(a_mttf):.1f} Days" if not np.isnan(float(a_mttf)) else "N/A", "card-purple"),
                kpi_card("Asset Availability", f"{a_avail:.2f}%", "card-green"),
                kpi_card("Asset MTTR",
                         f"{float(a_mttr)*24:.1f} Hours" if not np.isnan(float(a_mttr)) else "N/A", "card-orange"),
            ]), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="plot-tile-title">Failure Mode Share %</div>', unsafe_allow_html=True)
                st.plotly_chart(build_asset_pie(loc_data, top_n), use_container_width=True, key="n_pie")
            with c2:
                st.markdown('<div class="plot-tile-title">Chronological TTF Performance Trend</div>', unsafe_allow_html=True)
                st.plotly_chart(build_chronological_ttf(loc_data), use_container_width=True, key="n_chron")

            c3, c4 = st.columns(2)
            with c3:
                st.markdown('<div class="plot-tile-title">Cross-Section Count Matrix Map</div>', unsafe_allow_html=True)
                st.plotly_chart(build_heatmap(loc_data), use_container_width=True, key="n_heat")
            with c4:
                st.markdown('<div class="plot-tile-title">Distribution of Days TO Failures (TTF)</div>', unsafe_allow_html=True)
                st.plotly_chart(build_ttf_histogram(loc_data), use_container_width=True, key="n_ttf")

            c5, c6 = st.columns(2)
            with c5:
                st.markdown('<div class="plot-tile-title">Duane Plot – Reliability Growth Model</div>', unsafe_allow_html=True)
                st.plotly_chart(build_growth_pattern(loc_data), use_container_width=True, key="n_duane")
            with c6:
                st.markdown('<div class="plot-tile-title">Failure Mode Concentration</div>', unsafe_allow_html=True)
                st.plotly_chart(build_failure_concentration(loc_data, top_n), use_container_width=True, key="n_conc")

            st.markdown('<div class="plot-tile-title">MTTF vs MTTR Profile</div>', unsafe_allow_html=True)
            st.plotly_chart(build_mttf_mttr_profile(loc_data), use_container_width=True, key="n_profile")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: PREDICTIVE SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_view == "predictive":
    if not selected_location or filtered_df.empty:
        st.markdown('<div class="alert-info">ℹ Select a node from the sidebar to run predictive Weibull analysis.</div>',
                    unsafe_allow_html=True)
    else:
        loc_data = filtered_df[filtered_df["Functional_Location"] == selected_location]
        a_ttf    = loc_data["TTF_Any_Mode_Days"].dropna().astype(float)

        if loc_data.empty or len(a_ttf) <= 2:
            st.markdown(
                f'<div class="alert-warning">Insufficient data for Weibull analysis on node: '
                f'{selected_location}. Select a node with more failure events.</div>',
                unsafe_allow_html=True,
            )
        else:
            asset_fails = len(loc_data)
            try:
                shape, _, scale = stats.weibull_min.fit(a_ttf, floc=0)
                r_t = stats.weibull_min.sf(test_days, shape, loc=0, scale=scale)
                f_t = 1 - r_t
            except Exception:
                shape = r_t = f_t = None

            st.markdown(f'<div class="ela-section-title">Predictive Analysis: {selected_location}</div>',
                        unsafe_allow_html=True)
            st.markdown(kpi_row([
                kpi_card("Target Horizon",       f"{test_days} Days",         "card-teal"),
                kpi_card("Historical Failures",  f"{asset_fails} Events",     "card-purple"),
                kpi_card("Failure Probability F(t)",
                         f"{f_t*100:.1f}%" if f_t is not None else "N/A",    "card-orange"),
            ]), unsafe_allow_html=True)

            st.markdown(f'<div class="plot-tile-title">Asset Weibull Profile: {selected_location}</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(build_asset_weibull(loc_data, test_days),
                            use_container_width=True, key="p_weibull")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(FOOTER_HTML, unsafe_allow_html=True)