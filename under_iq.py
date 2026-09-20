#!/usr/bin/env python3
"""
Under IQ — built on TheStatsAPI (api.thestatsapi.com)

Same core model as Match IQ (recency-weighted goals form, small-sample
shrinkage toward league average, opponent-adjusted Poisson probability),
but scoped specifically to UNDER GOALS markets across leagues chosen for
being genuinely low-scoring, rather than Match IQ's high-scoring set.

WHY A SEPARATE PROJECT, NOT A MATCH IQ MARKET:
Match IQ's league list was deliberately chosen for high scoring (Bundesliga,
Eredivisie, Danish Superliga, Eliteserien — all ~3.0+ goals/match). Running
an Under scanner against that list would mean hunting for low-scoring
outcomes in leagues stocked for the opposite — thin signal by design. Under
IQ instead uses leagues picked FOR being low-scoring:

  Serie B (Italy):        2.34-2.56 goals/match (2022-23, 2025-26)
  Serie A (Italy):        2.43 goals/match (2025-26)
  Greek Super League:     2.45-2.57 goals/match (2024-25, 2025-26)
  Ligue 2 (France):       2.49 goals/match (2025-26)

All four sit consistently below 2.65 across multiple recent seasons —
a curated list based on actual multi-season data, not a guess. A 5th
candidate, Segunda División (Spain), was dropped after 9 name variants
plus a broad search all failed against TheStatsAPI — it appears not to
be covered by this data source at all.

API CALL BUDGET — cheaper than Match IQ per team:
This model only needs each team's GOALS SCORED/CONCEDED, which the base
/football/matches response already includes in its score field. Unlike
Match IQ (which needs shots/corners/cards/xG and therefore one extra
/stats call per game), Under IQ makes ZERO per-match /stats calls. Cost
per team is just the 1 call to list their recent matches — roughly 8x
cheaper per team than Match IQ's ~8 calls/team.

Setup:
    pip3 install requests --break-system-packages
    python3 under_iq.py YOUR_API_KEY

Output:
    docs/under-iq/under_iq_index.html
    docs/under-iq/under_iq_predictions.csv
    docs/under-iq/scanners/under25/  (date-paginated, CSV export)
    docs/under-iq/scanners/under35/  (date-paginated, CSV export)
"""

import os
import sys
import time
import math
import csv
import json
from datetime import datetime, timedelta, timezone
import requests

BASE = "https://api.thestatsapi.com/api"
RECENT_GAMES = 7
PRIOR_STRENGTH = 3
FIXTURE_WINDOW_DAYS = 10

# Thresholds — Under 3.5 is deliberately a much higher bar than Under 2.5.
# In leagues averaging ~2.5 goals/match, Under 3.5 is often true anyway
# (low signal value on its own), so it's set high to only surface matches
# where the model is genuinely confident, not just "usually true here".
SCANNER_UNDER25_MIN = 60
SCANNER_UNDER35_MIN = 80

# Verified via check_league_coverage.py-style research against real season
# data (goals/match across 2024-25 and 2025-26 seasons) before being added —
# see module docstring for the actual numbers. Names must match exactly
# what TheStatsAPI's /football/competitions search returns; find_competition()
# below falls back to the first search result with a warning if no exact
# match is found, so watch the first run's output closely.
LEAGUE_SEARCH_NAMES = [
    "Serie A",
    "Serie B",
    "Stoiximan Super League",  # Greek Super League's sponsor-branded exact name —
                                # confirmed via check_under_iq_leagues.py; "Super
                                # League Greece" doesn't match anything, this does
                                # (id comp_4008, country Greece)
    "Ligue 2",
    # Segunda División (Spain) dropped — 9 name variants plus a broad Spain
    # search all failed via check_segunda.py; TheStatsAPI appears not to
    # cover this league at all, not just under an unexpected name.
]


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


def _get(path, key, params=None, timeout=15):
    """Same adaptive rate-limiting as Match IQ — reads the real
    X-RateLimit-Remaining/Reset headers and backs off only when actually
    close to the limit."""
    try:
        r = requests.get(f"{BASE}{path}", headers=_headers(key), params=params or {}, timeout=timeout)
    except Exception as e:
        print(f"  [!] request failed: {path} ({e})")
        return None

    remaining = r.headers.get("X-RateLimit-Remaining")
    reset = r.headers.get("X-RateLimit-Reset")
    if remaining is not None:
        try:
            remaining = int(remaining)
            if remaining <= 2 and reset:
                wait = max(0, int(reset) - int(time.time())) + 3
                print(f"  Rate limit nearly exhausted ({remaining} left) — waiting {wait}s...")
                time.sleep(wait)
        except (ValueError, TypeError):
            pass

    if r.status_code == 429:
        retry_after = int(r.headers.get("Retry-After", 30))
        print(f"  [!] 429 rate limited — waiting {retry_after}s and retrying once...")
        time.sleep(retry_after)
        return _get(path, key, params, timeout)

    if r.status_code != 200:
        print(f"  [!] {r.status_code} on {path}: {r.text[:200]}")
        return None

    time.sleep(2.0)
    return r.json()


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def find_competition(name, key):
    """Search finds the competition, but its result rows don't include
    current_season_id at all (confirmed via a live API dump — the search
    endpoint's fields are id/name/country/.../xg_available, no season
    field anywhere). Only the single-competition DETAIL endpoint
    (GET /football/competitions/{id}) has it. So every match here gets
    a follow-up detail fetch to actually get a usable season id,
    regardless of what the search row alone would suggest."""
    data = _get("/football/competitions", key, params={"search": name, "per_page": 5})
    if not data:
        return None

    match = None
    for c in data.get("data", []):
        if c["name"].lower() == name.lower():
            match = c
            break
    if not match and data.get("data"):
        print(f"  [!] No exact match for '{name}' — using first result: "
              f"'{data['data'][0]['name']}'. Verify this is correct.")
        match = data["data"][0]
    if not match:
        return None

    detail = _get(f"/football/competitions/{match['id']}", key)
    if detail and detail.get("data"):
        match = {**match, **detail["data"]}  # merges in current_season_id etc.
    return match
    return None


def get_standings(competition_id, season_id, key):
    data = _get(f"/football/competitions/{competition_id}/seasons/{season_id}/standings", key)
    if not data:
        return {}
    return {row["team"]["id"]: row for row in data.get("data", [])}


def league_averages(standings):
    scored = [r["goals_for"] / r["matches_played"] for r in standings.values() if r.get("matches_played")]
    conceded = [r["goals_against"] / r["matches_played"] for r in standings.values() if r.get("matches_played")]
    lg_scored = sum(scored) / len(scored) if scored else 1.2
    lg_conceded = sum(conceded) / len(conceded) if conceded else 1.2
    return lg_scored, lg_conceded


def get_upcoming_matches(competition_id, season_id, key):
    date_from = datetime.now(timezone.utc).date().isoformat()
    date_to = (datetime.now(timezone.utc).date() + timedelta(days=FIXTURE_WINDOW_DAYS)).isoformat()
    data = _get("/football/matches", key, params={
        "competition_id": competition_id, "season_id": season_id,
        "status": "scheduled", "date_from": date_from, "date_to": date_to,
        "per_page": 20,
    })
    return data.get("data", []) if data else []


team_form_cache = {}


def get_team_form(team_id, competition_id, season_id, key):
    """Last N finished matches for a team — GOALS ONLY, no /stats calls.
    The base /football/matches response already includes each match's
    score, so this is a single call per team rather than Match IQ's
    ~8 calls/team (1 listing call + 1 /stats call per recent game)."""
    cache_key = (team_id, competition_id, season_id)
    if cache_key in team_form_cache:
        return team_form_cache[cache_key]

    data = _get("/football/matches", key, params={
        "team_id": team_id, "competition_id": competition_id, "season_id": season_id,
        "status": "finished", "per_page": 10,
    })
    if not data or not data.get("data"):
        team_form_cache[cache_key] = None
        return None

    matches = sorted(data["data"], key=lambda m: m["utc_date"], reverse=True)[:RECENT_GAMES]
    if not matches:
        team_form_cache[cache_key] = None
        return None

    scored, conceded = [], []
    for m in matches:
        is_home = m["home_team"]["id"] == team_id
        s = m["score"]
        if s.get("home") is None:
            continue
        scored.append(s["home"] if is_home else s["away"])
        conceded.append(s["away"] if is_home else s["home"])

    if not scored:
        team_form_cache[cache_key] = None
        return None

    n = len(scored)
    form = {
        "n_games": n,
        "avg_scored": round(sum(scored) / n, 2),
        "avg_conceded": round(sum(conceded) / n, 2),
        "goals_list": scored,
    }
    team_form_cache[cache_key] = form
    return form


def shrink(value, n, league_avg, prior=PRIOR_STRENGTH):
    """Same small-sample protection as Match IQ — a 1-2 game sample
    leans mostly on the league average; by RECENT_GAMES games the
    team's own form dominates."""
    if value is None:
        return league_avg
    return round((n * value + prior * league_avg) / (n + prior), 2)


def predict_goals(h_form, a_form, lg_scored, lg_conceded):
    """Same opponent-adjusted approach as Match IQ's predict_goals —
    each side's own scoring rate weighed against the OTHER side's
    conceding rate, not just a flat average."""
    h_scored = shrink(h_form["avg_scored"], h_form["n_games"], lg_scored)
    h_conceded = shrink(h_form["avg_conceded"], h_form["n_games"], lg_conceded)
    a_scored = shrink(a_form["avg_scored"], a_form["n_games"], lg_scored)
    a_conceded = shrink(a_form["avg_conceded"], a_form["n_games"], lg_conceded)

    exp_home = round(h_scored * (a_conceded / lg_conceded), 2)
    exp_away = round(a_scored * (h_conceded / lg_conceded), 2)
    exp_total = round(exp_home + exp_away, 2)

    p_under25 = poisson_cdf(2, exp_total)   # total <= 2, i.e. Under 2.5
    p_under35 = poisson_cdf(3, exp_total)   # total <= 3, i.e. Under 3.5

    return {
        "exp_home": exp_home, "exp_away": exp_away, "exp_total": exp_total,
        "under25": round(p_under25 * 100), "under35": round(p_under35 * 100),
    }


def date_page_filename(date_key):
    return f"{date_key}.html"


def format_date_label(date_key):
    d = datetime.strptime(date_key, "%Y-%m-%d")
    return d.strftime("%a %d %b")


def group_by_date(predictions):
    by_date = {}
    for p in predictions:
        by_date.setdefault(p["date_key"], []).append(p)
    return dict(sorted(by_date.items()))


def build_all_predictions(key):
    all_predictions = []

    for name in LEAGUE_SEARCH_NAMES:
        comp = find_competition(name, key)
        if not comp:
            print(f"[!] Couldn't find competition '{name}' — skipping.")
            continue

        season_id = comp.get("current_season_id") or comp.get("season_id")
        if not season_id:
            print(f"[!] No season id for '{comp['name']}' — skipping.")
            continue

        print(f"\nLooking up competition: {comp['name']}")
        standings = get_standings(comp["id"], season_id, key)
        lg_scored, lg_conceded = league_averages(standings)
        print(f"  League averages: {lg_scored:.2f} scored/gm, {lg_conceded:.2f} conceded/gm")

        matches = get_upcoming_matches(comp["id"], season_id, key)
        print(f"  {len(matches)} upcoming matches in window")

        for m in matches:
            print(f"    {m['home_team']['name']} vs {m['away_team']['name']}")
            h_form = get_team_form(m["home_team"]["id"], comp["id"], season_id, key)
            a_form = get_team_form(m["away_team"]["id"], comp["id"], season_id, key)
            if not h_form or not a_form:
                continue

            proj = predict_goals(h_form, a_form, lg_scored, lg_conceded)
            merged = {
                "league": comp["name"], "date": m["utc_date"], "date_key": m["utc_date"][:10],
                "home_team": m["home_team"]["name"], "away_team": m["away_team"]["name"],
                "home_form": h_form, "away_form": a_form,
                **proj,
            }
            all_predictions.append(merged)

    all_predictions.sort(key=lambda p: (p["date_key"], p["exp_total"]))
    return all_predictions


def write_csv(predictions, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "League", "HomeTeam", "AwayTeam", "ExpTotal", "Under25", "Under35"])
        for p in predictions:
            writer.writerow([p["date"], p["league"], p["home_team"], p["away_team"],
                              p["exp_total"], p["under25"], p["under35"]])


CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999">{league} · {time}</div>
  <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{home_team} vs {away_team}</div>
  <div style="font-size:11px;color:#aaa">Exp Total: <span style="color:#a0e8a0">{exp_total}</span> &nbsp;|&nbsp; Under 2.5: <span style="color:#a0e8a0">{under25}%</span> &nbsp;|&nbsp; Under 3.5: <span style="color:#a0e8a0">{under35}%</span></div>
</div>"""

def build_legs(all_predictions):
    """One Under 2.5 leg and one Under 3.5 leg per match, for the Acca
    Builder. Deliberately NOT filtered by the scanner thresholds — the
    builder draws from the full pool so it has enough legs to actually
    hit a target odds, same as Match IQ/Euro Ice's Safest Bet Builder."""
    legs = []
    for p in all_predictions:
        match_label = f"{p['home_team']} vs {p['away_team']}"
        for market_key, market_label in [("under25", "Under 2.5 Goals"), ("under35", "Under 3.5 Goals")]:
            prob = p[market_key]
            if prob <= 0:
                continue
            legs.append({
                "match": match_label,
                "subject": match_label,  # capped at 1 leg per MATCH below — see
                                          # module docstring on why Under 2.5 and
                                          # Under 3.5 on the same fixture aren't
                                          # real diversification
                "market": f"{match_label} {market_label}",
                "prob": prob,
                "category": market_label,
                "detail": f"exp total {p['exp_total']} goals",
                "league": p["league"],
            })
    return legs


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Under IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center;margin-bottom:2px">📉 Under IQ — Full Stats</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">Powered by TheStatsAPI · {generated}</p>
<p style="text-align:center;margin:6px 0 0;font-size:12px">Daily Signals: <a href="scanners/under25/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">Under 2.5</a>·<a href="scanners/under35/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">Under 3.5</a></p>
<p style="text-align:center;margin:12px 0 4px"><a href="under_iq_predictions.csv" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Download CSV</a></p>

<div id="builderPanel" style="background:#121820;border:1px solid #233040;border-radius:12px;padding:16px;margin:14px 0">
  <div style="font-size:15px;font-weight:800;margin-bottom:10px">🎯 Daily Acca Builder</div>
  <div id="builderCategoryToggles" style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;font-size:12px"></div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
    <label style="font-size:12px;color:#8b98a8">Target odds:</label>
    <input type="number" step="0.05" min="1.1" value="2.65" id="targetOdds" style="width:70px;background:#161d27;border:1px solid #233040;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <label style="font-size:12px;color:#8b98a8">Max legs:</label>
    <input type="number" step="1" min="2" value="6" id="maxLegs" style="width:70px;background:#161d27;border:1px solid #233040;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <button onclick="buildAcca()" style="background:#22c55e;color:#04140a;font-weight:700;border:none;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">Build</button>
    <button onclick="buildAcca()" style="background:#161d27;border:1px solid #233040;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">🔀 Shuffle</button>
  </div>
  <div id="builderResult" style="font-size:12px;color:#8b98a8">
    Defaults to your 2.3–3.0 target range. Capped at ONE leg per match — Under 2.5
    and Under 3.5 on the same fixture are correlated (not real diversification),
    so the builder never stacks both from one game. Tap Shuffle for a fresh pick
    among equally-safe options without changing your settings.
  </div>
</div>

{cards}

<script>
const LEGS = {legs_json};

function initToggles() {{
  const container = document.getElementById('builderCategoryToggles');
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label style="display:flex;align-items:center;gap:4px;color:white;cursor:pointer"><input type="checkbox" class="catToggle" value="${{c}}" checked> ${{c}}</label>
  `).join('');
}}
function shuffleArr(arr) {{
  for (let i = arr.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }}
  return arr;
}}
function tieredShuffle(legs, bandSize) {{
  const bands = {{}};
  legs.forEach(l => {{
    const band = Math.floor(l.prob / bandSize);
    (bands[band] = bands[band] || []).push(l);
  }});
  const keys = Object.keys(bands).map(Number).sort((a,b) => b-a);
  let result = [];
  keys.forEach(k => {{ result = result.concat(shuffleArr(bands[k])); }});
  return result;
}}
function buildAcca() {{
  const target = parseFloat(document.getElementById('targetOdds').value) || 2.65;
  const maxLegs = parseInt(document.getElementById('maxLegs').value) || 6;
  const activeCats = [...document.querySelectorAll('.catToggle:checked')].map(el => el.value);

  const byCategory = {{}};
  LEGS.filter(l => l.prob > 0 && activeCats.includes(l.category)).forEach(l => {{
    (byCategory[l.category] = byCategory[l.category] || []).push(l);
  }});
  const categories = Object.keys(byCategory);
  categories.forEach(c => {{ byCategory[c] = tieredShuffle(byCategory[c], 5); }});
  const cursor = {{}};
  categories.forEach(c => cursor[c] = 0);

  const chosen = [];
  const subjectCount = {{}};  // capped at 1 per MATCH (see module docstring —
                               // Under 2.5 + Under 3.5 on the same fixture are
                               // correlated, not real diversification)
  let combinedOdds = 1;
  let addedThisPass = true;

  while (addedThisPass && combinedOdds < target && chosen.length < maxLegs) {{
    addedThisPass = false;
    for (const cat of categories) {{
      if (combinedOdds >= target || chosen.length >= maxLegs) break;
      const arr = byCategory[cat];
      while (cursor[cat] < arr.length) {{
        const leg = arr[cursor[cat]];
        cursor[cat]++;
        if (subjectCount[leg.subject]) continue;
        chosen.push(leg);
        combinedOdds *= 100 / leg.prob;
        subjectCount[leg.subject] = 1;
        addedThisPass = true;
        break;
      }}
    }}
  }}

  const out = document.getElementById('builderResult');
  if (!chosen.length) {{ out.innerHTML = 'No legs available to build from.'; return; }}

  const rows = chosen.map(l => `
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #233040">
      <span>${{l.match}}<br><span style="color:#facc15">${{l.category}}</span> <span style="color:#8b98a8">· ${{l.league}}</span>
      <br><span style="color:#8b98a8;font-size:10px">${{l.detail}}</span></span>
      <span style="text-align:right"><span style="color:#facc15;font-weight:bold">${{l.prob}}%</span></span>
    </div>
  `).join('');

  const inRange = combinedOdds >= 2.3 && combinedOdds <= 3.0;
  const rangeNote = inRange
    ? ' <span style="color:#22c55e">(in your 2.3–3.0 target range)</span>'
    : ' <span style="color:#facc15">(outside your 2.3–3.0 target — adjust Target odds or Max legs)</span>';
  const capNote = chosen.length >= maxLegs && combinedOdds < target
    ? ' (hit the leg cap before reaching target — raise Max legs or lower Target odds)'
    : (combinedOdds < target ? ' (ran out of legs before reaching target)' : '');

  out.innerHTML = `
    <div style="color:white;font-size:13px;margin-bottom:6px">
      ${{chosen.length}} legs · est. combined odds ~<b>${{combinedOdds.toFixed(2)}}</b>${{rangeNote}}${{capNote}}
    </div>
    ${{rows}}
    <div style="color:#8b98a8;font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real sportsbook odds
      include their margin, so treat this as a ranking tool, not a firm price.
    </div>
  `;
}}
initToggles();
</script>
</body></html>"""

SCANNER_CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038;display:flex;gap:12px;align-items:flex-start">
  <div style="min-width:72px;text-align:center;background:#0f1318;border:1px solid #2a3038;border-radius:10px;padding:8px 6px;flex-shrink:0">
    <div style="font-size:10px;color:#888">{badge_label}</div>
    <div style="font-size:20px;font-weight:bold;color:#a0e8a0">{badge_value}%</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:11px;color:#999">{league} · {time}</div>
    <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{home_team} vs {away_team}</div>
    <div style="font-size:11px;color:#aaa">Exp Total: <span style="color:#a0e8a0">{exp_total}</span> &nbsp;|&nbsp; Under 2.5: <span style="color:#a0e8a0">{under25}%</span> &nbsp;|&nbsp; Under 3.5: <span style="color:#a0e8a0">{under35}%</span></div>
  </div>
</div>"""

SCANNER_HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{page_title} — Under IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../../under_iq_index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Under IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">{icon} {page_title}</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{subtitle} · {generated}</p>
<p style="text-align:center;margin:8px 0 4px;font-size:12px"><a href="../under25/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Under 2.5</a>·<a href="../under35/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Under 3.5</a></p>
{date_bar}
<p style="text-align:center;margin:8px 0 4px"><a href="{csv_name}" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Export CSV</a></p>
<p style="text-align:center;color:#888;font-size:12px;margin-bottom:14px">{qualified_count} matches qualified</p>
{cards}
</body></html>"""

SCANNER_CONFIGS = [
    {
        "dir": "under25", "market_key": "under25", "min": SCANNER_UNDER25_MIN,
        "title": "Under 2.5 Goals Daily Scanner", "icon": "📉", "badge_label": "U2.5",
        "subtitle_fmt": f"All matches with ≥{SCANNER_UNDER25_MIN}% Under 2.5 probability",
    },
    {
        "dir": "under35", "market_key": "under35", "min": SCANNER_UNDER35_MIN,
        "title": "Under 3.5 Goals Daily Scanner", "icon": "🔒", "badge_label": "U3.5",
        "subtitle_fmt": f"All matches with ≥{SCANNER_UNDER35_MIN}% Under 3.5 probability",
    },
]


def _scanner_badge_value(p, market_key):
    return p[market_key]


def render_scanner_cards(predictions, market_key, badge_label):
    if not predictions:
        return '<p style="text-align:center;color:#888">No fixtures on this date qualified.</p>'
    cards = ""
    for p in predictions:
        cards += SCANNER_CARD_TEMPLATE.format(
            badge_label=badge_label, badge_value=_scanner_badge_value(p, market_key),
            league=p["league"], time=p["date"][:16].replace("T", " "),
            home_team=p["home_team"], away_team=p["away_team"],
            exp_total=p["exp_total"], under25=p["under25"], under35=p["under35"],
        )
    return cards


def make_scanner_html(predictions, page_title, icon, subtitle, market_key, badge_label,
                       csv_name, date_label=None, prev_href=None, next_href=None):
    prev_link = f'<a href="{prev_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">◀</a>' if prev_href else '<span style="color:#444;font-size:20px">◀</span>'
    next_link = f'<a href="{next_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">▶</a>' if next_href else '<span style="color:#444;font-size:20px">▶</span>'
    date_bar = f"""
<div style="display:flex;align-items:center;justify-content:center;gap:20px;margin:10px 0 4px">
  {prev_link}
  <span style="font-size:15px;font-weight:bold">{date_label or ''}</span>
  {next_link}
</div>""" if date_label else ""

    return SCANNER_HTML_TEMPLATE.format(
        page_title=page_title, icon=icon, subtitle=subtitle,
        generated=datetime.now().strftime("%d %b %H:%M"),
        date_bar=date_bar, csv_name=csv_name,
        qualified_count=len(predictions),
        cards=render_scanner_cards(predictions, market_key, badge_label),
    )


def write_scanner_csv(predictions, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "League", "HomeTeam", "AwayTeam", "ExpTotal", "Under25", "Under35"])
        for p in predictions:
            writer.writerow([p["date"], p["league"], p["home_team"], p["away_team"],
                              p["exp_total"], p["under25"], p["under35"]])


def build_daily_signals_scanners(all_predictions, base_dir="docs/under-iq/scanners"):
    for cfg in SCANNER_CONFIGS:
        qualified = [p for p in all_predictions if _scanner_badge_value(p, cfg["market_key"]) >= cfg["min"]]
        qualified.sort(key=lambda p: (p["date_key"], -_scanner_badge_value(p, cfg["market_key"])))

        out_dir = f"{base_dir}/{cfg['dir']}"
        os.makedirs(out_dir, exist_ok=True)

        by_date = group_by_date(qualified)
        date_keys = list(by_date.keys())

        common = dict(page_title=cfg["title"], icon=cfg["icon"], subtitle=cfg["subtitle_fmt"],
                      market_key=cfg["market_key"], badge_label=cfg["badge_label"],
                      csv_name=f"{cfg['dir']}_predictions.csv")

        if not date_keys:
            with open(f"{out_dir}/index.html", "w") as f:
                f.write(make_scanner_html([], **common))
        else:
            for i, date_key in enumerate(date_keys):
                prev_href = date_page_filename(date_keys[i - 1]) if i > 0 else None
                next_href = date_page_filename(date_keys[i + 1]) if i < len(date_keys) - 1 else None
                page_html = make_scanner_html(
                    by_date[date_key], date_label=format_date_label(date_key),
                    prev_href=prev_href, next_href=next_href, **common,
                )
                with open(f"{out_dir}/{date_page_filename(date_key)}", "w") as f:
                    f.write(page_html)
            with open(f"{out_dir}/{date_page_filename(date_keys[0])}") as f:
                soonest_html = f.read()
            with open(f"{out_dir}/index.html", "w") as f:
                f.write(soonest_html)

        write_scanner_csv(qualified, f"{out_dir}/{cfg['dir']}_predictions.csv")
        print(f"  {cfg['title']}: {len(qualified)} fixtures across {len(date_keys)} date(s)")


def render_main_cards(predictions):
    if not predictions:
        return '<p style="text-align:center;color:#888">No fixtures found in the current window.</p>'
    cards = ""
    for p in predictions:
        cards += CARD_TEMPLATE.format(
            league=p["league"], time=p["date"][:16].replace("T", " "),
            home_team=p["home_team"], away_team=p["away_team"],
            exp_total=p["exp_total"], under25=p["under25"], under35=p["under35"],
        )
    return cards


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("THESTATSAPI_KEY")
    if not api_key:
        print("Usage: python3 under_iq.py YOUR_API_KEY  (or set THESTATSAPI_KEY)")
        raise SystemExit(1)

    all_predictions = build_all_predictions(api_key)

    os.makedirs("docs/under-iq", exist_ok=True)

    write_csv(all_predictions, "docs/under-iq/under_iq_predictions.csv")
    with open("docs/under-iq/under_iq.json", "w") as f:
        json.dump(all_predictions, f, indent=2, default=str)

    html = HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        cards=render_main_cards(all_predictions),
        legs_json=json.dumps(build_legs(all_predictions)),
    )
    with open("docs/under-iq/under_iq_index.html", "w") as f:
        f.write(html)

    print(f"\nMain page — {len(all_predictions)} fixtures.")

    print("\nBuilding Daily Signals scanners...")
    build_daily_signals_scanners(all_predictions)

    print(f"\nDone — {len(all_predictions)} total fixtures projected.")
