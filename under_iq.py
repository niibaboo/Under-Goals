#!/usr/bin/env python3
"""
Under IQ — built on TheStatsAPI (api.thestatsapi.com)

Same core model as Match IQ (recency-weighted goals form, small-sample
shrinkage toward league average, opponent-adjusted Poisson probability),
but scoped specifically to UNDER GOALS markets across leagues chosen for
being genuinely low-scoring, rather than Match IQ's high-scoring set.

PER-TEAM UNDER GOALS, NOT MATCH-TOTAL UNDER (pivoted from the original
match-total design -- user feedback, with screenshots of the exact
failure mode: Pau 3-2 Laval and Avellino 4-1 Sampdoria both missed
Under 3.5 on the match total, even though the LOSING side in each
(Laval, Sampdoria) individually stayed low-scoring). A match-total Under
bet can be wrecked by the OPPONENT having a big game -- the team you
actually had a read on can do exactly what you expected and the bet
still loses, because the total is one number built from two mostly-
independent scoring processes. Pricing each team's OWN goals separately
isolates the signal to the team the form data is actually about, instead
of exposing it to the other side's variance too. The daily scanners
below are now Team Under 1.5 and Team Under 2.5 (one entry per TEAM per
match, not one entry per match) -- match-total Under 2.5/Under 3.5
scanners have been removed entirely.

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
    docs/under-iq/scanners/team_under15/  (date-paginated, CSV export)
    docs/under-iq/scanners/team_under25/  (date-paginated, CSV export)
    docs/under-iq/scanners/cold_streak/   (date-paginated, CSV export)
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

# TEAM-LEVEL Under goals thresholds — each team's OWN expected goals,
# not the match total (see module docstring for why). Team Under 2.5 is
# deliberately a much higher bar than Team Under 1.5: in these low-
# scoring leagues a team's own Under 2.5 is often true anyway (a team
# scoring 3+ in one match is the exception, not the rule), so it's set
# high to only surface teams the model is genuinely confident about, not
# just "usually true for this team".
#
# TEAM_UNDER15_MIN raised 70 -> 80 (user feedback with real bet365
# evidence: Clermont Foot, Empoli and Genoa all missed Under 1.5 after
# scoring exactly 2, and the model had priced all three at 73-75%
# confidence -- just above the old 70% bar. Misses clustering right at
# the threshold edge is a real calibration signal, not bad luck: at a
# league average of ~2.4-2.5 total goals/match, each team's own share
# sits around 1.2 goals/game, right on top of the 1.5 line, so a bar
# only a few points above 70% was letting through picks that were still
# essentially coin-flip-adjacent. 80% pulls the bar well clear of where
# the actual misses were landing.
TEAM_UNDER15_MIN = 80
TEAM_UNDER25_MIN = 90

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

    # Build scored/conceded newest-first (matches its current order), then
    # reverse once at the end so goals_list reads OLDEST->NEWEST for
    # display — same convention as Euro Ice, after the earlier mix-up
    # there. Don't reorder `matches` itself; only the derived lists.
    scored, conceded = [], []
    for m in matches:
        is_home = m["home_team"]["id"] == team_id
        s = m["score"]
        if s.get("home") is None:
            continue
        scored.append(s["home"] if is_home else s["away"])
        conceded.append(s["away"] if is_home else s["home"])

    scored.reverse()
    conceded.reverse()

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
    conceding rate, not just a flat average.

    exp_total is still computed and kept in the output for reference
    display on the main page (useful context for how high-scoring a
    fixture looks overall), but it no longer drives a market of its own
    -- match-total Under 2.5/Under 3.5 probabilities have been removed.
    Every actual signal below is PER TEAM: Under 1.5 (team scores 0 or 1
    -- the natural tight line here, where a team's own expected goals
    often sits right around 1.0-1.5) and Under 2.5 (team scores 0, 1 or
    2 -- a looser line, naturally higher probability, for when 1.5 is
    too strict to find enough picks)."""
    h_scored = shrink(h_form["avg_scored"], h_form["n_games"], lg_scored)
    h_conceded = shrink(h_form["avg_conceded"], h_form["n_games"], lg_conceded)
    a_scored = shrink(a_form["avg_scored"], a_form["n_games"], lg_scored)
    a_conceded = shrink(a_form["avg_conceded"], a_form["n_games"], lg_conceded)

    exp_home = round(h_scored * (a_conceded / lg_conceded), 2)
    exp_away = round(a_scored * (h_conceded / lg_conceded), 2)
    exp_total = round(exp_home + exp_away, 2)

    p_home_under15 = poisson_cdf(1, exp_home)
    p_away_under15 = poisson_cdf(1, exp_away)
    p_home_under25 = poisson_cdf(2, exp_home)
    p_away_under25 = poisson_cdf(2, exp_away)

    return {
        "exp_home": exp_home, "exp_away": exp_away, "exp_total": exp_total,
        "home_under15": round(p_home_under15 * 100), "away_under15": round(p_away_under15 * 100),
        "home_under25": round(p_home_under25 * 100), "away_under25": round(p_away_under25 * 100),
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
                "match_id": m["id"],  # needed later to look up the real final
                                        # result for the results tracker
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
        writer.writerow(["Date", "League", "HomeTeam", "AwayTeam", "ExpTotal",
                          "HomeExpGoals", "HomeUnder15", "HomeUnder25",
                          "AwayExpGoals", "AwayUnder15", "AwayUnder25"])
        for p in predictions:
            writer.writerow([p["date"], p["league"], p["home_team"], p["away_team"], p["exp_total"],
                              p["exp_home"], p["home_under15"], p["home_under25"],
                              p["exp_away"], p["away_under15"], p["away_under25"]])


CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999">{league} · {time}</div>
  <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{home_team} vs {away_team}</div>
  <div style="font-size:11px;color:#777">Exp Total: {exp_total} <span style="color:#555">(context only — not a market; see Team Under scanners below for the actual signals)</span></div>
  <div style="font-size:11px;color:#aaa;margin-top:6px">{home_team} (exp {exp_home}) — Under 1.5: <span style="color:#a0e8a0">{home_under15}%</span> &nbsp;|&nbsp; Under 2.5: <span style="color:#a0e8a0">{home_under25}%</span></div>
  <div style="font-size:11px;color:#aaa">{away_team} (exp {exp_away}) — Under 1.5: <span style="color:#a0e8a0">{away_under15}%</span> &nbsp;|&nbsp; Under 2.5: <span style="color:#a0e8a0">{away_under25}%</span></div>
  <div style="font-size:10px;color:#8b98a8;margin-top:8px">last 5 (old→new): {home_team} {home_hist} &nbsp;|&nbsp; {away_team} {away_hist}</div>
</div>"""

def build_legs(all_predictions):
    """One Under 1.5 leg and one Under 2.5 leg per TEAM, for the Acca
    Builder. Deliberately NOT filtered by the scanner thresholds — the
    builder draws from the full pool so it has enough legs to actually
    hit a target odds, same as Match IQ/Euro Ice's Safest Bet Builder.

    Both legs for the same team share "subject" = team name, so the
    1-per-subject cap in the builder JS treats them as the SAME
    diversification slot — Under 1.5 and Under 2.5 for one team are the
    same underlying scoring-rate read at two different bars, not two
    independent signals, exactly the reasoning match Under 2.5/Under 3.5
    shared a subject key under the old match-total model. A team's two
    legs (home) and the opponent's two legs (away) from the same match
    ARE allowed to coexist, since they're different teams' own scoring,
    not the same number measured twice."""
    legs = []
    for p in all_predictions:
        match_label = f"{p['home_team']} vs {p['away_team']}"
        for team_key, team_name, exp_key, under15_key, under25_key in [
            ("home", p["home_team"], "exp_home", "home_under15", "home_under25"),
            ("away", p["away_team"], "exp_away", "away_under15", "away_under25"),
        ]:
            exp_goals = p[exp_key]
            for market_key, market_label, category in [
                (under15_key, "Under 1.5 Goals", "Team Under 1.5 Goals"),
                (under25_key, "Under 2.5 Goals", "Team Under 2.5 Goals"),
            ]:
                prob = p[market_key]
                if prob <= 0:
                    continue
                legs.append({
                    "match": match_label,
                    "subject": team_name,  # capped at 1 leg per TEAM — see docstring
                    "market": f"{team_name} {market_label}",
                    "prob": prob,
                    "category": category,
                    "detail": f"{team_name} exp {exp_goals} goals",
                    "league": p["league"],
                })
    return legs


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Under IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center;margin-bottom:2px">📉 Under IQ — Full Stats</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">Powered by TheStatsAPI · {generated}</p>
<p style="text-align:center;margin:6px 0 0;font-size:12px">Daily Signals: <a href="scanners/team_under15/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">Team Under 1.5</a>·<a href="scanners/team_under25/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">Team Under 2.5</a>·<a href="scanners/cold_streak/" style="color:#7ec8ff;text-decoration:none;margin:0 4px">Cold Form/Streak</a></p>
<p style="text-align:center;margin:4px 0 0;font-size:12px"><a href="results/index.html" style="color:#f59e0b;text-decoration:none">📊 Results Tracker</a></p>
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

# --- Team Under 1.5 / Team Under 2.5 scanners --------------------------
# Flattens each match into up to two TEAM entries (home + away), each
# carrying that team's OWN expected goals and Under 1.5/2.5 probabilities
# -- replaces the old match-total Under 2.5/Under 3.5 scanners entirely
# (see module docstring). A team's own goals_list (last 5, oldest->newest)
# is carried through too so the scanner card can show it as context,
# same as the old match card did for both sides.

def build_team_under_entries(all_predictions):
    entries = []
    for p in all_predictions:
        match_label = f"{p['home_team']} vs {p['away_team']}"
        for team_key, team_name, opp_name, exp_key, under15_key, under25_key, form_key, is_home in [
            ("home", p["home_team"], p["away_team"], "exp_home", "home_under15", "home_under25", "home_form", True),
            ("away", p["away_team"], p["home_team"], "exp_away", "away_under15", "away_under25", "away_form", False),
        ]:
            entries.append({
                "team": team_name, "opponent": opp_name, "is_home": is_home,
                "match": match_label, "league": p["league"],
                "date": p["date"], "date_key": p["date_key"],
                "exp_goals": p[exp_key], "under15": p[under15_key], "under25": p[under25_key],
                "goals_list": p[form_key].get("goals_list") or [],
            })
    return entries


TEAM_SCANNER_CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038;display:flex;gap:12px;align-items:flex-start">
  <div style="min-width:72px;text-align:center;background:#0f1318;border:1px solid #2a3038;border-radius:10px;padding:8px 6px;flex-shrink:0">
    <div style="font-size:10px;color:#888">{badge_label}</div>
    <div style="font-size:20px;font-weight:bold;color:#a0e8a0">{badge_value}%</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:11px;color:#999">{league} · {time}</div>
    <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{team} <span style="color:#8b98a8;font-weight:normal;font-size:12px">({home_away})</span> vs {opponent}</div>
    <div style="font-size:11px;color:#aaa">Exp Goals: <span style="color:#a0e8a0">{exp_goals}</span></div>
    <div style="font-size:10px;color:#8b98a8;margin-top:6px">last 5 (old→new): {hist}</div>
  </div>
</div>"""

TEAM_SCANNER_HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{page_title} — Under IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../../under_iq_index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Under IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">{icon} {page_title}</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{subtitle} · {generated}</p>
<p style="text-align:center;margin:8px 0 4px;font-size:12px"><a href="../team_under15/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Team Under 1.5</a>·<a href="../team_under25/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Team Under 2.5</a>·<a href="../cold_streak/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Cold Form/Streak</a></p>
{date_bar}
<p style="text-align:center;margin:8px 0 4px"><a href="{csv_name}" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Export CSV</a></p>
<p style="text-align:center;color:#888;font-size:12px;margin-bottom:14px">{qualified_count} team-entries qualified</p>
{cards}
</body></html>"""

TEAM_SCANNER_CONFIGS = [
    {
        "dir": "team_under15", "market_key": "under15", "min": TEAM_UNDER15_MIN,
        "title": "Team Under 1.5 Goals Daily Scanner", "icon": "📉", "badge_label": "U1.5",
        "subtitle_fmt": f"All teams with ≥{TEAM_UNDER15_MIN}% probability of scoring 0 or 1 goals in their own match",
    },
    {
        "dir": "team_under25", "market_key": "under25", "min": TEAM_UNDER25_MIN,
        "title": "Team Under 2.5 Goals Daily Scanner", "icon": "🔒", "badge_label": "U2.5",
        "subtitle_fmt": f"All teams with ≥{TEAM_UNDER25_MIN}% probability of scoring 2 or fewer goals in their own match",
    },
]


def render_team_scanner_cards(entries, market_key, badge_label):
    if not entries:
        return '<p style="text-align:center;color:#888">No teams on this date qualified.</p>'
    cards = ""
    for e in entries:
        cards += TEAM_SCANNER_CARD_TEMPLATE.format(
            badge_label=badge_label, badge_value=e[market_key],
            league=e["league"], time=e["date"][:16].replace("T", " "),
            team=e["team"], home_away="Home" if e["is_home"] else "Away", opponent=e["opponent"],
            exp_goals=e["exp_goals"], hist="/".join(str(v) for v in e["goals_list"]) or "—",
        )
    return cards


def make_team_scanner_html(entries, page_title, icon, subtitle, market_key, badge_label,
                            csv_name, date_label=None, prev_href=None, next_href=None):
    prev_link = f'<a href="{prev_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">◀</a>' if prev_href else '<span style="color:#444;font-size:20px">◀</span>'
    next_link = f'<a href="{next_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">▶</a>' if next_href else '<span style="color:#444;font-size:20px">▶</span>'
    date_bar = f"""
<div style="display:flex;align-items:center;justify-content:center;gap:20px;margin:10px 0 4px">
  {prev_link}
  <span style="font-size:15px;font-weight:bold">{date_label or ''}</span>
  {next_link}
</div>""" if date_label else ""

    return TEAM_SCANNER_HTML_TEMPLATE.format(
        page_title=page_title, icon=icon, subtitle=subtitle,
        generated=datetime.now().strftime("%d %b %H:%M"),
        date_bar=date_bar, csv_name=csv_name,
        qualified_count=len(entries),
        cards=render_team_scanner_cards(entries, market_key, badge_label),
    )


def write_team_scanner_csv(entries, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "League", "Team", "HomeAway", "Opponent", "ExpGoals", "Under15", "Under25", "Last5Goals"])
        for e in entries:
            writer.writerow([e["date"], e["league"], e["team"], "Home" if e["is_home"] else "Away",
                              e["opponent"], e["exp_goals"], e["under15"], e["under25"],
                              "/".join(str(v) for v in e["goals_list"])])


def build_daily_signals_scanners(all_predictions, base_dir="docs/under-iq/scanners"):
    team_entries = build_team_under_entries(all_predictions)

    for cfg in TEAM_SCANNER_CONFIGS:
        qualified = [e for e in team_entries if e[cfg["market_key"]] >= cfg["min"]]
        qualified.sort(key=lambda e: (e["date_key"], -e[cfg["market_key"]]))

        out_dir = f"{base_dir}/{cfg['dir']}"
        os.makedirs(out_dir, exist_ok=True)

        by_date = group_by_date(qualified)
        date_keys = list(by_date.keys())

        common = dict(page_title=cfg["title"], icon=cfg["icon"], subtitle=cfg["subtitle_fmt"],
                      market_key=cfg["market_key"], badge_label=cfg["badge_label"],
                      csv_name=f"{cfg['dir']}_predictions.csv")

        if not date_keys:
            with open(f"{out_dir}/index.html", "w") as f:
                f.write(make_team_scanner_html([], **common))
        else:
            for i, date_key in enumerate(date_keys):
                prev_href = date_page_filename(date_keys[i - 1]) if i > 0 else None
                next_href = date_page_filename(date_keys[i + 1]) if i < len(date_keys) - 1 else None
                page_html = make_team_scanner_html(
                    by_date[date_key], date_label=format_date_label(date_key),
                    prev_href=prev_href, next_href=next_href, **common,
                )
                with open(f"{out_dir}/{date_page_filename(date_key)}", "w") as f:
                    f.write(page_html)
            with open(f"{out_dir}/{date_page_filename(date_keys[0])}") as f:
                soonest_html = f.read()
            with open(f"{out_dir}/index.html", "w") as f:
                f.write(soonest_html)

        write_team_scanner_csv(qualified, f"{out_dir}/{cfg['dir']}_predictions.csv")
        print(f"  {cfg['title']}: {len(qualified)} team-entries across {len(date_keys)} date(s)")

    build_cold_streak_scanner(all_predictions, base_dir)


# --- Cold Form / Real Cold Streak --------------------------------------
# Same distinction as Match IQ/Euro Ice/Strike Zone's Hot Form vs Real
# Streak, inverted for Under IQ's whole thesis: teams genuinely
# UNDER-scoring, not over. "Cold Form" is an AVERAGE over the last 5
# games -- a team can qualify even if their most recent game was a
# blowout, as long as earlier games pulled the average down low enough.
# That's not what "streak" means (see the Match IQ Goal Streak fix this
# session), so "Real Cold Streak" is a separate, stricter check: walking
# backward from the most recent game and counting how many in a ROW
# stayed at or under a per-game threshold, stopping at the first game
# that broke it.
#
# TIGHTENED (user request: "tighten the under model" -> raise the
# confidence bars): COLD_FORM_MAX 1.0 -> 0.8, so a team now needs a
# genuinely low-scoring last-5 average to qualify, not just "below one
# goal a game on average". REAL_COLD_STREAK_MIN_LENGTH 3 -> 4, requiring
# one more consecutive cold game before a streak counts.
# REAL_COLD_STREAK_THRESHOLD stays at 1 -- tightening it to 0 would mean
# "only an outright shutout keeps the streak alive", which is too strict
# to realistically find a run of 4+ in most leagues; the extra length
# requirement does the tightening instead.
COLD_FORM_MAX = 0.8
COLD_FORM_MIN_GAMES = 5
REAL_COLD_STREAK_THRESHOLD = 1   # per-game goals AT OR BELOW this extends the cold streak
REAL_COLD_STREAK_MIN_LENGTH = 4  # shortest run that counts as "a streak"


def _last5_goal_avg_cold(form):
    """goals_list is oldest-first (see get_team_form) -- the LAST 5
    entries are the most recent 5 games. Returns None with fewer than
    5 games available."""
    gl = form.get("goals_list") or []
    if len(gl) < COLD_FORM_MIN_GAMES:
        return None
    last5 = gl[-COLD_FORM_MIN_GAMES:]
    return round(sum(last5) / len(last5), 2), last5


def _current_cold_streak(goals_list, threshold=REAL_COLD_STREAK_THRESHOLD):
    """oldest-first -- walk in REVERSE to go from the most recent game
    backward, exactly what a real streak needs."""
    streak = 0
    for g in reversed(goals_list):
        if g <= threshold:
            streak += 1
        else:
            break
    return streak


def build_cold_form_entries(all_predictions):
    """COLD FORM -- one entry per TEAM whose last 5 games average
    <= COLD_FORM_MAX goals scored."""
    entries = []
    for p in all_predictions:
        for team_key, opp_key, form_key, is_home in [
            ("home_team", "away_team", "home_form", True),
            ("away_team", "home_team", "away_form", False),
        ]:
            result = _last5_goal_avg_cold(p[form_key])
            if not result:
                continue
            avg5, last5 = result
            if avg5 <= COLD_FORM_MAX:
                entries.append({
                    "team": p[team_key], "opponent": p[opp_key], "is_home": is_home,
                    "league": p["league"], "date": p["date"], "date_key": p["date_key"],
                    "last5_avg": avg5, "last5_goals": last5,
                })
    entries.sort(key=lambda e: (e["date_key"], e["last5_avg"]))  # coldest (lowest) first
    return entries


def build_real_cold_streak_entries(all_predictions):
    """REAL COLD STREAK -- one entry per TEAM currently on a genuine
    CONSECUTIVE run of >= REAL_COLD_STREAK_MIN_LENGTH games scoring
    <= REAL_COLD_STREAK_THRESHOLD goals each, with no break."""
    entries = []
    for p in all_predictions:
        for team_key, opp_key, form_key, is_home in [
            ("home_team", "away_team", "home_form", True),
            ("away_team", "home_team", "away_form", False),
        ]:
            form = p[form_key]
            gl = form.get("goals_list") or []
            streak_len = _current_cold_streak(gl)
            if streak_len >= REAL_COLD_STREAK_MIN_LENGTH:
                entries.append({
                    "team": p[team_key], "opponent": p[opp_key], "is_home": is_home,
                    "league": p["league"], "date": p["date"], "date_key": p["date_key"],
                    "streak_len": streak_len, "streak_games": gl[-streak_len:],
                    "full_sample": streak_len >= len(gl),
                })
    entries.sort(key=lambda e: (e["date_key"], -e["streak_len"]))
    return entries


COLD_STREAK_CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038;display:flex;gap:12px;align-items:flex-start">
  <div style="min-width:72px;text-align:center;background:#0f1318;border:1px solid #2a3038;border-radius:10px;padding:8px 6px;flex-shrink:0">
    <div style="font-size:10px;color:#888">L5 AVG</div>
    <div style="font-size:20px;font-weight:bold;color:#7ec8ff">{last5_avg}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:11px;color:#999">{league} · {time}</div>
    <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{team} <span style="color:#8b98a8;font-weight:normal;font-size:12px">({home_away})</span> vs {opponent}</div>
    <div style="font-size:10px;color:#8b98a8">last 5 (old→new): {last5_str}</div>
  </div>
</div>"""

REAL_COLD_STREAK_CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:14px;margin:10px 0;border:1px solid #2a3038;display:flex;gap:12px;align-items:flex-start">
  <div style="min-width:72px;text-align:center;background:#0f1318;border:1px solid #7ec8ff;border-radius:10px;padding:8px 6px;flex-shrink:0">
    <div style="font-size:10px;color:#888">STREAK</div>
    <div style="font-size:20px;font-weight:bold;color:#7ec8ff">{streak_len}{plus}</div>
  </div>
  <div style="flex:1;min-width:0">
    <div style="font-size:11px;color:#999">{league} · {time}</div>
    <div style="font-size:15px;font-weight:bold;margin:2px 0 6px">{team} <span style="color:#8b98a8;font-weight:normal;font-size:12px">({home_away})</span> vs {opponent}</div>
    <div style="font-size:10px;color:#8b98a8">{streak_len} straight game{s} ≤{threshold} (old→new): {streak_str}</div>
  </div>
</div>"""

COLD_STREAK_HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cold Form &amp; Streaks — Under IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../../under_iq_index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Under IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">🧊 Cold Form &amp; Streaks</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{generated}</p>
<p style="text-align:center;margin:8px 0 4px;font-size:12px"><a href="../team_under15/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Team Under 1.5</a>·<a href="../team_under25/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Team Under 2.5</a>·<a href="../cold_streak/" style="color:#7ec8ff;text-decoration:none;margin:0 6px">Cold Form/Streak</a></p>
{date_bar}
<p style="text-align:center;margin:8px 0 4px"><a href="{csv_name}" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Export CSV</a></p>

<h3 style="margin:18px 0 2px;font-size:15px">📊 Cold Form <span style="color:#8b98a8;font-weight:normal;font-size:11px">(avg ≤{max_avg} over last {min_games} games — most recent game may not itself have been cold)</span></h3>
<p style="text-align:center;color:#888;font-size:12px;margin:2px 0 10px">{cold_form_count} team(s)</p>
{cold_form_cards}

<h3 style="margin:22px 0 2px;font-size:15px">🧊 Real Cold Streak <span style="color:#8b98a8;font-weight:normal;font-size:11px">(≥{min_streak}+ CONSECUTIVE games ≤{streak_threshold}, no break)</span></h3>
<p style="text-align:center;color:#888;font-size:12px;margin:2px 0 10px">{streak_count} team(s)</p>
{streak_cards}

<div style="font-size:11px;color:#8b98a8;text-align:center;margin-top:20px;line-height:1.6">
  Both sections are raw recent-FORM screens, not probabilistic predictions like the Under
  2.5/3.5 scanners. Cold Form and Real Cold Streak measure genuinely different things — a
  team can appear in one, both, or neither. Cross-check against that team's Under 2.5/3.5
  probability for their specific upcoming matchup before treating either alone as a signal.
</div>
</body></html>"""


def render_cold_form_cards(entries):
    if not entries:
        return '<p style="text-align:center;color:#888">No teams currently qualify.</p>'
    cards = ""
    for e in entries:
        cards += COLD_STREAK_CARD_TEMPLATE.format(
            last5_avg=e["last5_avg"], league=e["league"], time=e["date"][:16].replace("T", " "),
            team=e["team"], home_away="Home" if e["is_home"] else "Away", opponent=e["opponent"],
            last5_str="/".join(str(v) for v in e["last5_goals"]),  # already oldest-first
        )
    return cards


def render_real_cold_streak_cards(entries):
    if not entries:
        return '<p style="text-align:center;color:#888">No teams currently on a qualifying cold streak.</p>'
    cards = ""
    for e in entries:
        cards += REAL_COLD_STREAK_CARD_TEMPLATE.format(
            streak_len=e["streak_len"], plus="+" if e["full_sample"] else "",
            league=e["league"], time=e["date"][:16].replace("T", " "),
            team=e["team"], home_away="Home" if e["is_home"] else "Away", opponent=e["opponent"],
            s="" if e["streak_len"] == 1 else "s", threshold=REAL_COLD_STREAK_THRESHOLD,
            streak_str="/".join(str(v) for v in e["streak_games"]),
        )
    return cards


def make_cold_streak_html(cold_form_entries, streak_entries, date_label=None, prev_href=None, next_href=None):
    prev_link = f'<a href="{prev_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">◀</a>' if prev_href else '<span style="color:#444;font-size:20px">◀</span>'
    next_link = f'<a href="{next_href}" style="color:#7ec8ff;text-decoration:none;font-size:20px">▶</a>' if next_href else '<span style="color:#444;font-size:20px">▶</span>'
    date_bar = f"""
<div style="display:flex;align-items:center;justify-content:center;gap:20px;margin:10px 0 4px">
  {prev_link}
  <span style="font-size:15px;font-weight:bold">{date_label or ''}</span>
  {next_link}
</div>""" if date_label else ""

    return COLD_STREAK_HTML_TEMPLATE.format(
        max_avg=COLD_FORM_MAX, min_games=COLD_FORM_MIN_GAMES,
        min_streak=REAL_COLD_STREAK_MIN_LENGTH, streak_threshold=REAL_COLD_STREAK_THRESHOLD,
        generated=datetime.now().strftime("%d %b %H:%M"),
        date_bar=date_bar, csv_name="cold_streak_predictions.csv",
        cold_form_count=len(cold_form_entries), cold_form_cards=render_cold_form_cards(cold_form_entries),
        streak_count=len(streak_entries), streak_cards=render_real_cold_streak_cards(streak_entries),
    )


def write_cold_streak_csv(cold_form_entries, streak_entries, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Date", "League", "Team", "HomeAway", "Opponent", "Value", "Games"])
        for e in cold_form_entries:
            writer.writerow(["Cold Form (avg)", e["date"], e["league"], e["team"],
                              "Home" if e["is_home"] else "Away", e["opponent"],
                              e["last5_avg"], "/".join(str(v) for v in e["last5_goals"])])
        for e in streak_entries:
            writer.writerow(["Real Cold Streak (consecutive)", e["date"], e["league"], e["team"],
                              "Home" if e["is_home"] else "Away", e["opponent"],
                              e["streak_len"], "/".join(str(v) for v in e["streak_games"])])


def build_cold_streak_scanner(all_predictions, base_dir="docs/under-iq/scanners"):
    cold_form_entries = build_cold_form_entries(all_predictions)
    streak_entries = build_real_cold_streak_entries(all_predictions)
    out_dir = f"{base_dir}/cold_streak"
    os.makedirs(out_dir, exist_ok=True)

    cold_form_by_date = group_by_date(cold_form_entries)
    streak_by_date = group_by_date(streak_entries)
    date_keys = sorted(set(cold_form_by_date.keys()) | set(streak_by_date.keys()))

    if not date_keys:
        with open(f"{out_dir}/index.html", "w") as f:
            f.write(make_cold_streak_html([], []))
    else:
        for i, date_key in enumerate(date_keys):
            prev_href = date_page_filename(date_keys[i - 1]) if i > 0 else None
            next_href = date_page_filename(date_keys[i + 1]) if i < len(date_keys) - 1 else None
            page_html = make_cold_streak_html(
                cold_form_by_date.get(date_key, []), streak_by_date.get(date_key, []),
                date_label=format_date_label(date_key),
                prev_href=prev_href, next_href=next_href,
            )
            with open(f"{out_dir}/{date_page_filename(date_key)}", "w") as f:
                f.write(page_html)
        with open(f"{out_dir}/{date_page_filename(date_keys[0])}") as f:
            soonest_html = f.read()
        with open(f"{out_dir}/index.html", "w") as f:
            f.write(soonest_html)

    write_cold_streak_csv(cold_form_entries, streak_entries, f"{out_dir}/cold_streak_predictions.csv")
    print(f"  Cold Form: {len(cold_form_entries)} team-entries · Real Cold Streak: {len(streak_entries)} "
          f"team-entries across {len(date_keys)} date(s)")


def render_main_cards(predictions):
    if not predictions:
        return '<p style="text-align:center;color:#888">No fixtures found in the current window.</p>'
    cards = ""
    for p in predictions:
        home_hist = "/".join(str(v) for v in p["home_form"]["goals_list"]) or "—"
        away_hist = "/".join(str(v) for v in p["away_form"]["goals_list"]) or "—"
        cards += CARD_TEMPLATE.format(
            league=p["league"], time=p["date"][:16].replace("T", " "),
            home_team=p["home_team"], away_team=p["away_team"], exp_total=p["exp_total"],
            exp_home=p["exp_home"], home_under15=p["home_under15"], home_under25=p["home_under25"],
            exp_away=p["exp_away"], away_under15=p["away_under15"], away_under25=p["away_under25"],
            home_hist=home_hist, away_hist=away_hist,
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

    try:
        import results_tracker
        results_tracker.run_results_tracker(
            build_team_under_entries(all_predictions),
            build_cold_form_entries(all_predictions),
            build_real_cold_streak_entries(all_predictions),
            api_key,
            thresholds={
                "team_under15_min": TEAM_UNDER15_MIN, "team_under25_min": TEAM_UNDER25_MIN,
                "real_cold_streak_threshold": REAL_COLD_STREAK_THRESHOLD,
            },
        )
    except Exception as e:
        # Results tracking sits on top of everything above, which has
        # already succeeded by this point -- a failure here should
        # never take down an otherwise-successful run.
        print(f"\n[!] Results tracker failed, but the rest of this run succeeded: {e}")

    print(f"\nDone — {len(all_predictions)} total fixtures projected.")
