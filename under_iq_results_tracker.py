#!/usr/bin/env python3
"""
Under IQ Results Tracker
--------------------------------------------------------------
Same architecture as Match IQ's results tracker -- logs every qualifying
pick from Under IQ's scanners (Under 2.5, Under 3.5, Cold Form, Real
Cold Streak) into a persistent JSON log, then on LATER runs checks back
on entries whose match has finished and marks hit/miss against real
results. Uses the same TheStatsAPI base/auth pattern (same account as
Match IQ).

Designed to be imported and called from under_iq.py's main() -- save
this file as results_tracker.py in the Under-Goals repo (same folder
as under_iq.py).

Output:
    docs/under-iq/results/log.json    -- the full log
    docs/under-iq/results/index.html  -- dashboard: overall + per-scanner
                                          win rate, recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE = "https://api.thestatsapi.com/api"
LOG_PATH = "docs/under-iq/results/log.json"
DASHBOARD_PATH = "docs/under-iq/results/index.html"


def _headers(key):
    return {"Authorization": f"Bearer {key}"}


def _get(path, key, params=None, timeout=15):
    try:
        r = requests.get(f"{BASE}{path}", headers=_headers(key), params=params or {}, timeout=timeout)
    except Exception as e:
        print(f"    [!] verification request failed: {path} ({e})")
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on {path}: {r.text[:150]}")
        return None
    return r.json()


def _entry_id(scanner, subject, match_date_key, market):
    raw = f"{scanner}|{subject}|{match_date_key}|{market}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Couldn't read existing results log ({e}) -- starting fresh.")
        return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_todays_signals(all_predictions, cold_form_entries, streak_entries, log, thresholds):
    """thresholds must contain: under25_min, under35_min, cold_form_max,
    real_cold_streak_threshold -- passed in explicitly from under_iq.py's
    live constants rather than imported, so this module can't silently
    drift out of sync with them."""
    existing_ids = {e["id"] for e in log}
    added = 0

    def add(scanner, subject, market, value, match_id, match_date, date_key,
             league, home_team, away_team, detail=None):
        nonlocal added
        eid = _entry_id(scanner, subject, date_key, market)
        if eid in existing_ids:
            return
        log.append({
            "id": eid, "scanner": scanner, "subject": subject, "market": market,
            "value": value, "detail": detail,
            "match_id": match_id, "match_date": match_date, "date_key": date_key,
            "league": league, "home_team": home_team, "away_team": away_team,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    for p in all_predictions:
        match_label = f"{p['home_team']} vs {p['away_team']}"
        base = dict(match_id=p["match_id"], match_date=p["date"], date_key=p["date_key"],
                    league=p["league"], home_team=p["home_team"], away_team=p["away_team"])

        if p.get("under25", 0) >= thresholds["under25_min"]:
            add("under25", match_label, "Under 2.5 Goals", p["under25"], detail=f"exp {p.get('exp_total')} goals", **base)
        if p.get("under35", 0) >= thresholds["under35_min"]:
            add("under35", match_label, "Under 3.5 Goals", p["under35"], detail=f"exp {p.get('exp_total')} goals", **base)

    for e in cold_form_entries:
        add("cold_form", e["team"], f"Cold Form vs {e['opponent']}", e["last5_avg"],
            match_id=None, match_date=e["date"], date_key=e["date_key"], league=e["league"],
            home_team=e["team"] if e["is_home"] else e["opponent"],
            away_team=e["opponent"] if e["is_home"] else e["team"],
            detail=f"is_home={e['is_home']}")

    for e in streak_entries:
        add("real_cold_streak", e["team"], f"Real Cold Streak vs {e['opponent']}", e["streak_len"],
            match_id=None, match_date=e["date"], date_key=e["date_key"], league=e["league"],
            home_team=e["team"] if e["is_home"] else e["opponent"],
            away_team=e["opponent"] if e["is_home"] else e["team"],
            detail=f"is_home={e['is_home']}")

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def _verify_under_entry(entry, key, threshold):
    """Under 2.5 / Under 3.5 -- verify from the match's final score.
    Under-goals hit means the ACTUAL total stayed AT OR BELOW the
    threshold (opposite direction from an Over market)."""
    data = _get(f"/football/matches/{entry['match_id']}", key)
    if not data or not data.get("data"):
        return None
    m = data["data"]
    if m.get("status") != "finished":
        return None
    s = m.get("score", {})
    if s.get("home") is None or s.get("away") is None:
        return None
    total = s["home"] + s["away"]
    return {"actual": total, "result": "hit" if total <= threshold else "miss"}


def _verify_cold_entry(entry, key, cold_threshold):
    """Cold Form / Real Cold Streak -- these describe a team's PAST
    (low) scoring form, not a prediction about the match total. Verify
    against whether the flagged team stayed cold (scored <= threshold)
    in the very match the signal was flagged alongside -- the direct
    test of "does being cold coming in correlate with staying cold"."""
    data = _get(f"/football/matches/{entry['match_id']}", key) if entry.get("match_id") else None
    if not data:
        search = _get("/football/matches", key, params={
            "date_from": entry["date_key"], "date_to": entry["date_key"], "per_page": 50,
        })
        if not search or not search.get("data"):
            return None
        match = next((m for m in search["data"]
                      if m["home_team"]["name"] == entry["home_team"] and m["away_team"]["name"] == entry["away_team"]),
                     None)
        if not match or match.get("status") != "finished":
            return None
        s = match.get("score", {})
    else:
        m = data["data"]
        if m.get("status") != "finished":
            return None
        s = m.get("score", {})

    if s.get("home") is None or s.get("away") is None:
        return None
    is_home = entry["detail"] == "is_home=True"
    team_goals = s["home"] if is_home else s["away"]
    return {"actual": team_goals, "result": "hit" if team_goals <= cold_threshold else "miss"}


def verify_pending_results(log, key, thresholds, max_checks=60):
    today = datetime.now(timezone.utc).date().isoformat()
    checked = 0
    updated = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue
        if checked >= max_checks:
            break
        checked += 1

        result = None
        try:
            if entry["scanner"] == "under25":
                result = _verify_under_entry(entry, key, 2.5)
            elif entry["scanner"] == "under35":
                result = _verify_under_entry(entry, key, 3.5)
            elif entry["scanner"] in ("cold_form", "real_cold_streak"):
                result = _verify_cold_entry(entry, key, thresholds["real_cold_streak_threshold"])
        except Exception as e:
            print(f"    [!] verification error for entry {entry['id']} ({entry['scanner']}): {e}")
            result = None

        if result:
            entry["status"] = "verified"
            entry["result"] = result["result"]
            entry["actual"] = result["actual"]
            entry["verified_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1

    print(f"  Results verification: checked {checked} pending entries, {updated} newly verified")
    return log


def build_results_dashboard(log):
    verified = [e for e in log if e["status"] == "verified"]
    pending = [e for e in log if e["status"] == "pending"]

    by_scanner = {}
    for e in verified:
        d = by_scanner.setdefault(e["scanner"], {"hit": 0, "miss": 0})
        d[e["result"]] += 1

    SCANNER_LABELS = {
        "under25": "Under 2.5 Goals", "under35": "Under 3.5 Goals",
        "cold_form": "Cold Form", "real_cold_streak": "Real Cold Streak",
    }

    total_hit = sum(d["hit"] for d in by_scanner.values())
    total_miss = sum(d["miss"] for d in by_scanner.values())
    total = total_hit + total_miss
    overall_pct = round(100 * total_hit / total) if total else None

    rows = ""
    for scanner, label in SCANNER_LABELS.items():
        d = by_scanner.get(scanner, {"hit": 0, "miss": 0})
        n = d["hit"] + d["miss"]
        pct = round(100 * d["hit"] / n) if n else None
        pct_str = f"{pct}%" if pct is not None else "—"
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #233040">
  <span>{label}</span>
  <span style="color:#7ec8ff;font-weight:bold">{pct_str}</span>
  <span style="color:#8b98a8;font-size:12px">{d['hit']}/{n}</span>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #233040;font-size:12px">
  <span>{e['subject']} — {e['market']}</span>
  <span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — Under IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../under_iq_index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← Under IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">📊 Results Tracker</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:11px;color:#888">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:#7ec8ff">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#8b98a8">{total_hit}/{total} verified picks · {len(pending)} pending (match not finished yet)</div>
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">By Scanner</div>
  {rows}
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#888;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#8b98a8;text-align:center;margin-top:20px;line-height:1.6">
  Cold Form / Real Cold Streak are verified against whether the flagged team STAYED cold
  (scored ≤ threshold) in the SAME match the signal was flagged alongside. Under 2.5/3.5 are
  verified against their own actual market. Sample sizes are still small early on — treat
  percentages with real caution until there's a few weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(all_predictions, cold_form_entries, streak_entries, key, thresholds):
    """Single entry point called from under_iq.py's main(). thresholds
    must contain: under25_min, under35_min, real_cold_streak_threshold."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(all_predictions, cold_form_entries, streak_entries, log, thresholds)
    log = verify_pending_results(log, key, thresholds)
    save_log(log)
    build_results_dashboard(log)
