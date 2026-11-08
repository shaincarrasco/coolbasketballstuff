"""
Synthetic, offline dataset for Lineup Intelligence (Five IQ).

Offline fallback / benchmark (`cli.py generate --source sample`). The default build
uses real ESPN data (data/espn_loader.py); this module invents a small fictional league:

* 8 fictional teams (no real NBA franchises)
* 13 fictional players per team with season-level stats
* ~480 historical 5-man lineup rows with minutes, OffRtg, DefRtg, NetRtg

Lineup ratings are produced by a HIDDEN "true" synergy function (spacing,
rim protection, ball-handler redundancy, size balance, turnover drag) plus
noise scaled by 1/sqrt(minutes). The model never sees that function; it only
sees player box-score-style stats and lineup results, exactly as it would
with real data.

Everything is seeded and deterministic.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 5150
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Fictional league ---------------------------------------------------------
TEAMS = [
    ("Harbor City", "Tidewalkers", "HCT"),
    ("Ironvale", "Foundry", "IVF"),
    ("Sable Ridge", "Nightjars", "SRN"),
    ("Port Lumen", "Beacons", "PLB"),
    ("Copperfield", "Kestrels", "CFK"),
    ("Glasswater", "Current", "GWC"),
    ("Redmesa", "Monarchs", "RMM"),
    ("Northgate", "Voltage", "NGV"),
]

FIRST = [
    "Dax", "Rion", "Kael", "Tovin", "Marek", "Jalen", "Osric", "Bram", "Ezra",
    "Cyrus", "Lior", "Teo", "Niko", "Soren", "Idris", "Kofi", "Amari", "Zeke",
    "Quill", "Rafe", "Hollis", "Tariq", "Lennox", "Jory", "Emeka", "Vance",
    "Paxton", "Ruben", "Tobiah", "Caius", "Dorian", "Ansel", "Fenwick", "Gideon",
    "Hale", "Imre", "Jude", "Kip", "Lucan", "Mace", "Nolan", "Orin", "Pike",
    "Quade", "Remy", "Silas", "Thane", "Ulric", "Vidal", "Wes", "Xavi", "Yusuf",
]
LAST = [
    "Okonkwo", "Vasquez", "Hartwell", "Lindqvist", "Mbeki", "Castellano", "Reyes",
    "Thorne", "Abara", "Delacroix", "Fairbanks", "Ngata", "Petrov", "Sandoval",
    "Whitlock", "Adeyemi", "Barrowman", "Calloway", "Draven", "Ellsworth",
    "Faulkner", "Granger", "Haldane", "Iverness", "Jorgensen", "Kettering",
    "Lockridge", "Marchetti", "Navarro", "Oyelaran", "Pembrook", "Quintero",
    "Rasmussen", "Sorensen", "Tanaka-Bell", "Umbert", "Valdemar", "Wyndham",
    "Xiong", "Yarrow", "Zabala", "Achterberg", "Bellweather", "Corvino",
    "Dunmore", "Esposito", "Fairweather", "Grimaldi", "Hollander", "Ivory",
    "Juneau", "Kristiansen",
]


@dataclass(frozen=True)
class Archetype:
    name: str
    position: str
    height: tuple[float, float]       # mean, sd (inches)
    usage: tuple[float, float]
    fg3_pct: tuple[float, float]
    fg3a_rate: tuple[float, float]    # share of FGA that are threes
    ast_pct: tuple[float, float]
    reb_pct: tuple[float, float]
    stl_pct: tuple[float, float]
    blk_pct: tuple[float, float]
    tov_pct: tuple[float, float]
    obpm: tuple[float, float]
    dbpm: tuple[float, float]


ARCHETYPES = {
    "lead_guard":    Archetype("lead_guard", "PG", (74.5, 1.5), (27, 4), (.355, .035), (.38, .08), (30, 6), (5, 1.5), (1.6, .4), (0.4, .2), (13.5, 2.0), (2.0, 1.8), (-0.8, 0.8)),
    "combo_guard":   Archetype("combo_guard", "SG", (75.5, 1.5), (23, 4), (.365, .035), (.45, .08), (18, 5), (5.5, 1.5), (1.4, .4), (0.5, .2), (11.5, 2.0), (0.8, 1.6), (-0.6, 0.8)),
    "sniper_wing":   Archetype("sniper_wing", "SF", (78.0, 1.5), (17, 3), (.400, .030), (.62, .08), (9, 3), (7, 1.5), (1.0, .3), (0.8, .3), (8.5, 1.5), (0.6, 1.2), (-0.4, 0.8)),
    "three_and_d":   Archetype("three_and_d", "SF", (78.5, 1.5), (15, 3), (.375, .030), (.55, .08), (8, 3), (8, 1.5), (1.9, .4), (1.3, .4), (9.0, 1.5), (-0.3, 1.0), (1.2, 0.8)),
    "slasher":       Archetype("slasher", "SF", (78.0, 1.5), (24, 4), (.305, .035), (.22, .07), (15, 4), (9, 2), (1.3, .4), (0.9, .3), (12.5, 2.0), (1.2, 1.6), (0.0, 0.8)),
    "stretch_big":   Archetype("stretch_big", "PF", (81.5, 1.5), (18, 3), (.370, .030), (.42, .08), (10, 3), (14, 2.5), (0.9, .3), (2.2, .6), (10.0, 1.5), (0.4, 1.2), (0.6, 0.9)),
    "rim_protector": Archetype("rim_protector", "C", (83.5, 1.5), (16, 3), (.150, .080), (.04, .04), (7, 3), (19, 3), (0.8, .3), (5.5, 1.2), (12.0, 2.0), (-0.6, 1.2), (2.6, 0.9)),
    "post_big":      Archetype("post_big", "C", (82.5, 1.5), (22, 4), (.200, .090), (.08, .05), (12, 4), (17, 3), (0.9, .3), (2.8, .8), (13.0, 2.0), (1.0, 1.5), (0.8, 0.9)),
    "energy_big":    Archetype("energy_big", "PF", (81.0, 1.5), (14, 3), (.280, .060), (.20, .08), (7, 3), (15, 2.5), (1.2, .3), (2.6, .8), (12.5, 2.0), (-0.9, 1.0), (1.1, 0.8)),
}

# Roster template: 13 slots per team (3 guards / 5 wings / 5 bigs-ish, with variety)
ROSTER_TEMPLATE = [
    "lead_guard", "lead_guard", "combo_guard", "combo_guard",
    "sniper_wing", "three_and_d", "three_and_d", "slasher", "slasher",
    "stretch_big", "rim_protector", "post_big", "energy_big",
]


def _draw(rng: np.random.Generator, spec: tuple[float, float], lo=None, hi=None) -> float:
    v = rng.normal(*spec)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return float(v)


def generate_players(rng: np.random.Generator) -> pd.DataFrame:
    # Enough unique fictional names for the whole league (two shuffled pairings, de-duplicated).
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < len(TEAMS) * len(ROSTER_TEMPLATE):
        for f, l in zip(rng.permutation(FIRST), rng.permutation(LAST)):
            nm = f"{f} {l}"
            if nm not in seen:
                seen.add(nm)
                names.append(nm)
    rows = []
    pid = 1000
    name_i = 0
    for (city, nick, abbr) in TEAMS:
        slots = list(ROSTER_TEMPLATE)
        # a little roster variety: swap 1-2 slots at random
        for _ in range(rng.integers(1, 3)):
            slots[rng.integers(len(slots))] = rng.choice(list(ARCHETYPES))
        # minute distribution: 8-9 rotation players, a few deep bench, 1 with zero minutes
        minutes = np.sort(rng.gamma(shape=2.2, scale=520, size=len(slots)))[::-1]
        minutes = np.clip(minutes, 0, 2900)
        minutes[-1] = 0.0                       # two-way / injured: zero minutes
        minutes[-2] = float(rng.integers(20, 120))
        rng.shuffle(slots)
        team_rows = []
        for slot in slots:
            a = ARCHETYPES[slot]
            team_rows.append({
                "player_id": pid,
                "player_name": names[name_i],
                "team": f"{city} {nick}",
                "team_abbr": abbr,
                "archetype": slot,
                "position": a.position,
                "height_in": round(_draw(rng, a.height, 70, 88), 1),
                "usage_pct": round(_draw(rng, a.usage, 8, 38), 1),
                "fg3_pct": round(_draw(rng, a.fg3_pct, 0.0, 0.48), 3),
                "fg3a_rate": round(_draw(rng, a.fg3a_rate, 0.0, 0.85), 3),
                "ast_pct": round(_draw(rng, a.ast_pct, 2, 45), 1),
                "reb_pct": round(_draw(rng, a.reb_pct, 2, 28), 1),
                "stl_pct": round(_draw(rng, a.stl_pct, 0.2, 3.5), 2),
                "blk_pct": round(_draw(rng, a.blk_pct, 0.0, 9.0), 2),
                "tov_pct": round(_draw(rng, a.tov_pct, 5, 22), 1),
                "obpm": round(_draw(rng, a.obpm, -5, 8), 2),
                "dbpm": round(_draw(rng, a.dbpm, -4, 5), 2),
            })
            pid += 1
            name_i += 1
        # Better players play more: order the minutes by a noisy talent signal.
        talent = np.array([r["obpm"] + r["dbpm"] for r in team_rows]) + rng.normal(0, 1.0, len(team_rows))
        for r, mins in zip(team_rows, minutes[np.argsort(np.argsort(-talent))]):
            r["minutes"] = round(float(mins), 0)
        rows.extend(team_rows)
    df = pd.DataFrame(rows)
    # Defensive rating proxy (lower = better), built from dbpm/stl/blk with noise.
    df["def_rating_proxy"] = (
        112.0 - 1.8 * df["dbpm"] - 0.9 * df["stl_pct"] - 0.35 * df["blk_pct"] + rng.normal(0, 1.2, len(df))
    ).round(1)
    df["is_shooter"] = ((df["fg3_pct"] > 0.36) & (df["fg3a_rate"] > 0.30)).astype(int)
    df["is_handler"] = ((df["ast_pct"] > 20) & (df["usage_pct"] > 20)).astype(int)
    df["is_big"] = ((df["height_in"] >= 81) | (df["position"] == "C")).astype(int)
    return df


# --- Hidden "true" synergy ------------------------------------------------------
# The model never sees this. It is here to make the synthetic results coherent.
# Relative weight of additive talent (BPM sums) vs. non-additive "fit" effects.
# Tuned so that fit matters as much as talent -- the premise of the project.
ADDITIVE_W = 0.35
FIT_W = 0.9


def true_synergy(p: pd.DataFrame) -> tuple[float, float]:
    """Return (offensive_delta, defensive_delta) vs league average for 5 players.
    Positive is good for both (defensive_delta is subtracted from DefRtg)."""
    assert len(p) == 5
    shooters = int(((p.fg3_pct > 0.36) & (p.fg3a_rate > 0.30)).sum())
    handlers = int(((p.ast_pct > 20) & (p.usage_pct > 20)).sum())
    bigs = int(((p.height_in >= 81) | (p.position == "C")).sum())
    h_mean, h_std = p.height_in.mean(), p.height_in.std(ddof=0)

    # Offense
    off = ADDITIVE_W * p.obpm.sum()
    fit_off = 0.0
    fit_off += {0: -7.0, 1: -4.5, 2: -1.5, 3: 1.0, 4: 2.5, 5: 3.0}[shooters]          # spacing (non-linear)
    if (5 - shooters) >= 3:                                                       # three non-shooters clog the lane
        fit_off -= 1.5 * ((5 - shooters) - 2)
    fit_off += {0: -4.0, 1: 1.0, 2: 1.8, 3: -1.0, 4: -4.0, 5: -7.0}[handlers]          # ball-handler redundancy
    fit_off -= 0.30 * max(0.0, p.usage_pct.sum() - 110)                               # only one ball
    fit_off -= 0.25 * (p.tov_pct.sum() - 57) * (2.2 if handlers == 0 else 1.0)        # turnovers hurt more with no creator
    fit_off += 0.22 * shooters * max(0.0, p.ast_pct.max() - 15)                       # shooters need a real creator
    fit_off -= 1.5 * (bigs >= 3)                                                      # three bigs -> cramped
    fit_off += 2.0 * (shooters >= 3 and p.blk_pct.max() >= 4.0)                       # pick-and-roll gravity: spacing + lob threat
    fit_off -= 0.08 * max(0.0, p.usage_pct.max() - 30) * max(0, handlers - 1)         # a heliocentric star + extra handlers = waste

    # Defense
    dfn = ADDITIVE_W * p.dbpm.sum()
    fit_def = 0.0
    fit_def += min(p.blk_pct.max(), 6.5) * 1.1 - 3.0                                  # rim protection (saturating)
    fit_def += {0: -5.0, 1: 0.0, 2: 1.0, 3: 0.5, 4: -1.0, 5: -2.0}[bigs]              # need a big, but not four
    fit_def -= 3.0 * max(0.0, 78.0 - h_mean)                                          # too small (hinge, not linear)
    fit_def -= 1.5 * (h_std < 2.0)                                                    # no size differentiation
    fit_def += 0.4 * (p.stl_pct.sum() - 6.5)                                          # active hands
    fit_def += 0.10 * (p.reb_pct.sum() - 50)                                          # finish possessions
    fit_def -= 0.45 * max(0.0, 52.0 - p.reb_pct.sum()) * (bigs <= 1)                 # small lineups get killed on the glass
    fit_def -= 1.2 * max(0, shooters - 3) * (p.blk_pct.max() < 3.0)                   # 4-5 shooters with no rim protector get hunted
    fit_def += 1.0 * (p.stl_pct.sum() - 6.5) * (p.blk_pct.max() >= 4.0)              # perimeter pressure pays with a backstop
    off += FIT_W * fit_off + 2.0
    dfn += FIT_W * fit_def
    return float(off), float(dfn)


def generate_lineups(players: pd.DataFrame, rng: np.random.Generator, per_team: int = 70) -> pd.DataFrame:
    rows = []
    lid = 1
    for team, roster in players.groupby("team", sort=False):
        roster = roster.reset_index(drop=True)
        combos = list(itertools.combinations(range(len(roster)), 5))
        # Coaches play rotation guys together: weight combos by product of player minutes.
        w = np.array([np.prod(roster.loc[list(c), "minutes"].values + 30.0) for c in combos], dtype=float)
        w = w ** 0.55
        w /= w.sum()
        idx = rng.choice(len(combos), size=per_team, replace=False, p=w)
        for k in idx:
            five = roster.loc[list(combos[k])]
            off, dfn = true_synergy(five)
            true_net = off + dfn
            # Minutes: heavy tail, mildly correlated with true quality (coaches notice what works)
            base = rng.lognormal(mean=np.log(70), sigma=0.85)
            mins = float(np.clip(base * np.exp(0.018 * true_net), 12, 700))
            sigma = 24.0 / np.sqrt(mins) + 1.0
            off_noise, def_noise = rng.normal(0, sigma), rng.normal(0, sigma)
            off_rtg = 113.0 + off + off_noise
            def_rtg = 113.0 - dfn + def_noise
            pids = sorted(five.player_id.tolist())
            rows.append({
                "lineup_id": lid,
                "team": team,
                "p1": pids[0], "p2": pids[1], "p3": pids[2], "p4": pids[3], "p5": pids[4],
                "minutes": round(mins, 1),
                "off_rating": round(off_rtg, 1),
                "def_rating": round(def_rtg, 1),
                "net_rating": round(off_rtg - def_rtg, 1),
            })
            lid += 1
    df = pd.DataFrame(rows)
    # A league's minutes-weighted offensive and defensive ratings are equal by
    # construction (every possession is scored by one team and allowed by the
    # other). Center the synthetic league the same way.
    shift = np.average(df.off_rating - df.def_rating, weights=df.minutes) / 2.0
    df["off_rating"] = (df.off_rating - shift).round(1)
    df["def_rating"] = (df.def_rating + shift).round(1)
    df["net_rating"] = (df.off_rating - df.def_rating).round(1)
    return df


def generate(seed: int = SEED, per_team: int = 70, out_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    players = generate_players(rng)
    lineups = generate_lineups(players, rng, per_team=per_team)
    os.makedirs(out_dir, exist_ok=True)
    players.to_csv(os.path.join(out_dir, "players.csv"), index=False)
    lineups.to_csv(os.path.join(out_dir, "lineups.csv"), index=False)
    return players, lineups


if __name__ == "__main__":
    p, l = generate()
    print(f"players: {len(p)} rows -> data/players.csv")
    print(f"lineups: {len(l)} rows -> data/lineups.csv")
    print(l.net_rating.describe().round(2))
