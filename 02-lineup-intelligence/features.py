"""
Lineup feature engineering.

Given the 5 players' season stats, build an ORDER-INVARIANT feature vector
(every feature is a sum / mean / std / min / max / count / pairwise aggregate,
so the permutation of players never matters).

Feature families
----------------
* shooting      : fg3_pct mean/std/min, fg3a_rate sum, shooter count, non-shooter count
* creation      : ast_pct sum/max, usage sum/max/std, handler count, redundancy terms
* size          : height mean/std/min/max, big count, reb_pct sum
* defense       : dbpm sum, blk max/sum, stl sum, def_rating_proxy mean
* impact        : obpm sum, dbpm sum, bpm sum (obpm+dbpm)
* turnover      : tov_pct sum / max
* pairwise      : mean pairwise |usage diff|, mean pairwise |height diff|,
                  number of pairs where both are handlers / both are non-shooters
* interactions  : shooters x bigs, shooters x best-passer AST%, handlers x top usage,
                  steals x rim protection, spacing x rim protection

Deliberately NOT included: hand-coded threshold flags such as "exactly three
non-shooters" or "zero handlers". The counts are there; letting the tree model
discover where the steps are is the point of the comparison with Ridge.
"""
from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd

SHOOTER_3P = 0.36
SHOOTER_RATE = 0.30
HANDLER_AST = 20.0
HANDLER_USG = 20.0
BIG_HEIGHT = 81.0

PLAYER_COLS = [
    "player_id", "player_name", "team", "position", "height_in", "minutes", "usage_pct",
    "fg3_pct", "fg3a_rate", "ast_pct", "reb_pct", "stl_pct", "blk_pct", "tov_pct",
    "obpm", "dbpm", "def_rating_proxy",
]


def _flags(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["is_shooter"] = ((p.fg3_pct > SHOOTER_3P) & (p.fg3a_rate > SHOOTER_RATE)).astype(int)
    p["is_handler"] = ((p.ast_pct > HANDLER_AST) & (p.usage_pct > HANDLER_USG)).astype(int)
    p["is_big"] = ((p.height_in >= BIG_HEIGHT) | (p.position == "C")).astype(int)
    return p


def lineup_features(five: pd.DataFrame) -> dict[str, float]:
    """Feature dict for exactly five player rows (any order)."""
    if len(five) != 5:
        raise ValueError(f"lineup_features expects 5 players, got {len(five)}")
    p = _flags(five)
    f: dict[str, float] = {}

    # shooting / spacing
    f["fg3_pct_mean"] = p.fg3_pct.mean()
    f["fg3_pct_std"] = p.fg3_pct.std(ddof=0)
    f["fg3_pct_min"] = p.fg3_pct.min()
    f["fg3a_rate_sum"] = p.fg3a_rate.sum()
    f["shooters"] = p.is_shooter.sum()
    f["non_shooters"] = 5 - f["shooters"]

    # creation / ball handling
    f["ast_pct_sum"] = p.ast_pct.sum()
    f["ast_pct_max"] = p.ast_pct.max()
    f["usage_sum"] = p.usage_pct.sum()
    f["usage_max"] = p.usage_pct.max()
    f["usage_std"] = p.usage_pct.std(ddof=0)
    f["handlers"] = p.is_handler.sum()

    # size / rebounding
    f["height_mean"] = p.height_in.mean()
    f["height_std"] = p.height_in.std(ddof=0)
    f["height_min"] = p.height_in.min()
    f["height_max"] = p.height_in.max()
    f["bigs"] = p.is_big.sum()
    f["reb_pct_sum"] = p.reb_pct.sum()

    # defense
    f["dbpm_sum"] = p.dbpm.sum()
    f["blk_pct_max"] = p.blk_pct.max()
    f["blk_pct_sum"] = p.blk_pct.sum()
    f["stl_pct_sum"] = p.stl_pct.sum()
    f["def_rating_proxy_mean"] = p.def_rating_proxy.mean()

    # offensive / overall impact
    f["obpm_sum"] = p.obpm.sum()
    f["obpm_min"] = p.obpm.min()
    f["bpm_sum"] = (p.obpm + p.dbpm).sum()

    # turnovers
    f["tov_pct_sum"] = p.tov_pct.sum()
    f["tov_pct_max"] = p.tov_pct.max()

    # experience / minutes (how much the coach already trusts these players)
    f["minutes_mean_log"] = float(np.log1p(p.minutes).mean())
    f["zero_minute_players"] = float((p.minutes <= 0).sum())

    # pairwise redundancy features (all 10 pairs)
    pairs = list(itertools.combinations(range(5), 2))
    usg = p.usage_pct.values
    hgt = p.height_in.values
    hand = p.is_handler.values
    shoot = p.is_shooter.values
    f["pair_usage_absdiff_mean"] = float(np.mean([abs(usg[i] - usg[j]) for i, j in pairs]))
    f["pair_height_absdiff_mean"] = float(np.mean([abs(hgt[i] - hgt[j]) for i, j in pairs]))
    f["pair_both_handlers"] = float(sum(hand[i] and hand[j] for i, j in pairs))
    f["pair_both_nonshooters"] = float(sum((not shoot[i]) and (not shoot[j]) for i, j in pairs))

    # interactions
    f["shooters_x_bigs"] = f["shooters"] * f["bigs"]
    f["shooters_x_ast_max"] = f["shooters"] * f["ast_pct_max"]
    f["handlers_x_usage_max"] = f["handlers"] * f["usage_max"]
    f["stl_x_blk"] = f["stl_pct_sum"] * f["blk_pct_max"]
    f["spacing_x_rim"] = f["shooters"] * f["blk_pct_max"]
    return {k: float(v) for k, v in f.items()}


def feature_names() -> list[str]:
    """Stable feature order (derived from a dummy lineup)."""
    dummy = pd.DataFrame({
        "player_id": range(5), "player_name": ["a"] * 5, "team": ["t"] * 5, "position": ["PG", "SG", "SF", "PF", "C"],
        "height_in": [74, 76, 78, 81, 84], "minutes": [1000] * 5, "usage_pct": [25, 20, 18, 16, 20],
        "fg3_pct": [.37, .38, .39, .35, .1], "fg3a_rate": [.4, .5, .6, .4, .05], "ast_pct": [30, 15, 10, 8, 10],
        "reb_pct": [5, 6, 8, 14, 20], "stl_pct": [1.5, 1.4, 1.5, 1, .8], "blk_pct": [.4, .5, .9, 2, 5],
        "tov_pct": [13, 11, 9, 10, 12], "obpm": [2, 1, 0, 0, 0], "dbpm": [-.5, -.5, 1, .5, 2.5],
        "def_rating_proxy": [112] * 5,
    })
    return list(lineup_features(dummy).keys())


def build_feature_matrix(lineups: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Features for a lineups table with columns p1..p5 (vectorised)."""
    players = players.reset_index(drop=True)
    pos = pd.Series(players.index.values, index=players.player_id.values)
    combos = np.column_stack([pos.loc[lineups[f"p{i}"].values].values for i in range(1, 6)])
    X = features_for_combos(players, combos)
    X.index = lineups.index
    return X


def features_for_combos(roster: pd.DataFrame, combos: Iterable[tuple[int, ...]]) -> pd.DataFrame:
    """Features for many index-combos of a roster DataFrame (positional indices).

    Vectorised over combos with numpy (identical output to calling
    `lineup_features` row by row, ~50x faster for the 1287 fives of a roster).
    """
    roster = _flags(roster.reset_index(drop=True))
    C = np.asarray(list(combos), dtype=int)
    if C.size == 0:
        return pd.DataFrame(columns=feature_names())
    if C.shape[1] != 5:
        raise ValueError("features_for_combos expects 5-player combos")

    def col(name):
        return roster[name].to_numpy(dtype=float)[C]            # (n_combos, 5)

    fg3, rate, ast, usg = col("fg3_pct"), col("fg3a_rate"), col("ast_pct"), col("usage_pct")
    hgt, reb, dbpm, blk = col("height_in"), col("reb_pct"), col("dbpm"), col("blk_pct")
    stl, drp, obpm, tov = col("stl_pct"), col("def_rating_proxy"), col("obpm"), col("tov_pct")
    mins, sh, hd, bg = col("minutes"), col("is_shooter"), col("is_handler"), col("is_big")
    pairs = np.array(list(itertools.combinations(range(5), 2)))
    pi, pj = pairs[:, 0], pairs[:, 1]

    f: dict[str, np.ndarray] = {}
    f["fg3_pct_mean"] = fg3.mean(1); f["fg3_pct_std"] = fg3.std(1); f["fg3_pct_min"] = fg3.min(1)
    f["fg3a_rate_sum"] = rate.sum(1)
    f["shooters"] = sh.sum(1); f["non_shooters"] = 5 - f["shooters"]
    f["ast_pct_sum"] = ast.sum(1); f["ast_pct_max"] = ast.max(1)
    f["usage_sum"] = usg.sum(1); f["usage_max"] = usg.max(1); f["usage_std"] = usg.std(1)
    f["handlers"] = hd.sum(1)
    f["height_mean"] = hgt.mean(1); f["height_std"] = hgt.std(1); f["height_min"] = hgt.min(1); f["height_max"] = hgt.max(1)
    f["bigs"] = bg.sum(1); f["reb_pct_sum"] = reb.sum(1)
    f["dbpm_sum"] = dbpm.sum(1); f["blk_pct_max"] = blk.max(1); f["blk_pct_sum"] = blk.sum(1)
    f["stl_pct_sum"] = stl.sum(1); f["def_rating_proxy_mean"] = drp.mean(1)
    f["obpm_sum"] = obpm.sum(1); f["obpm_min"] = obpm.min(1); f["bpm_sum"] = (obpm + dbpm).sum(1)
    f["tov_pct_sum"] = tov.sum(1); f["tov_pct_max"] = tov.max(1)
    f["minutes_mean_log"] = np.log1p(mins).mean(1); f["zero_minute_players"] = (mins <= 0).sum(1).astype(float)
    f["pair_usage_absdiff_mean"] = np.abs(usg[:, pi] - usg[:, pj]).mean(1)
    f["pair_height_absdiff_mean"] = np.abs(hgt[:, pi] - hgt[:, pj]).mean(1)
    f["pair_both_handlers"] = (hd[:, pi] * hd[:, pj]).sum(1)
    f["pair_both_nonshooters"] = ((1 - sh[:, pi]) * (1 - sh[:, pj])).sum(1)
    f["shooters_x_bigs"] = f["shooters"] * f["bigs"]
    f["shooters_x_ast_max"] = f["shooters"] * f["ast_pct_max"]
    f["handlers_x_usage_max"] = f["handlers"] * f["usage_max"]
    f["stl_x_blk"] = f["stl_pct_sum"] * f["blk_pct_max"]
    f["spacing_x_rim"] = f["shooters"] * f["blk_pct_max"]
    return pd.DataFrame(f)[feature_names()]


if __name__ == "__main__":
    names = feature_names()
    print(f"{len(names)} features:")
    for n in names:
        print(" -", n)
