import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


DATASET_PATH = Path("data/dataset-unique(sport-football-2015-16).csv")
DATA_SOURCE_URL = "https://raw.githubusercontent.com/footballcsv/europe-champions-league/master/2015-16/champs.csv"


def parse_match_date(date_str: str) -> pd.Timestamp:
    if not isinstance(date_str, str) or not date_str.strip():
        return pd.NaT
    m = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})", date_str)
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%d %b %Y", errors="coerce")


def parse_score(score: str) -> Tuple[float, float]:
    if not isinstance(score, str):
        return float("nan"), float("nan")
    score = score.strip()
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", score)
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def normalize_team_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.split("›", 1)[0].strip()
    name = re.sub(r"\s*\(\d+\)\s*$", "", name).strip()
    return name


@st.cache_data(show_spinner=False)
def load_ucl_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        st.error(f"Dataset file not found: {DATASET_PATH}")
        st.stop()

    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    df = df.rename(columns={"\u2211FT": "SumFT"})

    df["match_date"] = df["Date"].apply(parse_match_date)
    df["home_team"] = df["Team 1"].apply(normalize_team_name)
    df["away_team"] = df["Team 2"].apply(normalize_team_name)

    ft_home, ft_away = zip(*df["FT"].map(parse_score))
    df["home_goals"] = pd.to_numeric(pd.Series(ft_home), errors="coerce")
    df["away_goals"] = pd.to_numeric(pd.Series(ft_away), errors="coerce")
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    df["result"] = pd.Series([""] * len(df))
    df.loc[df["home_goals"] > df["away_goals"], "result"] = "Home win"
    df.loc[df["home_goals"] < df["away_goals"], "result"] = "Away win"
    df.loc[df["home_goals"] == df["away_goals"], "result"] = "Draw"

    df = df.dropna(subset=["match_date", "home_team", "away_team", "home_goals", "away_goals"])
    df = df.sort_values("match_date")
    df["month"] = df["match_date"].dt.to_period("M").astype(str)
    return df


def render_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    stages = sorted(df["Stage"].dropna().unique().tolist())
    selected_stages = st.sidebar.multiselect("Stage", options=stages, default=["Group stage"] if "Group stage" in stages else stages[:1])

    available_rounds = (
        df[df["Stage"].isin(selected_stages)]["Round"].dropna().unique().tolist() if selected_stages else df["Round"].dropna().unique().tolist()
    )
    rounds = sorted(available_rounds)
    selected_rounds = st.sidebar.multiselect("Round", options=rounds, default=rounds[:3] if len(rounds) >= 3 else rounds)

    teams = sorted(pd.unique(df[["home_team", "away_team"]].values.ravel("K")).tolist())
    team = st.sidebar.selectbox("Team focus", options=teams, index=teams.index("Barcelona") if "Barcelona" in teams else 0)

    min_d = df["match_date"].min().date()
    max_d = df["match_date"].max().date()
    date_value = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    if isinstance(date_value, (tuple, list)) and len(date_value) == 2:
        start_d, end_d = date_value
    else:
        start_d = date_value
        end_d = date_value

    return selected_stages, selected_rounds, team, start_d, end_d


def filter_df(df: pd.DataFrame, stages, rounds, team: str, start_d, end_d) -> pd.DataFrame:
    out = df.copy()
    if stages:
        out = out[out["Stage"].isin(stages)]
    if rounds:
        out = out[out["Round"].isin(rounds)]
    start_ts = pd.to_datetime(start_d, errors="coerce")
    end_ts = pd.to_datetime(end_d, errors="coerce")
    out = out[(out["match_date"] >= start_ts) & (out["match_date"] <= end_ts)]
    if team:
        out = out[(out["home_team"] == team) | (out["away_team"] == team)]
    return out


def main() -> None:
    st.set_page_config(page_title="UCL 2015-16 Dashboard", layout="wide")
    st.title("UEFA Champions League 2015–16 Dashboard")
    st.caption("Interactive match analytics using a public football dataset (UCL 2015–16).")

    base = load_ucl_dataset()
    stages, rounds, team, start_d, end_d = render_filters(base)
    df = filter_df(base, stages, rounds, team, start_d, end_d)

    if df.empty:
        st.warning("No matches for current filters. Try a different stage/round/date/team.")
        st.stop()

    st.write(f"Dataset source: {DATA_SOURCE_URL}")

    total_matches = len(df)
    total_goals = int(df["total_goals"].sum())
    avg_goals = float(df["total_goals"].mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{total_matches}")
    c2.metric("Total goals", f"{total_goals}")
    c3.metric("Avg goals/match", f"{avg_goals:.2f}")

    team_timeline = df.copy()
    team_timeline["team_goals"] = team_timeline.apply(
        lambda r: r["home_goals"] if r["home_team"] == team else r["away_goals"],
        axis=1,
    )
    line_fig = px.line(
        team_timeline,
        x="match_date",
        y="team_goals",
        title=f"{team}: goals per match over time",
        markers=True,
        labels={"match_date": "Date", "team_goals": "Goals"},
    )
    st.plotly_chart(line_fig, use_container_width=True)

    goals_long = pd.concat(
        [
            df[["home_team", "home_goals"]].rename(columns={"home_team": "team", "home_goals": "goals"}),
            df[["away_team", "away_goals"]].rename(columns={"away_team": "team", "away_goals": "goals"}),
        ],
        ignore_index=True,
    )
    top = goals_long.groupby("team", as_index=False)["goals"].sum().sort_values("goals", ascending=False).head(12)
    bar_fig = px.bar(
        top,
        x="team",
        y="goals",
        title="Top teams by total goals (filtered matches)",
        labels={"team": "Team", "goals": "Goals"},
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    result_counts = (
        df.groupby("result", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    pie_fig = px.pie(
        result_counts,
        names="result",
        values="count",
        title="Match outcomes (filtered)",
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    with st.expander("Data preview"):
        cols = ["Stage", "Round", "Group", "match_date", "home_team", "away_team", "home_goals", "away_goals", "result"]
        st.dataframe(df[cols].sort_values("match_date", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
