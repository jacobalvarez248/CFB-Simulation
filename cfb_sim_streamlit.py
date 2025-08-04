import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

st.title("College Football Season Simulator")

st.markdown("""
Upload your Schedule file and simulate the season X times.

**Select the version:**
- **JPR** = reads the 'Schedule' sheet
- **Composite** = reads the 'industry schedule' sheet

You'll see each team's percent chance to:
- Make the playoffs
- Win their conference
- Win the national championship
""")

uploaded_file = st.file_uploader("Upload your 'Preseason 2025.xlsm' Schedule file", type=["xlsm", "xlsx"])

# DROPDOWN FOR SCHEDULE VERSION
sheet_selector = st.selectbox(
    "Select simulation data version:",
    options=[("JPR", "Schedule"), ("Composite", "industry schedule")],
    format_func=lambda x: x[0],
)
selected_sheet_name = sheet_selector[1]

if uploaded_file:
    N_SIMULATIONS = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

    # --- Read and Clean Data ---
    schedule = pd.read_excel(uploaded_file, sheet_name=selected_sheet_name, engine="openpyxl")
    
    # Robust win prob cleaning (handles floats, ints, percents as strings)
    def winprob_clean(val):
        if pd.isnull(val):
            return np.nan
        if isinstance(val, str):
            return float(val.rstrip('%')) / 100
        if isinstance(val, (float, int)):
            return val / 100 if val > 1 else val
        return np.nan

    schedule['Team'] = schedule['Team'].str.strip()
    schedule['Opponent'] = schedule['Opponent'].str.strip()
    schedule['Win Prob'] = schedule['Win Prob'].apply(winprob_clean)

    team_info = schedule.groupby('Team').agg({
        'Conference': 'first',
        'Rank': 'first',
        'Team Strength': 'first'
    }).reset_index().set_index('Team')

    def simulate_one_season(schedule, team_info):
        records = defaultdict(lambda: {'wins': 0, 'losses': 0, 'conf_wins': 0, 'conf_games': 0, 'conference': '', 'rank': 0, 'strength': 0})
        for _, row in schedule.iterrows():
            team, opp = row['Team'], row['Opponent']
            win_prob = row['Win Prob']
            conf = row['Conference']
            try:
                opp_conf = schedule[schedule['Team'] == opp]['Conference'].iloc[0]
            except IndexError:
                opp_conf = None
            win = np.random.rand() < win_prob
            records[team]['conference'] = conf
            records[team]['rank'] = team_info.loc[team, 'Rank']
            records[team]['strength'] = team_info.loc[team, 'Team Strength']
            records[opp]['conference'] = opp_conf if opp_conf else records[opp]['conference']
            records[opp]['rank'] = team_info.loc[opp, 'Rank'] if opp in team_info.index else 0
            records[opp]['strength'] = team_info.loc[opp, 'Team Strength'] if opp in team_info.index else 0
            if win:
                records[team]['wins'] += 1
                records[opp]['losses'] += 1
            else:
                records[opp]['wins'] += 1
                records[team]['losses'] += 1
            if conf == opp_conf and pd.notnull(conf):
                records[team]['conf_games'] += 1
                records[opp]['conf_games'] += 1
                if win:
                    records[team]['conf_wins'] += 1
                else:
                    records[opp]['conf_wins'] += 1
        teams_df = pd.DataFrame.from_dict(records, orient='index').reset_index().rename(columns={'index': 'Team'})
        conf_standings = teams_df.sort_values(['conference', 'conf_wins', 'wins', 'strength', 'rank'], ascending=[True, False, False, False, True])
        return conf_standings

    def get_conference_champs(conf_standings):
        champs = {}
        for conf, group in conf_standings.groupby('conference'):
            top2 = group.head(2)
            if len(top2) < 2:
                continue
            t1, t2 = top2.iloc[0], top2.iloc[1]
            spread = t1['rank'] - t2['rank']
            win_prob = 1 / (1 + np.exp(-spread / 6))
            winner = t1['Team'] if np.random.rand() < win_prob else t2['Team']
            champs[conf] = winner
        return champs

    def select_playoff_field(conf_standings, champs):
        champ_teams = [team for team in champs.values()]
        champ_df = conf_standings[conf_standings['Team'].isin(champ_teams)].copy()
        champ_df = champ_df.sort_values(['wins', 'strength', 'rank'], ascending=[False, False, True])
        top5 = champ_df.head(5)['Team'].tolist()
        others = conf_standings[~conf_standings['Team'].isin(top5)].copy()
        atlarge = others.sort_values(['wins', 'strength', 'rank'], ascending=[False, False, True]).head(7)['Team'].tolist()
        field = top5 + atlarge
        playoff_df = conf_standings[conf_standings['Team'].isin(field)].copy()
        playoff_df = playoff_df.sort_values(['wins', 'strength', 'rank'], ascending=[False, False, True]).reset_index(drop=True)
        playoff_df['Seed'] = playoff_df.index + 1
        return playoff_df

    def simulate_playoff(playoff_df):
        seeds = playoff_df.sort_values('Seed')
        matchups = [(4+i, 11-i) for i in range(4)]
        winners = []
        for hi, lo in matchups:
            high_team = seeds.iloc[hi]['Team']
            low_team = seeds.iloc[lo]['Team']
            spread = seeds.iloc[hi]['rank'] - seeds.iloc[lo]['rank']
            win_prob = 1 / (1 + np.exp(-spread / 6))
            winner = high_team if np.random.rand() < win_prob else low_team
            winners.append(winner)
        champion = np.random.choice(winners + seeds.iloc[:4]['Team'].tolist())
        return champion

    if st.button("Run Simulation"):
        with st.spinner("Simulating..."):
            champions = []
            playoff_appearances = defaultdict(int)
            conference_champs = defaultdict(int)
            for sim in range(N_SIMULATIONS):
                standings = simulate_one_season(schedule, team_info)
                champs = get_conference_champs(standings)
                playoff_df = select_playoff_field(standings, champs)
                champion = simulate_playoff(playoff_df)
                champions.append(champion)
                # Record playoff appearances
                for team in playoff_df['Team']:
                    playoff_appearances[team] += 1
                # Record conference champs
                for team in champs.values():
                    conference_champs[team] += 1
            champ_counts = Counter(champions)
            all_teams = set(list(playoff_appearances.keys()) + list(conference_champs.keys()) + list(champ_counts.keys()))
            results_df = pd.DataFrame({
                'Team': list(all_teams),
                'Playoff %': [100 * playoff_appearances[t]/N_SIMULATIONS for t in all_teams],
                'Conf Champ %': [100 * conference_champs[t]/N_SIMULATIONS for t in all_teams],
                'Title %': [100 * champ_counts[t]/N_SIMULATIONS for t in all_teams]
            }).sort_values('Title %', ascending=False)

        st.success("Simulation complete!")
        st.subheader("Top Teams by Title %, Playoff %, and Conference Champ %")
        st.write("Percentages are out of all simulations. Sorted by national title chance.")
        st.dataframe(results_df.sort_values("Title %", ascending=False).reset_index(drop=True))
        st.bar_chart(results_df.set_index("Team")[["Title %", "Playoff %", "Conf Champ %"]].head(10))

else:
    st.info("Please upload your schedule file to begin.")
