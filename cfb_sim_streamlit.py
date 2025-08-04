import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

st.title("College Football Season Simulator")

st.markdown("""
This app simulates a full college football season based on your uploaded Excel schedule.
It will:
- Simulate all regular-season games X times
- Hold conference championships (top two in each conference)
- Select a 12-team playoff field
- Simulate the playoffs and report which teams most often win the title
""")

uploaded_file = st.file_uploader("Upload your 'Preseason 2025.xlsm' Schedule file", type=["xlsm", "xlsx"])

if uploaded_file:
    N_SIMULATIONS = st.slider("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)

    # --- Read and Clean Data ---
    schedule = pd.read_excel(uploaded_file, sheet_name="Schedule")
    schedule['Win Prob'] = schedule['Win Prob'].str.rstrip('%').astype(float) / 100
    schedule['Team'] = schedule['Team'].str.strip()
    schedule['Opponent'] = schedule['Opponent'].str.strip()

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
            # Look up opponent's conference if possible
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
            for sim in range(N_SIMULATIONS):
                standings = simulate_one_season(schedule, team_info)
                champs = get_conference_champs(standings)
                playoff_df = select_playoff_field(standings, champs)
                champion = simulate_playoff(playoff_df)
                champions.append(champion)
            champ_counts = Counter(champions)
            results_df = pd.DataFrame(champ_counts.items(), columns=["Team", "Titles"]).sort_values("Titles", ascending=False)
            results_df['Share'] = results_df['Titles'] / N_SIMULATIONS

        st.success("Simulation complete!")
        st.subheader("Top Champions by Simulated Titles")
        st.dataframe(results_df.head(10).reset_index(drop=True))
        st.bar_chart(results_df.set_index("Team")["Titles"].head(10))

else:
    st.info("Please upload your schedule file to begin.")
