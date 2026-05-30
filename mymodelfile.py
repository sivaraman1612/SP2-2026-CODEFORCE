import pandas as pd
import numpy as np
import warnings
import os
import xgboost as xgb

warnings.filterwarnings('ignore')

class MyModel:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        
        # Stats Dictionaries
        self.team_bat_avg = {}
        self.team_bowl_avg = {}
        self.h2h_avg = {}
        self.venue_avg = {}
        self.venue_inn_avg = {}
        self.player_bat_strength = {}
        self.player_bowl_economy = {}
        
        # Player Mapping
        self.player_mapping = {}
        
        self.global_avg = 48.0
        self.recent_global_avg = 52.0

    def _parse_ids(self, val):
        if pd.isna(val) or str(val).strip() == '':
            return []
        ids_str = str(val).strip().replace('"', '').replace("'", "")
        if ids_str.isdigit():
            return [int(ids_str)]
        parts = [p.strip() for p in ids_str.split(',') if p.strip()]
        return [int(p) for p in parts if p.isdigit()]

    def fit(self, deliveries_df, players_df=None, matches_df=None):
        if deliveries_df is None or deliveries_df.empty:
            return self

        # 1. Player Mapping (Required by Rule #10)
        if players_df is not None and not players_df.empty:
            id_col = next((c for c in players_df.columns if c.lower() in ['id', 'player_id', 'uniqueid']), None)
            name_col = next((c for c in players_df.columns if c.lower() in ['player_name', 'name', 'player']), None)
            if id_col and name_col:
                self.player_mapping = dict(zip(players_df[id_col].astype(str), players_df[name_col]))

        df = deliveries_df.copy()
        df.columns = df.columns.str.strip()

        # 2. Extract Year to overweight recent trends (Powerplay scores have increased)
        if 'date' in df.columns:
            unique_dates = df['date'].dropna().unique()
            date_to_year = {}
            for d in unique_dates:
                try:
                    parts = str(d).split('-')
                    y = int(parts[0]) if len(parts[0]) == 4 else int(parts[-1])
                    date_to_year[d] = y
                except:
                    date_to_year[d] = 2015
            df['year'] = df['date'].map(date_to_year).fillna(2015)
        else:
            df['year'] = 2015

        # 3. Filter Powerplay Data (Overs 1-6 or 0-5)
        min_over = df['over'].min()
        max_pp_over = 5 if min_over == 0 else 6
        pp_df = df[df['over'] <= max_pp_over].copy()

        pp_df['total_runs'] = pp_df['batsman_runs']
        if 'extras' in pp_df.columns:
            pp_df['total_runs'] += pp_df['extras'].fillna(0)

        # Merge venue if missing in deliveries
        if 'venue' not in pp_df.columns and matches_df is not None and 'venue' in matches_df.columns and 'id' in matches_df.columns:
            venue_map = matches_df[['id', 'venue']].rename(columns={'id': 'matchId'})
            pp_df = pp_df.merge(venue_map, on='matchId', how='left')

        # 4. Aggregate to Innings Level
        group_cols = ['matchId', 'inning', 'batting_team', 'bowling_team']
        if 'year' in pp_df.columns: group_cols.append('year')
        if 'venue' in pp_df.columns: group_cols.append('venue')
            
        innings_grp = pp_df.groupby(group_cols)['total_runs'].sum().reset_index()
        innings_grp.rename(columns={'total_runs': 'target'}, inplace=True)
        
        self.global_avg = float(innings_grp['target'].mean())
        
        # Recent data statistics for better baseline
        recent_innings = innings_grp[innings_grp['year'] >= 2022] if 'year' in innings_grp.columns else innings_grp
        if len(recent_innings) > 50:
            self.recent_global_avg = float(recent_innings['target'].mean())
        else:
            self.recent_global_avg = self.global_avg

        # Use recent data for team and venue averages if possible
        base_df = recent_innings if len(recent_innings) > 200 else innings_grp
        
        self.team_bat_avg = base_df.groupby('batting_team')['target'].mean().to_dict()
        self.team_bowl_avg = base_df.groupby('bowling_team')['target'].mean().to_dict()
        self.h2h_avg = base_df.groupby(['batting_team', 'bowling_team'])['target'].mean().to_dict()
        
        if 'venue' in base_df.columns:
            self.venue_avg = base_df.groupby('venue')['target'].mean().to_dict()
            self.venue_inn_avg = base_df.groupby(['venue', 'inning'])['target'].mean().to_dict()

        # 5. Player Stats (Use all data to ensure we have stats for most players)
        batsman_col = next((c for c in df.columns if c in ['batsman', 'batsman_id']), None)
        bowler_col = next((c for c in df.columns if c in ['bowler', 'bowler_id']), None)
        
        if batsman_col:
            bat_stats = pp_df.groupby(batsman_col).agg(total_runs=('batsman_runs', 'sum'), matches=('matchId', 'nunique'))
            bat_stats['avg_runs_per_pp'] = bat_stats['total_runs'] / bat_stats['matches']
            reliable = bat_stats[bat_stats['matches'] >= 3]
            self.player_bat_strength = reliable['avg_runs_per_pp'].to_dict()
            
        if bowler_col:
            bowl_stats = pp_df.groupby(bowler_col).agg(total_conceded=('total_runs', 'sum'), matches=('matchId', 'nunique'))
            bowl_stats['avg_runs_conceded_per_pp'] = bowl_stats['total_conceded'] / bowl_stats['matches']
            reliable = bowl_stats[bowl_stats['matches'] >= 3]
            self.player_bowl_economy = reliable['avg_runs_conceded_per_pp'].to_dict()

        # 6. Train XGBoost ML Regressor Algorithm
        # Only training on highly localized target-encoded variables guarantees it executes comfortably under the 20s time limit
        # XGBoost captures non-linear relationships much better than basic tree averages
        train_df = innings_grp[innings_grp['year'] >= 2020].copy() if 'year' in innings_grp.columns else innings_grp.copy()
        
        features_list = []
        for _, row in train_df.iterrows():
            bat = row['batting_team']
            bowl = row['bowling_team']
            inn = row['inning']
            venue = row.get('venue', '')
            
            feat = {
                'innings': inn,
                'is_2nd_innings': 1 if inn == 2 else 0,
                'hist_bat_avg': self.team_bat_avg.get(bat, self.recent_global_avg),
                'hist_bowl_avg': self.team_bowl_avg.get(bowl, self.recent_global_avg),
                'hist_venue_avg': self.venue_avg.get(venue, self.recent_global_avg),
                'hist_venue_inn_avg': self.venue_inn_avg.get((venue, inn), self.venue_avg.get(venue, self.recent_global_avg)),
            }
            features_list.append(feat)
            
        X = pd.DataFrame(features_list)
        y = train_df['target'].values
        
        self.feature_columns = list(X.columns)
        
        # Powerplay prediction regression utilizing powerful non-linear optimization rules: XGBRegression
        self.model = xgb.XGBRegressor(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1, 
            random_state=42, 
            n_jobs=1
        )
        self.model.fit(X, y)
        
        return self

    def predict(self, test_df):
        if test_df is None or test_df.empty:
            return pd.DataFrame(columns=['id', 'predicted_score'])

        test_df = test_df.copy()
        test_df.columns = test_df.columns.str.strip()
        
        predictions = []
        
        # Group by innings to ensure only 1 prediction per inning 
        # (Resolves the issue where 72 rows were generated instead of 2)
        if 'innings' in test_df.columns:
            group_by_col = 'innings'
        else:
            # Fallback if no innings column
            test_df['innings'] = 1
            group_by_col = 'innings'
            
        for inn_val, group in test_df.groupby(group_by_col):
            inn = int(inn_val)
            
            # The context (venue, team) is constant per inning, so we take the first row
            row = group.iloc[0]
            bat = str(row.get('batting_team', '')).strip()
            bowl = str(row.get('bowling_team', '')).strip()
            venue = str(row.get('venue', '')).strip() if pd.notna(row.get('venue')) else ''
            
            # Identify all players involved in this inning across all rows
            bat_ids = []
            bowl_ids = []
            for _, r in group.iterrows():
                for col in r.index:
                    col_lower = str(col).lower()
                    if 'batsman' in col_lower and ('id' in col_lower or col_lower == 'batsman'):
                        bat_ids.extend(self._parse_ids(r[col]))
                    if 'bowler' in col_lower and ('id' in col_lower or col_lower == 'bowler'):
                        bowl_ids.extend(self._parse_ids(r[col]))
            
            # Base features going into the XGBoost algorithm
            feat = {
                'innings': inn,
                'is_2nd_innings': 1 if inn == 2 else 0,
                'hist_bat_avg': self.team_bat_avg.get(bat, self.recent_global_avg),
                'hist_bowl_avg': self.team_bowl_avg.get(bowl, self.recent_global_avg),
                'hist_venue_avg': self.venue_avg.get(venue, self.recent_global_avg),
                'hist_venue_inn_avg': self.venue_inn_avg.get((venue, inn), self.venue_avg.get(venue, self.recent_global_avg)),
            }
            
            pred_df = pd.DataFrame([feat])
            for col in self.feature_columns:
                if col not in pred_df.columns:
                    pred_df[col] = 0
            pred_df = pred_df[self.feature_columns]
            
            if self.model is not None:
                base_score = self.model.predict(pred_df)[0]
            else:
                base_score = self.recent_global_avg
            
            # Player level adjustments
            bat_diff = 0.0
            if bat_ids:
                p_avgs = [self.player_bat_strength.get(pid, self.player_bat_strength.get(str(pid))) for pid in bat_ids]
                p_avgs = [p for p in p_avgs if p is not None]
                if p_avgs:
                    bat_diff = np.mean(p_avgs) - (self.global_avg / 3.0)
                    
            bowl_diff = 0.0
            if bowl_ids:
                p_avgs = [self.player_bowl_economy.get(pid, self.player_bowl_economy.get(str(pid))) for pid in bowl_ids]
                p_avgs = [p for p in p_avgs if p is not None]
                if p_avgs:
                    bowl_diff = (self.global_avg / 2.0) - np.mean(p_avgs)
            
            # Add algorithmic bounds
            final_score = base_score + (bat_diff * 1.5) + (bowl_diff * 1.5)
            
            # Truncate to integers properly mapping to requirements
            final_score = int(max(35, min(85, round(final_score))))

            predictions.append({
                "id": inn,
                "predicted_score": final_score,
            })

        ans_df = pd.DataFrame(predictions)
        try:
            # Safely persist
            ans_df.to_csv("submission.csv", index=False)
            ans_df.to_csv("/var/submission.csv", index=False)
        except Exception:
            pass
            
        return ans_df
