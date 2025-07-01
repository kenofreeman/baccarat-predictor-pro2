import pandas as pd
import numpy as np
import gdown
import os
import joblib
from datetime import datetime, timedelta
import warnings

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Chargement des données
def load_data():
    """Charge les données historiques depuis le fichier local ou Google Drive."""
    try:
        # Chemin du fichier local
        data_path = 'data/baccara_data.csv'
        
        # Créer le dossier s'il n'existe pas
        os.makedirs('data', exist_ok=True)
        
        # Télécharger les données si nécessaire
        if not os.path.exists(data_path):
            # ID Google Drive - à remplacer par le vôtre
            file_id = '1YOUR_GOOGLE_DRIVE_FILE_ID'
            gdown.download(
                f'https://drive.google.com/uc?id={file_id}',
                data_path,
                quiet=False
            )
        
        # Charger les données
        df = pd.read_csv(data_path, low_memory=False)
        
        # Nettoyage des colonnes
        df.columns = [col.strip() for col in df.columns]
        
        # Conversion des dates/heures
        if 'Date' in df.columns and 'Time' in df.columns:
            try:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            except:
                df['DateTime'] = pd.to_datetime(df['Date'])
        elif 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
        else:
            raise ValueError("Colonnes de date/heure manquantes")
        
        # Extraction des composants temporels
        df['Hour'] = df['DateTime'].dt.hour
        df['Minute'] = df['DateTime'].dt.minute
        df['Day_of_week'] = df['DateTime'].dt.dayofweek
        df['Month'] = df['DateTime'].dt.month
        
        # Features statistiques
        if 'Joueur_Score' in df.columns and 'Banquier_Score' in df.columns:
            player_avg = df.groupby('Hour')['Joueur_Score'].mean().rename('Player_Avg')
            banker_avg = df.groupby('Hour')['Banquier_Score'].mean().rename('Banker_Avg')
            
            df = df.merge(player_avg, on='Hour', how='left')
            df = df.merge(banker_avg, on='Hour', how='left')
            
            # Remplir les moyennes manquantes par les moyennes globales
            df['Player_Avg'].fillna(df['Joueur_Score'].mean(), inplace=True)
            df['Banker_Avg'].fillna(df['Banquier_Score'].mean(), inplace=True)
        else:
            df['Player_Avg'] = 0
            df['Banker_Avg'] = 0
        
        # Encodage cible
        winner_mapping = {
            'Joueur': 0, 'Player': 0,
            'Banquier': 1, 'Banker': 1,
            'Egalite': 2, 'Tie': 2, 'Égalité': 2
        }
        
        if 'Winner' in df.columns:
            df['Target'] = df['Winner'].map(winner_mapping).fillna(2).astype(int)
        else:
            df['Target'] = 2
        
        # Features séquentielles
        df.sort_values('DateTime', inplace=True)
        df['Rolling_Player_Win'] = (df['Target'] == 0).rolling(window=100, min_periods=1).mean().fillna(0)
        
        return df
    
    except Exception as e:
        print(f"Erreur de chargement des données: {str(e)}")
        return pd.DataFrame()

# Mise à jour des données
def update_data():
    """Met à jour les données avec les nouveaux fichiers."""
    try:
        # Chemin des données
        data_path = 'data/baccara_data.csv'
        backup_dir = 'data/backups/'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Télécharger les nouveaux fichiers depuis Google Drive
        new_data = None
        today = datetime.today().strftime('%Y-%m-%d')
        new_file_id = '1YOUR_NEW_FILE_ID'  # Remplacer par l'ID réel
        
        try:
            new_data_path = f'data/baccara_{today}.csv'
            gdown.download(
                f'https://drive.google.com/uc?id={new_file_id}',
                new_data_path,
                quiet=False
            )
            new_data = pd.read_csv(new_data_path)
        except:
            print("Aucune nouvelle donnée disponible")
            return False
        
        if new_data is not None:
            # Charger les données existantes
            if os.path.exists(data_path):
                existing_data = pd.read_csv(data_path)
                # Sauvegarde
                backup_path = f"{backup_dir}backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                existing_data.to_csv(backup_path, index=False)
                # Concaténation
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                updated_data = new_data
            
            # Sauvegarder
            updated_data.to_csv(data_path, index=False)
            return True
        return False
    except Exception as e:
        print(f"Erreur de mise à jour: {str(e)}")
        return False

# Génération des parties pour prédiction
def generate_games(date, start_hour, end_hour):
    """Génère un DataFrame des parties à prédire pour une plage horaire."""
    try:
        games = []
        game_id = int(date.strftime("%Y%m%d")) * 10000
        current_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=start_hour)
        
        # Charger les données pour les moyennes
        df = load_data()
        
        # Calculer les moyennes globales
        player_global_avg = df['Joueur_Score'].mean() if 'Joueur_Score' in df.columns else 4.5
        banker_global_avg = df['Banquier_Score'].mean() if 'Banquier_Score' in df.columns else 4.5
        
        rolling_player_win = df['Rolling_Player_Win'].iloc[-1] if not df.empty and 'Rolling_Player_Win' in df.columns else 0.48
        
        while current_time.hour <= end_hour:
            for minute in range(60):
                if current_time.hour == end_hour and minute > 0:
                    break
                
                # Calculer les moyennes pour cette heure
                hour = current_time.hour
                
                if not df.empty:
                    hour_data = df[df['Hour'] == hour]
                    player_avg = hour_data['Player_Avg'].mean() if not hour_data.empty else player_global_avg
                    banker_avg = hour_data['Banker_Avg'].mean() if not hour_data.empty else banker_global_avg
                else:
                    player_avg = player_global_avg
                    banker_avg = banker_global_avg
                
                games.append({
                    'GameId': game_id,
                    'Date': date,
                    'Time': current_time.strftime("%H:%M"),
                    'Hour': hour,
                    'Minute': minute,
                    'Day_of_week': date.weekday(),
                    'Month': date.month,
                    'Player_Avg': player_avg,
                    'Banker_Avg': banker_avg,
                    'Rolling_Player_Win': rolling_player_win
                })
                game_id += 1
                current_time += timedelta(minutes=1)
        
        games_df = pd.DataFrame(games)
        games_df['PredictionID'] = f"{date.strftime('%Y%m%d')}_{start_hour}_{end_hour}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return games_df
    
    except Exception as e:
        print(f"Erreur de génération des parties: {str(e)}")
        return pd.DataFrame()

# Sauvegarde des prédictions
def save_prediction(prediction_df):
    """Sauvegarde les prédictions dans l'historique."""
    try:
        history_path = 'data/prediction_history.csv'
        prediction_df['PredictionDate'] = datetime.now()
        
        if os.path.exists(history_path):
            history = pd.read_csv(history_path)
            history = pd.concat([history, prediction_df], ignore_index=True)
        else:
            history = prediction_df
        
        history.to_csv(history_path, index=False)
        return True
    except:
        return False

# Chargement de l'historique des prédictions
def load_prediction_history(start_date, end_date):
    """Charge l'historique des prédictions entre deux dates."""
    try:
        history_path = 'data/prediction_history.csv'
        if not os.path.exists(history_path):
            return None
        
        history = pd.read_csv(history_path)
        history['Date'] = pd.to_datetime(history['Date'])
        mask = (history['Date'] >= pd.Timestamp(start_date)) & (history['Date'] <= pd.Timestamp(end_date))
        return history.loc[mask]
    except:
        return None
