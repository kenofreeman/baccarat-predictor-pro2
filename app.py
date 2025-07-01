import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import data_utils  # Fichier local avec fonctions utilitaires

# Configuration de la page
st.set_page_config(
    page_title="🔮 Baccara Predictor Pro",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre avec style CSS
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #4a4a4a;
            text-align: center;
            padding: 15px;
            background: linear-gradient(45deg, #f5f7fa, #c3cfe2);
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .prediction-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.3s;
        }
        .prediction-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        .stMetric {
            border-radius: 10px;
            padding: 15px;
            background-color: #f0f2f6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .winner-player { color: #4CAF50; font-weight: bold; }
        .winner-banker { color: #2196F3; font-weight: bold; }
        .winner-tie { color: #9C27B0; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔮 Baccara Predictor Pro</div>', unsafe_allow_html=True)

# Chargement des modèles
@st.cache_resource
def load_models():
    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Télécharger ou charger les modèles
    models = {}
    
    # LightGBM
    lgbm_path = 'models/lgbm_model.joblib'
    if not os.path.exists(lgbm_path):
        # Télécharger depuis Google Drive
        try:
            gdown.download(
                'https://drive.google.com/file/d/1Wyk2Ws1f3RFgrarBkZhX0H4G6zgmzC2s/view?usp=drive_link',
                lgbm_path,
                quiet=False
            )
        except:
            st.warning("Échec du téléchargement du modèle LightGBM")
    
    if os.path.exists(lgbm_path):
        try:
            models['lgbm'] = joblib.load(lgbm_path)
        except:
            st.error("Erreur de chargement du modèle LightGBM")
    
    # LSTM (optionnel)
    lstm_path = 'models/lstm_model.h5'
    if not os.path.exists(lstm_path):
        try:
            gdown.download(
                'https://drive.google.com/file/d/1QBNy50B3sbaKJTg4uW57XZKImnRVI95m/view?usp=drive_link',
                lstm_path,
                quiet=False
            )
        except:
            st.warning("Échec du téléchargement du modèle LSTM")
    
    if os.path.exists(lstm_path):
        try:
            from tensorflow.keras.models import load_model
            models['lstm'] = load_model(lstm_path)
        except:
            st.warning("Erreur de chargement du modèle LSTM")
    
    return models

# Charger les modèles
with st.spinner('Chargement des modèles...'):
    models = load_models()
    if not models:
        st.error("Aucun modèle chargé! Vérifiez les fichiers de modèles.")
        st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres de prédiction")
    date = st.date_input("Date", datetime.today() + timedelta(days=1))
    start_hour = st.slider("Heure début", 0, 23, 14)
    end_hour = st.slider("Heure fin", 0, 23, 16)
    
    st.divider()
    st.header("🔧 Options avancées")
    available_models = ['LightGBM']
    if 'lstm' in models:
        available_models.append('LSTM')
    
    model_choice = st.selectbox(
        "Modèle de prédiction",
        available_models,
        index=0
    )
    show_details = st.toggle("Afficher les détails", True)
    
    st.divider()
    st.header("🔄 Comparaison de modèles")
    compare_models = st.toggle("Comparer plusieurs modèles", False)
    if compare_models and len(available_models) > 1:
        selected_models = st.multiselect(
            "Modèles à comparer",
            options=available_models,
            default=available_models
        )
    
    st.divider()
    st.header("📅 Historique")
    show_history = st.toggle("Afficher l'historique", False)
    
    st.divider()
    if st.button("🔁 Actualiser les données", use_container_width=True):
        with st.spinner("Mise à jour des données en cours..."):
            if data_utils.update_data():
                st.success("Données mises à jour avec succès!")
                # Recharger les modèles
                st.cache_resource.clear()
                models = load_models()
            else:
                st.error("Échec de la mise à jour des données")

# Fonction de prédiction
def predict_games(date, start_hour, end_hour, model_choice):
    # Générer les parties
    games = data_utils.generate_games(date, start_hour, end_hour)
    
    if games.empty:
        st.error("Aucune partie générée! Vérifiez les paramètres.")
        return pd.DataFrame()
    
    # Préparer les features
    features = ['Hour', 'Minute', 'Day_of_week', 'Month', 'Player_Avg', 'Banker_Avg', 'Rolling_Player_Win']
    missing_features = [f for f in features if f not in games.columns]
    
    if missing_features:
        st.error(f"Features manquantes: {', '.join(missing_features)}")
        return pd.DataFrame()
    
    # Sélection du modèle
    model = models.get(model_choice.lower())
    if not model:
        st.error(f"Modèle {model_choice} non disponible!")
        return pd.DataFrame()
    
    # Prédiction
    try:
        if model_choice == 'LightGBM':
            predictions = model.predict(games[features])
        else:  # LSTM
            features_reshaped = games[features].values.reshape((len(games), 1, len(features)))
            predictions = model.predict(features_reshaped).argmax(axis=1)
    except Exception as e:
        st.error(f"Erreur de prédiction: {str(e)}")
        return pd.DataFrame()
    
    games['Prediction'] = predictions
    games['Winner'] = games['Prediction'].map({0: 'Joueur', 1: 'Banquier', 2: 'Egalite'})
    
    return games

# Bouton principal
if st.button("🎲 Générer les prédictions", use_container_width=True, type="primary"):
    with st.spinner('Calcul des prédictions...'):
        results = predict_games(date, start_hour, end_hour, model_choice)
        
        if results.empty:
            st.stop()
        
        # Sauvegarder dans l'historique
        data_utils.save_prediction(results.copy())
        
        # Afficher les résultats
        st.subheader(f"📊 Prédictions pour le {date.strftime('%d/%m/%Y')} de {start_hour}H à {end_hour}H")
        
        # Statistiques globales
        col1, col2, col3 = st.columns(3)
        player_wins = len(results[results['Winner'] == 'Joueur'])
        banker_wins = len(results[results['Winner'] == 'Banquier'])
        ties = len(results[results['Winner'] == 'Egalite'])
        
        col1.metric("Victoires Joueur", player_wins, "parties")
        col2.metric("Victoires Banquier", banker_wins, "parties")
        col3.metric("Égalités", ties, "parties")
        
        # Visualisation
        st.bar_chart(results['Winner'].value_counts())
        
        # Détails des prédictions
        if show_details:
            st.subheader("🔍 Détails des prédictions")
            
            # Pagination
            page_size = 10
            total_pages = (len(results) // page_size) + (1 if len(results) % page_size > 0 else 0)
            page = st.number_input('Page', min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(results))
            
            # Afficher les résultats paginés
            for idx in range(start_idx, end_idx):
                row = results.iloc[idx]
                winner_class = ""
                if row['Winner'] == 'Joueur':
                    winner_class = "winner-player"
                elif row['Winner'] == 'Banquier':
                    winner_class = "winner-banker"
                else:
                    winner_class = "winner-tie"
                
                with st.container():
                    st.markdown(f"<div class='prediction-card'>", unsafe_allow_html=True)
                    
                    # En-tête avec résultat
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                        f"<div><strong>Partie #{row['GameId']}</strong> - {row['Time']}</div>"
                        f"<div class='{winner_class}'>{row['Winner']}</div></div>",
                        unsafe_allow_html=True
                    )
                    
                    # Détails
                    st.markdown(
                        f"<div style='margin-top:10px; font-size:0.9em; color:#555;'>"
                        f"Joueur: {row['Player_Avg']:.1f} pts | "
                        f"Banquier: {row['Banker_Avg']:.1f} pts</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # Télécharger les résultats
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Télécharger les prédictions (CSV)",
            data=csv,
            file_name=f"predictions_baccara_{date.strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True
        )

# === COMPARAISON DE MODÈLES ===
if compare_models and len(available_models) > 1 and st.sidebar.checkbox("Comparer les modèles", True):
    st.divider()
    st.subheader("🔄 Comparaison des prédictions entre modèles")
    
    # Générer les prédictions pour chaque modèle
    comparison_results = {}
    for model_name in selected_models:
        with st.spinner(f"Calcul avec {model_name}..."):
            results = predict_games(date, start_hour, end_hour, model_name)
            if not results.empty:
                comparison_results[model_name] = results
    
    if not comparison_results:
        st.warning("Aucun résultat de comparaison disponible")
        st.stop()
    
    # Créer un DataFrame comparatif
    comparison_df = pd.DataFrame()
    for model_name, results in comparison_results.items():
        comparison_df[model_name] = results['Winner']
    
    # Ajouter les GameId et Time
    first_model = list(comparison_results.keys())[0]
    comparison_df.insert(0, 'GameId', comparison_results[first_model]['GameId'])
    comparison_df.insert(1, 'Time', comparison_results[first_model]['Time'])
    
    # Calculer le consensus
    def get_consensus(row):
        values = row[2:].value_counts()
        if len(values) == 1:
            return values.index[0]
        if values.iloc[0] > values.iloc[1]:
            return values.index[0]
        return 'Indécis'
    
    comparison_df['Consensus'] = comparison_df.apply(
        lambda row: get_consensus(row), 
        axis=1
    )
    
    # Afficher les résultats comparatifs
    st.dataframe(comparison_df)
    
    # Métriques d'accord
    agreement_rate = (comparison_df['Consensus'] != 'Indécis').mean()
    st.metric("Taux d'accord entre modèles", f"{agreement_rate:.1%}")
    
    # Analyse des divergences
    st.subheader("📊 Analyse des divergences")
    divergences = comparison_df[comparison_df['Consensus'] == 'Indécis']
    
    if not divergences.empty:
        st.write(f"{len(divergences)} parties avec désaccord entre modèles")
        
        # Pour chaque divergence, afficher les détails
        for _, row in divergences.iterrows():
            with st.expander(f"Partie #{row['GameId']} - {row['Time']}"):
                st.write("**Prédictions:**")
                for model in selected_models:
                    st.write(f"- {model}: {row[model]}")
    else:
        st.success("Tous les modèles sont d'accord sur toutes les prédictions!")

# === DASHBOARD DE PERFORMANCE ===
if st.sidebar.checkbox("📈 Dashboard de performance", True):
    st.divider()
    st.subheader("📈 Dashboard de performance des modèles")
    
    try:
        # Charger les données pour les statistiques
        df = data_utils.load_data()
        
        # Métriques clés
        col1, col2, col3 = st.columns(3)
        col1.metric("Parties analysées", f"{len(df):,}")
        col2.metric("Jours couverts", f"{df['DateTime'].dt.date.nunique():,}")
        col3.metric("Taux d'égalité", f"{len(df[df['Winner'] == 'Egalite'])/len(df):.1%}")
        
        # Graphiques de performance
        tab1, tab2 = st.tabs(["Distribution des victoires", "Performance horaire"])
        
        with tab1:
            # Distribution des gagnants
            winner_dist = df['Winner'].value_counts(normalize=True)
            st.bar_chart(winner_dist)
            
        with tab2:
            # Performance par heure
            hourly_win_rate = df.groupby('Hour')['Winner'].value_counts(normalize=True).unstack()
            st.line_chart(hourly_win_rate)
        
        # Analyse des erreurs
        st.subheader("🔍 Analyse des résultats")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribution des scores Joueur:**")
            st.bar_chart(df['Joueur_Score'].value_counts().sort_index())
        
        with col2:
            st.markdown("**Distribution des scores Banquier:**")
            st.bar_chart(df['Banquier_Score'].value_counts().sort_index())
        
        # Recommandations
        st.info("💡 Conseil: Les prédictions sont plus précises entre 14h et 18h")
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")

# === HISTORIQUE DES PRÉDICTIONS ===
if show_history:
    st.divider()
    st.subheader("📅 Historique des prédictions")
    
    # Options de filtrage
    col1, col2 = st.columns(2)
    with col1:
        history_start = st.date_input("Date de début", datetime.today() - timedelta(days=7))
    with col2:
        history_end = st.date_input("Date de fin", datetime.today())
    
    # Charger l'historique
    try:
        history = data_utils.load_prediction_history(history_start, history_end)
        
        if history is None or history.empty:
            st.warning("Aucune donnée historique disponible pour cette période")
        else:
            # Statistiques globales
            st.write(f"**{len(history)} prédictions** entre {history_start.strftime('%d/%m/%Y')} et {history_end.strftime('%d/%m/%Y')}")
            
            # KPI
            player_wins = (history['Winner'] == 'Joueur').sum()
            banker_wins = (history['Winner'] == 'Banquier').sum()
            draws = (history['Winner'] == 'Egalite').sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Victoires Joueur", player_wins, f"{player_wins/len(history):.1%}")
            col2.metric("Victoires Banquier", banker_wins, f"{banker_wins/len(history):.1%}")
            col3.metric("Égalités", draws, f"{draws/len(history):.1%}")
            
            # Graphique d'évolution
            st.subheader("Évolution des résultats")
            history['Date'] = pd.to_datetime(history['Date'])
            daily_trend = history.groupby(history['Date'].dt.date)['Winner'].value_counts().unstack().fillna(0)
            st.area_chart(daily_trend)
            
            # Détails des prédictions
            st.subheader("Détails des prédictions")
            
            # Filtres supplémentaires
            col1, col2 = st.columns(2)
            with col1:
                selected_winner = st.selectbox("Filtrer par résultat", ['Tous', 'Joueur', 'Banquier', 'Egalite'])
            with col2:
                show_full_details = st.checkbox("Afficher les détails complets", False)
            
            # Application des filtres
            if selected_winner != 'Tous':
                filtered_history = history[history['Winner'] == selected_winner]
            else:
                filtered_history = history
            
            # Affichage
            if show_full_details:
                st.dataframe(filtered_history)
            else:
                st.dataframe(filtered_history[['GameId', 'Date', 'Time', 'Winner']])
            
            # Téléchargement
            csv = filtered_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Télécharger l'historique filtré",
                data=csv,
                file_name=f"historique_baccara_{history_start}_{history_end}.csv",
                mime='text/csv',
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Erreur de chargement de l'historique: {str(e)}")

# Footer
st.divider()
st.caption("© 2025 Baccara Predictor Pro | Mise à jour quotidienne automatique")
