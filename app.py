import os
import json
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import xgboost, sklearn
from oracle import fetch_oracle_labels,fetch_mpd_labels,fetch_table_structure_by_mpd, set_owner_for_mpd

# =========================
# ÉTAT SESSION (confirmations utilisateur)
# =========================
if 'temp_confirmed_types' not in st.session_state:
    st.session_state.temp_confirmed_types = {}
if 'confirmed_types' not in st.session_state:
    st.session_state.confirmed_types = {}

# =========================
# REGEX (IDENTIQUES AU TRAINING)
# =========================
RE_NUMERIC = re.compile(r'NUMBER\(\d+(,\d+)?\)|INTEGER|DECIMAL|FLOAT|NUMERIC|BIGINT', re.I)
RE_TEXT    = re.compile(r'CHAR\(\d+\)|VARCHAR2\(\d+\)|VARCHAR|TEXT|STRING', re.I)
RE_DATE    = re.compile(r'DATE|TIMESTAMP|TIME', re.I)
RE_KEYS    = re.compile(r'ID|CODE|NUM|KEY|REF|LIBL', re.I)
RE_KEYS_RELATION = re.compile(r'CODE|TYPE|CODE TYPE', re.I)
RE_MEAS    = re.compile(r'MONTANT|AMOUNT|VALEUR|VALUE|PRICE|PRIX|QTE|QTY|COUNT|TOTAL|SUM|MT_|SOLD_|VALR_', re.I)
RE_CODE_PREFIX = re.compile(r'^CODE_|^TYPE_', re.I)
RE_FACT_PREFIX = re.compile(r'FACT|MESURE|FLUX|TRANSACTION|EVNM|MVT_|ECRT_|BALN_', re.I)

# =========================
# HELPERS de normalisation (pour fiabiliser les matches)
# =========================
def _norm(s: str) -> str:
    return str(s).replace('\u00A0', ' ').strip().upper()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'LIBELLE_DU_SEGMENT' in df.columns:
        df['LIBELLE_DU_SEGMENT'] = df['LIBELLE_DU_SEGMENT'].map(_norm)
    if 'NOM_EPURE_DE_LA_RUBRIQUE' in df.columns:
        df['NOM_EPURE_DE_LA_RUBRIQUE'] = df['NOM_EPURE_DE_LA_RUBRIQUE'].map(_norm)
    if 'NOM_DE_LA_COLONNE_NORME_ADD' in df.columns:
        df['NOM_DE_LA_COLONNE_NORME_ADD'] = df['NOM_DE_LA_COLONNE_NORME_ADD'].map(_norm)
    for alt in ['Nom', 'Code']:
        if alt in df.columns:
            df[alt] = df[alt].map(_norm)
    return df

# =========================
# CHARGEMENT MODÈLES / FEATURES
# =========================
@st.cache_resource
def load_xgb_model():
    xgb_model = XGBClassifier()
    xgb_model.load_model('best_xgb_model.json')
    return xgb_model

xgb_model = load_xgb_model()

@st.cache_resource
def load_model_features():
    """
    Source de vérité : best_xgb_features.json sauvegardé à l'entraînement.
    Fallbacks: training_features_by_table.csv puis feature_names_in_ si dispo.
    """
    if os.path.exists('best_xgb_features.json'):
        with open('best_xgb_features.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    if os.path.exists('training_features_by_table.csv'):
        base = pd.read_csv('training_features_by_table.csv')
        cols = [c for c in base.columns if c not in ('Table_Name', 'Table_Type')]
        return cols

    if hasattr(xgb_model, 'feature_names_in_'):
        return list(xgb_model.feature_names_in_)

    raise RuntimeError(
        "Impossible de retrouver les features du modèle. "
        "Assure-toi d'avoir 'best_xgb_features.json' ou 'training_features_by_table.csv'."
    )

# =========================
# UTILS : RELATIONS FACT-DIM
# =========================
def detect_fact_dim_links(original_df, type_predictions_df, prob_diff_threshold=0.2):
    df = normalize_df(original_df)
    type_map = dict(zip(
        type_predictions_df['Table_Name'].map(_norm),
        type_predictions_df['Type Prédit']
    ))
    candidate_name_cols = []
    for c in ['NOM_EPURE_DE_LA_RUBRIQUE', 'NOM_DE_LA_COLONNE_NORME_ADD', 'Nom', 'Code']:
        if c in df.columns:
            candidate_name_cols.append(c)
    if not candidate_name_cols:
        candidate_name_cols = ['NOM_EPURE_DE_LA_RUBRIQUE']
    all_tables = df['LIBELLE_DU_SEGMENT'].unique()

    def cols_for_table_key_relation(t):
        subset = df[df['LIBELLE_DU_SEGMENT'] == t]
        key_cols = set()
        for c in candidate_name_cols:
            key_cols |= set(
                col for col in subset[c].dropna() if RE_KEYS_RELATION.search(str(col))
            )
        key_cols_norm = set(_norm(col) for col in key_cols)
        return key_cols, key_cols_norm

    table_columns = {t: cols_for_table_key_relation(t) for t in all_tables}
    fact_tables = [t for t in all_tables if type_map.get(_norm(t)) == 'FAIT']
    dim_tables  = [t for t in all_tables if type_map.get(_norm(t)) == 'DIMENSION']

    EXCLUDE_COLS = {"PERD_ARRT_INFO", "CODE_ORGN_FINN"}
    relations = []
    for fact in fact_tables:
        fact_cols, fact_cols_norm = table_columns.get(fact, (set(), set()))
        for dim in dim_tables:
            dim_cols, dim_cols_norm = table_columns.get(dim, (set(), set()))
            # Intersection des colonnes clés relationnelles, hors colonnes à exclure
            common_cols = sorted([col for col in (fact_cols & dim_cols) if col not in EXCLUDE_COLS])
            if common_cols:
                relations.append({
                    'Table_Fact': fact,
                    'Table_Dimension': dim,
                    'Colonnes_Communes': ', '.join(common_cols),
                    'Nb_Colonnes_Communes': len(common_cols)
                })
    return pd.DataFrame(relations)

# =========================
# FEATURES LOCALES (mêmes règles que training)
# =========================
def has_code_prefix(table_name):
    return 1 if RE_CODE_PREFIX.search(str(table_name)) else 0

def has_measure_prefix(table_name):
    return 1 if RE_FACT_PREFIX.search(str(table_name)) else 0

def count_key_like_columns(column_names):
    return sum(1 for c in column_names if RE_KEYS.search(str(c)))

def count_measure_like_columns(column_names):
    return sum(1 for c in column_names if RE_MEAS.search(str(c)))

# =========================
# PRÉTRAITEMENT EXCEL (aligné sur training + normalisation)
# =========================
def preprocess_excel_file(file):
    """
    Construit les features EXACTEMENT comme au training.
    Seules colonnes nécessaires : LIBELLE_DU_SEGMENT, NOM_EPURE_DE_LA_RUBRIQUE, TYPE_DONNEES_COLONNE, PK.
    'Table_Type' est optionnelle (pour calculer la précision si présente).
    """
    try:
        df = pd.read_excel(file)
        df = normalize_df(df)

        required_columns = [
            'LIBELLE_DU_SEGMENT',
            'NOM_EPURE_DE_LA_RUBRIQUE',
            'TYPE_DONNEES_COLONNE',
            'PK'
        ]
        missing = [c for c in required_columns if c not in df.columns]
        with rel_sections[1]:
            with st.expander("Chercher les colonnes communes entre vos FAITS et toutes les tables ODS", expanded=True):
                c1, c2, c3 = st.columns([1,1,2], gap="small")
                with c1:
                    only_keylike = st.checkbox("Limiter aux colonnes type clé (CODE/ID/REF...)", value=True, key="ods_keylike")
                with c2:
                    min_common = st.number_input("Nb min de colonnes communes", min_value=1, max_value=20, value=1, step=1, key="ods_min_common")
                with c3:
                    excl = st.text_input("Colonnes à exclure (séparées par des virgules)", "PERD_ARRT_INFO,CODE_ORGN_FINN", key="ods_excl")

                run_ods = st.button("Lancer la recherche ODS", key="ods_run")
                if run_ods:
                    try:
                        ods_df = fetch_external_ods_relations()
                        exclude_cols = {_norm(x) for x in (excl.split(",") if excl else []) if x.strip()}
                        rel_ods = detect_fact_vs_ods_links(
                            original_df=original_df,
                            type_predictions_df=results,
                            ods_df=ods_df,
                            only_keylike=only_keylike,
                            min_common=int(min_common),
                            exclude_cols=exclude_cols if exclude_cols else None
                        )
                        if rel_ods.empty:
                            st.info("Aucune relation FAIT ↔ ODS trouvée.")
                        else:
                            st.success(f"{len(rel_ods)} correspondances trouvées.")
                            cols_order = [
                                'Table_Fact', 'ODS_Table',
                                'Nb_Colonnes_Communes', 'Colonnes_Communes',
            
                            ]
                            display_df = rel_ods.loc[:, [c for c in cols_order if c in rel_ods.columns]]
                            st.dataframe(
                                display_df.sort_values(['Table_Fact', 'Nb_Colonnes_Communes'], ascending=[True, False]),
                                use_container_width=True
                            )
                            csv = rel_ods.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Télécharger FAIT↔ODS (CSV)",
                                data=csv,
                                file_name="fact_vers_ods_correspondances.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Erreur ODS : {e}")

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier Excel : {str(e)}")
        return None, None

def preprocess_df(df):
    """
    Même logique que preprocess_excel_file mais prend un DataFrame en entrée.
    """
    df = normalize_df(df)
    required_columns = [
        'LIBELLE_DU_SEGMENT',
        'NOM_EPURE_DE_LA_RUBRIQUE',
        'TYPE_DONNEES_COLONNE',
        'PK'
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {missing}")
        return None, None

    agg_dict = {
        'NOM_EPURE_DE_LA_RUBRIQUE': ['count', lambda x: [str(v).upper() for v in x.fillna('')]],
        'TYPE_DONNEES_COLONNE': lambda x: (
            sum(1 for t in x if RE_NUMERIC.search(str(t))),
            sum(1 for t in x if RE_TEXT.search(str(t))),
            sum(1 for t in x if RE_DATE.search(str(t)))
        ),
        'PK': lambda x: sum(1 for p in x if str(p).strip().upper() == 'O')
    }

    has_table_type = 'Table_Type' in df.columns
    if has_table_type:
        agg_dict['Table_Type'] = 'first'

    grouped = df.groupby('LIBELLE_DU_SEGMENT').agg(agg_dict).reset_index()

    expected_cols = ['Table_Name', 'Num_Columns', 'Column_Names', 'Type_Counts', 'Num_PKs']
    if has_table_type:
        expected_cols.append('Table_Type')
    grouped.columns = expected_cols

    grouped['Num_Columns'] = grouped['Num_Columns'].fillna(0).replace(0, 1)
    grouped['Num_Numeric'] = grouped['Type_Counts'].apply(lambda x: x[0]).fillna(0)
    grouped['Num_Text']    = grouped['Type_Counts'].apply(lambda x: x[1]).fillna(0)
    grouped['Num_Date']    = grouped['Type_Counts'].apply(lambda x: x[2]).fillna(0)
    grouped['Num_PKs']     = grouped['Num_PKs'].fillna(0).astype(int)

    grouped['Pct_Numeric'] = grouped['Num_Numeric'] / grouped['Num_Columns']
    grouped['Pct_Text']    = grouped['Num_Text']    / grouped['Num_Columns']
    grouped['Pct_Date']    = grouped['Num_Date']    / grouped['Num_Columns']
    grouped['PK_Ratio']    = grouped['Num_PKs']     / grouped['Num_Columns']

    grouped['Has_Code_Prefix']    = grouped['Table_Name'].apply(has_code_prefix)
    grouped['Has_Measure_Prefix'] = grouped['Table_Name'].apply(has_measure_prefix)

    grouped['Num_Key_Like']     = grouped['Column_Names'].apply(count_key_like_columns)
    grouped['Num_Measure_Like'] = grouped['Column_Names'].apply(count_measure_like_columns)

    grouped = grouped.drop(columns=['Type_Counts'])
    grouped = grouped.fillna(0)

    return grouped, df


st.title("Prédiction de type de table (FAIT ou DIMENSION)")

st.write("Choisissez le mode de test :")
# Utilisation d'un switch/toggle pour le choix du mode
if 'mode' not in st.session_state:
    st.session_state.mode = "Sélectionner un MPD Oracle"

toggle = st.toggle(
    label="Mode MPD Oracle / Fichier Excel",
    value=(st.session_state.mode == "Sélectionner un MPD Oracle"),
    help="Activez pour MPD Oracle, désactivez pour Fichier Excel."
)
if toggle:
    st.session_state.mode = "Sélectionner un MPD Oracle"
else:
    st.session_state.mode = "Charger un fichier Excel"
mode = st.session_state.mode

prob_diff_threshold = st.slider(
    "Seuil de différence de probabilité pour considérer une table comme incertaine",
    min_value=0.0,
    max_value=0.5,
    value=0.15,
    step=0.05,
    help="Si |P(FACT) - P(DIM)| < seuil, la table est considérée comme incertaine."
)

grouped, original_df = None, None
results, predictions, predictions_proba = None, None, None


if mode == "Sélectionner un MPD Oracle":
    st.header("Tester avec un MPD Oracle")
    mpd_labels = fetch_mpd_labels()
    selected_mpd = st.selectbox(
        "Sélectionner un MPD (modèle de données) :",
        options=mpd_labels,
        index=None,
        placeholder="Commencez à saisir le libellé..."
    )
    if selected_mpd:
        owner = set_owner_for_mpd(selected_mpd)
        df_mpd = fetch_table_structure_by_mpd(selected_mpd, owner)
        grouped, original_df = preprocess_df(df_mpd)

elif mode == "Charger un fichier Excel":
    st.header("Tester avec un fichier Excel")
    uploaded_file = st.file_uploader("Charger un fichier Excel", type=["xlsx", "xls"])
    if uploaded_file:
        grouped, original_df = preprocess_excel_file(uploaded_file)

# === TRAITEMENT ET AFFICHAGE COMMUN ===
if grouped is not None:
    if grouped['Table_Name'].duplicated().any():
        st.warning("Attention : Certains noms de tables sont dupliqués.")

    model_features = [c for c in load_model_features() if c != 'Table_Type']

    X_test = grouped.copy()
    for col in ['Table_Name', 'Table_Type', 'Column_Names']:
        if col in X_test.columns:
            X_test = X_test.drop(columns=col)

    for col in model_features:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[model_features].astype('float32')

    predictions = xgb_model.predict(X_test)
    predictions_proba = xgb_model.predict_proba(X_test)

    results = pd.DataFrame({
        'Table_Name': grouped['Table_Name'],
        'Probabilité FACT': predictions_proba[:, 1],
        'Probabilité DIMENSION': predictions_proba[:, 0],
        'Type Prédit': np.where(predictions == 1, 'FAIT', 'DIMENSION'),
        'Confiance': np.maximum(predictions_proba[:, 1], predictions_proba[:, 0]),
        'Confirmed': False
    })

    for table, typ in st.session_state.confirmed_types.items():
        mask = results['Table_Name'] == table
        if mask.any():
            results.loc[mask, 'Type Prédit'] = typ
            results.loc[mask, 'Confirmed'] = True

    if 'Table_Type' in grouped.columns:
        results['Vrai Type'] = grouped['Table_Type'].map({1: 'FAIT', 0: 'DIMENSION'})

    incertains_mask = (
        (np.abs(results['Probabilité FACT'] - results['Probabilité DIMENSION']) < prob_diff_threshold)
        & (~results['Confirmed'])
    )
    incertains = results[incertains_mask].copy()

    st.subheader("Résultats des prédictions")

    def color_row(row):
            # On récupère la valeur Confirmed depuis results
            # Si la table est confirmée (expert ou auto), on affiche en vert
            table_name = row['Table_Name'] if 'Table_Name' in row else None
            if table_name is not None and 'Confirmed' in results.columns:
                confirmed = results.loc[results['Table_Name'] == table_name, 'Confirmed'].values
                if len(confirmed) > 0 and confirmed[0]:
                    return ['background-color: #d4edda'] * len(row)
            # Sinon, logique habituelle
            diff = abs(row['Probabilité FACT'] - row['Probabilité DIMENSION'])
            if diff >= prob_diff_threshold:
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)

    # On garde la colonne Confirmed pour la coloration
    results_display = results.copy().drop(columns=['Confirmed'])
    styled_results = results_display.style.apply(color_row, axis=1).format({
        'Probabilité FACT': '{:.3f}',
        'Probabilité DIMENSION': '{:.3f}',
        'Confiance': '{:.3f}'
    })
    st.dataframe(styled_results, use_container_width=True)

    if not incertains.empty:
        st.subheader("Cas incertains à valider par un expert")
        if mode == "Sélectionner un MPD Oracle":
            oracle_labels = fetch_oracle_labels(list(incertains["Table_Name"]))
        else:
            oracle_labels = {}

        def _preview_cols(table_name: str) -> str:
            cols_arr = grouped.loc[grouped['Table_Name'] == table_name, 'Column_Names'].values
            if len(cols_arr) == 0 or not cols_arr[0]:
                return "Aucune colonne disponible"
            lst = list(cols_arr[0])
            preview = '\n'.join([f"• {col}" for col in lst])
            return preview

        inc_view = pd.DataFrame({
            "Nom table": incertains["Table_Name"].values,
            "P(FAIT)": incertains["Probabilité FACT"].round(3).values,
            "P(DIM)": incertains["Probabilité DIMENSION"].round(3).values,
        })

        default_types = []
        for t in incertains["Table_Name"]:
            if t in st.session_state.confirmed_types:
                default_types.append(st.session_state.confirmed_types[t])
            else:
                default_types.append(results.loc[results["Table_Name"] == t, "Type Prédit"].iloc[0])
        inc_view["Type proposé"] = default_types

        inc_view["Colonnes"] = [_preview_cols(t) for t in incertains["Table_Name"]]

        inc_view["Note"] = [
            "\n".join(oracle_labels.get(t, [])) if t in oracle_labels else "Aucune note trouvée"
            for t in incertains["Table_Name"]
        ]

        q = st.text_input("Filtrer par nom de table", "")
        filtered = inc_view.copy()
        if q:
            filtered = filtered[filtered["Nom table"].str.contains(q, case=False, na=False)]

        st.markdown("""
            <style>
            .stDataFrame [data-testid="stTable"] td {
                white-space: pre-wrap !important;
                max-width: 300px;
                word-wrap: break-word;
            }
            </style>
        """, unsafe_allow_html=True)

        with st.form("form_incertains"):
            edited = st.data_editor(
                filtered,
                hide_index=True,
                use_container_width=True,
                disabled=["Nom table", "P(FAIT)", "P(DIM)", "Colonnes", "Note"],
                column_config={
                    "P(FAIT)": st.column_config.NumberColumn(
                        "P(FAIT)", format="%.3f",
                        help="Probabilité prédite que la table soit un FAIT."
                    ),
                    "P(DIM)": st.column_config.NumberColumn(
                        "P(DIM)", format="%.3f",
                        help="Probabilité prédite que la table soit une DIMENSION."
                    ),
                    "Type proposé": st.column_config.SelectboxColumn(
                        "Type proposé",
                        options=["DIMENSION", "FAIT"],
                        help="Corrigez le type si besoin."
                    ),
                    "Colonnes": st.column_config.TextColumn(
                        "Colonnes",
                        width="large",
                        help="Aperçu des colonnes de la table (beaucoup plus complet, tronqué si >200)."
                    ),
                    "Note": st.column_config.TextColumn(
                        "Note",
                        width="large",
                        help="Note associée à la table (issue d'Oracle)."
                    ),
                },
            )
            submitted = st.form_submit_button("Valider les modifications")
            if submitted:
                for _, r in edited.iterrows():
                    st.session_state.confirmed_types[r["Nom table"]] = r["Type proposé"]
                st.success("Modifications validées !")
                st.rerun()
    else:
        st.info("Aucun cas incertain à valider.")

    if 'Table_Type' in grouped.columns and not grouped['Table_Type'].isna().all():
        accuracy = accuracy_score(
            grouped['Table_Type'].map({1: 1, 0: 0}),
            results['Type Prédit'].map({'FAIT': 1, 'DIMENSION': 0})
        )
        st.write(f"**Précision sur la source sélectionnée :** {accuracy:.4f}")

    relations_df = detect_fact_dim_links(original_df, results, prob_diff_threshold=prob_diff_threshold)
    st.subheader("Relations détectées entre tables de faits et dimensions")
    if relations_df.empty:

        st.info("Aucune relation détectée.")
    else:
        # Fusion des vérifications dans le même tableau
        def check_keys_in_fact_dim(original_df, relations_df):
            results = []
            df = normalize_df(original_df)
            for _, rel in relations_df.iterrows():
                fact = rel['Table_Fact']
                dim = rel['Table_Dimension']
                # Colonnes de la dimension
                dim_cols = df[df['LIBELLE_DU_SEGMENT'] == dim]['NOM_EPURE_DE_LA_RUBRIQUE'].dropna().map(_norm).tolist()
                # Colonnes de la table de fait
                fact_cols = df[df['LIBELLE_DU_SEGMENT'] == fact]['NOM_EPURE_DE_LA_RUBRIQUE'].dropna().map(_norm).tolist()
                # Clés relationnelles attendues
                keys_relation = [c for c in dim_cols if RE_KEYS_RELATION.search(c)]
                missing_keys_relation = [c for c in keys_relation if c not in fact_cols]
                # Clés primaires de la dimension
                pk_dim = df[(df['LIBELLE_DU_SEGMENT'] == dim) & (df['PK'].map(_norm) == 'O')]['NOM_EPURE_DE_LA_RUBRIQUE'].dropna().map(_norm).tolist()
                missing_pk_dim = [c for c in pk_dim if c not in fact_cols]
                results.append({
                    'Clés relationnelles manquantes': ', '.join(missing_keys_relation) if missing_keys_relation else 'Aucune',
                    'PK de la dimension manquantes': ', '.join(missing_pk_dim) if missing_pk_dim else 'Aucune'
                })
            return results

        # Ajout des colonnes de vérification au DataFrame des relations
        verif_cols = check_keys_in_fact_dim(original_df, relations_df)
        relations_df_ext = relations_df.copy()
        relations_df_ext['Clés relationnelles manquantes'] = [v['Clés relationnelles manquantes'] for v in verif_cols]
        relations_df_ext['PK de la dimension manquantes'] = [v['PK de la dimension manquantes'] for v in verif_cols]
        # st.subheader("Relations détectées entre tables de faits et dimensions (avec vérification des clés)")
        st.dataframe(relations_df_ext, use_container_width=True)

    # =========================
    # Correspondances FAIT ↔ ODS
    # =========================
    from oracle import fetch_external_ods_relations
    ods_df = fetch_external_ods_relations()

    def _columns_by_table_from_original_df(original_df: pd.DataFrame) -> dict:
        df = original_df.copy()
        df['LIBELLE_DU_SEGMENT'] = df['LIBELLE_DU_SEGMENT'].map(_norm)
        df['NOM_EPURE_DE_LA_RUBRIQUE'] = df['NOM_EPURE_DE_LA_RUBRIQUE'].map(_norm)
        table_columns = {}
        for t, sub in df.groupby('LIBELLE_DU_SEGMENT'):
            raw_set = set(sub['NOM_EPURE_DE_LA_RUBRIQUE'].dropna())
            norm_set = set(raw_set)
            norm2raw = {c: c for c in raw_set}
            table_columns[t] = {
                'raw': raw_set,
                'norm': norm_set,
                'norm2raw': norm2raw
            }
        return table_columns

    def _columns_by_table_from_ods_df(ods_df: pd.DataFrame) -> dict:
        df = ods_df.copy()
        df['ODS_TABLE_NAME']  = df['ODS_TABLE_NAME'].map(_norm)
        df['ODS_COLUMN_NAME'] = df['ODS_COLUMN_NAME'].map(_norm)
        table_columns = {}
        for t, sub in df.groupby('ODS_TABLE_NAME'):
            raw_set = set(sub['ODS_COLUMN_NAME'].dropna())
            norm_set = set(raw_set)
            norm2raw = {c: c for c in raw_set}
            table_columns[t] = {
                'raw': raw_set,
                'norm': norm_set,
                'norm2raw': norm2raw
            }
        return table_columns

    def detect_fact_vs_ods_links(original_df, type_predictions_df, ods_df, only_keylike=False, min_common=1, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = {"PERD_ARRT_INFO", "CODE_ORGN_FINN"}
        base_cols = _columns_by_table_from_original_df(original_df)
        ods_cols  = _columns_by_table_from_ods_df(ods_df)
        type_map = dict(zip(
            type_predictions_df['Table_Name'].map(_norm),
            type_predictions_df['Type Prédit']
        ))
        fact_tables = [t for t in base_cols.keys() if type_map.get(_norm(t)) == 'FAIT']
        out = []
        for fact in fact_tables:
            info_fact = base_cols.get(fact, {'raw': set(), 'norm': set(), 'norm2raw': {}})
            fact_norm = info_fact['norm']
            # Toujours filtrer sur les colonnes clés côté FAIT
            fact_filtered = {c for c in fact_norm if (RE_KEYS.search(c) or RE_KEYS_RELATION.search(c))}
            fact_filtered = {c for c in fact_filtered if c not in exclude_cols}
            if not fact_filtered:
                continue
            for ods_table, info_ods in ods_cols.items():
                ods_norm = info_ods['norm']
                # Toujours filtrer sur les colonnes clés côté ODS
                ods_filtered = {c for c in ods_norm if (RE_KEYS.search(c) or RE_KEYS_RELATION.search(c))}
                ods_filtered = {c for c in ods_filtered if c not in exclude_cols}
                commons = sorted(fact_filtered & ods_filtered)
                if len(commons) >= min_common:
                    fact_raw_cols = [info_fact['norm2raw'].get(c, c) for c in commons]
                    ods_raw_cols  = [info_ods['norm2raw'].get(c, c) for c in commons]
                    out.append({
                        'Table_Fact': fact,
                        'ODS_Table': ods_table,
                        'Colonnes_Communes': ', '.join(commons),
                        'Nb_Colonnes_Communes': len(commons),
                        'Colonnes_Fact': ', '.join(fact_raw_cols),
                        'ODS_Colonnes': ', '.join(ods_raw_cols),
                    })
        return pd.DataFrame(out)

    ods_links_df = detect_fact_vs_ods_links(original_df, results, ods_df)
    st.subheader("Correspondances entre tables de faits et ODS")
    if ods_links_df.empty:
        st.info("Aucune correspondance FAIT↔ODS détectée.")
    else:
        st.dataframe(ods_links_df, use_container_width=True)

    modifications_data = []
    for table, typ in st.session_state.confirmed_types.items():
        result_mask = results['Table_Name'] == table
        if not result_mask.any():
            continue
        grouped_mask = grouped['Table_Name'] == table
        if not grouped_mask.any():
            continue
        idx = grouped[grouped_mask].index[0]
        original_prediction = 'FAIT' if predictions[idx] == 1 else 'DIMENSION'
        if typ != original_prediction:
            modifications_data.append({
                'Table_Name': table,
                'Type Initial': original_prediction,
                'Type Corrigé': typ
            })
    modifications_df = pd.DataFrame(modifications_data)

    # Export Excel : structure identique à la source + colonne de prédiction
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # On part du DataFrame source original_df
        export_df = original_df.copy()
        # Ajout de la colonne de prédiction
        type_map = dict(zip(results['Table_Name'].map(_norm), results['Type Prédit']))
        export_df['TYPE_DIMENSIONNEL_TABLE'] = export_df['LIBELLE_DU_SEGMENT'].map(type_map).fillna('UNKNOWN')
        # On place la colonne de prédiction à la fin
        cols = [c for c in export_df.columns if c != 'TYPE_DIMENSIONNEL_TABLE'] + ['TYPE_DIMENSIONNEL_TABLE']
        export_df = export_df.loc[:, cols]
        export_df.to_excel(writer, index=False, sheet_name='Structure+Prédiction')
    output.seek(0)
    st.download_button(
        label="Télécharger la structure + prédiction (Excel)",
        data=output,
        file_name="structure_et_prediction.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

