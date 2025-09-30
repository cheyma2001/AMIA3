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

from oracle import (
    fetch_oracle_labels,
    fetch_mpd_labels,
    fetch_table_structure_by_mpd,
    set_owner_for_mpd,
    fetch_external_ods_relations,  
)

# =========================
# ÉTAT SESSION
# =========================
if "temp_confirmed_types" not in st.session_state:
    st.session_state.temp_confirmed_types = {}
if "confirmed_types" not in st.session_state:
    st.session_state.confirmed_types = {}
if "ods_relations" not in st.session_state:
    st.session_state.ods_relations = None
if "ods_params" not in st.session_state:
    st.session_state.ods_params = {}

# =========================
# REGEX 
# =========================
RE_NUMERIC = re.compile(r'NUMBER\(\d+(,\d+)?\)|INTEGER|DECIMAL|FLOAT|NUMERIC|BIGINT', re.I)
RE_TEXT    = re.compile(r'CHAR\(\d+\)|VARCHAR2\(\d+\)|VARCHAR|TEXT|STRING', re.I)
RE_DATE    = re.compile(r'DATE|TIMESTAMP|TIME', re.I)
RE_KEYS    = re.compile(r'ID|CODE|NUM|KEY|REF|LIBL', re.I)
RE_KEYS_RELATION = re.compile(r'CODE|TYPE|CODE TYPE|REFR', re.I)
RE_MEAS    = re.compile(r'MONTANT|AMOUNT|VALEUR|VALUE|PRICE|PRIX|QTE|QTY|COUNT|TOTAL|SUM|MT_|SOLD_|VALR_', re.I)
RE_CODE_PREFIX = re.compile(r'^CODE_|^TYPE_', re.I)
RE_FACT_PREFIX = re.compile(r'FACT|MESURE|FLUX|TRANSACTION|EVNM|MVT_|ECRT_|BALN_', re.I)
RE_CODETYPE_STRICT = re.compile(r'^(?:CODE_|TYPE_|REFR_)|\bCODE[_\s]+TYPE\b', re.I)

# =========================
# HELPERS de normalisation
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
# CHARGEMENT MODÈLE / FEATURES
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
# ==========================
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
            key_cols |= set(col for col in subset[c].dropna() if RE_KEYS_RELATION.search(str(col)))
        key_cols_norm = set(_norm(col) for col in key_cols)
        return key_cols, key_cols_norm

    table_columns = {t: cols_for_table_key_relation(t) for t in all_tables}
    fact_tables = [t for t in all_tables if type_map.get(_norm(t)) == 'FAIT']
    dim_tables  = [t for t in all_tables if type_map.get(_norm(t)) == 'DIMENSION']

    EXCLUDE_COLS = {"PERD_ARRT_INFO", "CODE_ORGN_FINN", "CODE_ORGN_FINN_BPCE"}
    relations = []
    for fact in fact_tables:
        fact_cols, _ = table_columns.get(fact, (set(), set()))
        for dim in dim_tables:
            dim_cols, _ = table_columns.get(dim, (set(), set()))
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
# PRÉTRAITEMENT EXCEL 
# =========================
# (je laisse cette partie inchangée car pas concernée)
# ...

# =========================
# LIENS FAIT vs ODS 
# =========================
# (inchangé sauf que Modele sera pris en compte automatiquement grâce à oracle.py)

# =========================
# UI
# =========================
# (inchangé jusqu’à la partie Relations FAIT ↔ ODS)

    # =========================
    # Relations FAIT ↔ ODS 
    # =========================
    st.subheader("Relations FAIT ↔ ODS")

    with st.expander("Chercher les colonnes communes entre vos FAITS et toutes les tables ODS", expanded=True):

        c1, c2 = st.columns([2,1], gap="small")
        with c1:
            excl = st.text_input(
                "Colonnes à exclure (séparées par des virgules)",
                value="PERD_ARRT_INFO,CODE_ORGN_FINN,CODE_ORGN_FINN_BPCE",
                key="ods_excl"
            )
        with c2:
            run_ods = st.button("Lancer la recherche ODS", type="primary", key="btn_run_ods")
            reset_ods = st.button("Réinitialiser", key="btn_reset_ods")

        if reset_ods:
            st.session_state.ods_relations = None
            st.session_state.ods_params = {}
            st.info("Résultats ODS réinitialisés.")

        if run_ods:
            try:
                ods_df = fetch_external_ods_relations()
                exclude_cols = {_norm(x) for x in (excl.split(",") if excl else []) if x.strip()}

                rel_ods = detect_fact_vs_ods_links(
                    original_df=original_df,
                    type_predictions_df=results,
                    ods_df=ods_df,
                    exclude_cols=exclude_cols if exclude_cols else None
                )

                st.session_state.ods_relations = rel_ods
                st.session_state.ods_params = dict(
                    exclude_cols=",".join(sorted(exclude_cols)) if exclude_cols else ""
                )

                st.success(f"{len(rel_ods)} correspondance(s) trouvée(s).")

            except Exception as e:
                st.error(f"Erreur lors de la recherche ODS : {e}")

    # Affichage des résultats mémorisés 
    if st.session_state.ods_relations is None or st.session_state.ods_relations.empty:
        st.info("Aucun résultat ODS à afficher. Utilise le bouton **Lancer la recherche ODS**.")
    else:
        cols_order = [
            "Table_Fact", "ODS_Table",
            "Colonnes_PK_Fact", "Colonnes_Communes", "ODS_Colonnes",
            "Modele"  # ✅ ajout pour affichage du modèle
        ]
        display_df = st.session_state.ods_relations.loc[:, [c for c in cols_order if c in st.session_state.ods_relations.columns]]
        st.dataframe(
            display_df.sort_values(["Table_Fact"], ascending=[True]),
            use_container_width=True
        )
        output_ods = io.BytesIO()
        with pd.ExcelWriter(output_ods, engine='xlsxwriter') as writer:
            display_df.to_excel(writer, index=False, sheet_name='FAIT_ODS')
        output_ods.seek(0)
        st.download_button(
            "Télécharger FAIT↔ODS (Excel)",
            data=output_ods,
            file_name="fact_vers_ods_correspondances.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
