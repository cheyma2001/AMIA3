import re
import pandas as pd

# REGEX utilisées pour les features
RE_NUMERIC = re.compile(r'NUMBER\(\d+(,\d+)?\)|INTEGER|DECIMAL|FLOAT|NUMERIC|BIGINT', re.I)
RE_TEXT    = re.compile(r'CHAR\(\d+\)|VARCHAR2\(\d+\)|VARCHAR|TEXT|STRING', re.I)
RE_DATE    = re.compile(r'DATE|TIMESTAMP|TIME', re.I)
RE_KEYS    = re.compile(r'ID|CODE|NUM|KEY|REF|LIBL', re.I)
RE_MEAS    = re.compile(r'MONTANT|AMOUNT|VALEUR|VALUE|PRICE|PRIX|QTE|QTY|COUNT|TOTAL|SUM|MT_|SOLD_|VALR_', re.I)
RE_CODE_PREFIX = re.compile(r'^CODE_|^TYPE_', re.I)
RE_FACT_PREFIX = re.compile(r'FACT|MESURE|FLUX|TRANSACTION|EVNM|MVT_|ECRT_|BALN_', re.I)

def has_code_prefix(table_name):
    return 1 if RE_CODE_PREFIX.search(str(table_name)) else 0

def has_measure_prefix(table_name):
    return 1 if RE_FACT_PREFIX.search(str(table_name)) else 0

def count_key_like_columns(column_names):
    return sum(1 for c in column_names if RE_KEYS.search(str(c)))

def count_measure_like_columns(column_names):
    return sum(1 for c in column_names if RE_MEAS.search(str(c)))

def feature_engineering(df):
    # df doit contenir les colonnes nécessaires
    grouped = df.groupby('LIBELLE_DU_SEGMENT').agg({
        'NOM_EPURE_DE_LA_RUBRIQUE': ['count', lambda x: [str(v).upper() for v in x.fillna('')]],
        'TYPE_DONNEES_COLONNE': lambda x: (
            sum(1 for t in x if RE_NUMERIC.search(str(t))),
            sum(1 for t in x if RE_TEXT.search(str(t))),
            sum(1 for t in x if RE_DATE.search(str(t)))
        ),
        'PK': lambda x: sum(1 for p in x if str(p).strip().upper() == 'O')
    }).reset_index()

    grouped.columns = ['Table_Name', 'Num_Columns', 'Column_Names', 'Type_Counts', 'Num_PKs']

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
    return grouped