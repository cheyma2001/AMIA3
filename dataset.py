import pandas as pd
import re

# Charger le fichier Excel
df = pd.read_excel('srcData/rr.xlsx')

# Regrouper par table
grouped = df.groupby('LIBELLE_DU_SEGMENT').agg({
    'NOM_EPURE_DE_LA_RUBRIQUE': 'count',  # Nombre de colonnes
    'TYPE_DONNEES_COLONNE': lambda x: (
        sum(1 for t in x if re.search(r'NUMBER|INTEGER|DECIMAL|FLOAT', str(t), re.IGNORECASE)),  # Numériques
        sum(1 for t in x if re.search(r'CHAR|VARCHAR|VARCHAR2|TEXT', str(t), re.IGNORECASE)),  # Textuelles
        sum(1 for t in x if re.search(r'DATE|TIMESTAMP', str(t), re.IGNORECASE))  # Dates
    ),
    'PK': lambda x: sum(1 for p in x if p == 'O'),  # Nombre de clés primaires
    'Table_Type': 'first'  # Type de table
}).reset_index()

# Renommer les colonnes
grouped.columns = [
    'Table_Name', 'Num_Columns', 'Type_Counts', 'Num_PKs', 'Table_Type'
]

# Extraire les comptes de types
grouped['Num_Numeric'] = grouped['Type_Counts'].apply(lambda x: x[0])
grouped['Num_Text'] = grouped['Type_Counts'].apply(lambda x: x[1])
grouped['Num_Date'] = grouped['Type_Counts'].apply(lambda x: x[2])

# Calculer les pourcentages
grouped['Pct_Numeric'] = grouped['Num_Numeric'] / grouped['Num_Columns']
grouped['Pct_Text'] = grouped['Num_Text'] / grouped['Num_Columns']
grouped['Pct_Date'] = grouped['Num_Date'] / grouped['Num_Columns']

# Ajouter des caractéristiques basées sur le nom de la table
grouped['Has_Code_Prefix'] = grouped['Table_Name'].apply(
    lambda x: 1 if re.search(r'^CODE_|^TYPE_', str(x), re.IGNORECASE) else 0
)
grouped['Has_Measure_Prefix'] = grouped['Table_Name'].apply(
    lambda x: 1 if re.search(r'FACT|MESURE|FLUX|TRANSACTION|EVNM|MVT_|ECRT_|BALN_', str(x), re.IGNORECASE) else 0
)

# Compter les colonnes avec des motifs spécifiques
def count_key_like_columns(table_name, df):
    cols = df[df['LIBELLE_DU_SEGMENT'] == table_name]['NOM_EPURE_DE_LA_RUBRIQUE']
    return sum(1 for c in cols if re.search(r'ID|CODE|NUM|KEY|REF', str(c), re.IGNORECASE))

def count_measure_like_columns(table_name, df):
    cols = df[df['LIBELLE_DU_SEGMENT'] == table_name]['NOM_EPURE_DE_LA_RUBRIQUE']
    return sum(1 for c in cols if re.search(r'MONTANT|AMOUNT|VALEUR|PRICE|QTE|COUNT|TOTAL|SUM|MT_|SOLD_', str(c), re.IGNORECASE))

grouped['Num_Key_Like'] = grouped['Table_Name'].apply(lambda x: count_key_like_columns(x, df))
grouped['Num_Measure_Like'] = grouped['Table_Name'].apply(lambda x: count_measure_like_columns(x, df))

# Supprimer la colonne temporaire
grouped = grouped.drop(columns=['Type_Counts'])

# Encoder la cible
grouped['Table_Type'] = grouped['Table_Type'].map({'FACT': 1, 'DIMENSION': 0})

# Sauvegarder le dataset
grouped.to_csv('dataset_ml.csv', index=False)
print("Dataset sauvegardé dans dataset_ml.csv")