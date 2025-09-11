import pandas as pd
import re
import os
import glob
import io 

def detect_fact_dimension_tables(excel_file_path):
    """
    Fonction qui lit un fichier Excel contenant des métadonnées de tables
    et identifie automatiquement les tables de fait et de dimension.
    
    Args:
        excel_file_path (str): Chemin vers le fichier Excel
        
    Returns:
        tuple: (dict des résultats, DataFrame avec colonne Table_Type)
    """
    print(f"Lecture du fichier Excel: {excel_file_path}")
    
    if not os.path.exists(excel_file_path):
        print(f"Erreur : Le fichier {excel_file_path} n'existe pas.")
        return None, None
    
    try:
        df = pd.read_excel(excel_file_path)
        print(f"Fichier chargé avec succès. {len(df)} lignes trouvées.")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        return None, None
    
    print(f"Colonnes trouvées dans le fichier : {df.columns.tolist()}")
    
    table_name_column = None
    for col in df.columns:
        if 'NORME' in str(col).upper() or 'TABLE' in str(col).upper() or 'E_D' in str(col).upper():
            table_name_column = col
            break
    
    if not table_name_column:
        print("Erreur : Impossible de trouver la colonne contenant les noms de tables.")
        print("Utilisation de la première colonne par défaut.")
        table_name_column = df.columns[0]
    
    print(f"Utilisation de '{table_name_column}' comme colonne pour les noms de tables.")
    
    column_name_column = None
    format_column = None
    pk_column = None
    
    for col in df.columns:
        if 'NOM' in str(col).upper() and ('RUBRIQUE' in str(col).upper() or 'COLONNE' in str(col).upper()):
            column_name_column = col
        elif 'FORMAT' in str(col).upper() or 'TYPE' in str(col).upper():
            format_column = col
        elif 'PK' in str(col).upper() or 'CLE' in str(col).upper():
            pk_column = col
    
    print(f"Colonne pour les noms de colonnes : {column_name_column}")
    print(f"Colonne pour les formats : {format_column}")
    print(f"Colonne pour les clés primaires : {pk_column}")
    
    if not column_name_column or not format_column:
        print("Attention : Certaines colonnes requises n'ont pas été trouvées.")
        print("Recherche de colonnes alternatives...")
        
        if not column_name_column and len(df.columns) > 3:
            column_name_column = df.columns[3]
            print(f"Utilisation de '{column_name_column}' pour les noms de colonnes.")
        
        if not format_column and len(df.columns) > 5:
            format_column = df.columns[5]
            print(f"Utilisation de '{format_column}' pour les formats.")
        
        if not pk_column and len(df.columns) > 6:
            pk_column = df.columns[6]
            print(f"Utilisation de '{pk_column}' pour les clés primaires.")
    
    tables = {}
    table_types = {}  # Dictionnaire pour stocker le type de chaque table
    
    for index, row in df.iterrows():
        table_name = row[table_name_column] if table_name_column in df.columns else None
        
        if pd.isna(table_name) or not isinstance(table_name, str) or table_name.strip() == "":
            continue
        
        column_name = row[column_name_column] if column_name_column in df.columns else None
        data_type = row[format_column] if format_column in df.columns else None
        
        is_pk = False
        if pk_column in df.columns:
            pk_value = row[pk_column]
            is_pk = (isinstance(pk_value, str) and pk_value.upper() == 'O') or pk_value == 1
        
        if pd.isna(column_name) or not isinstance(column_name, str) or column_name.strip() == "":
            continue
        
        if table_name not in tables:
            tables[table_name] = {
                'columns': [], 
                'pk_count': 0, 
                'numeric_count': 0, 
                'date_count': 0, 
                'text_count': 0,
                'key_like_columns': 0,
                'measure_like_columns': 0,
                'excluded_columns': 0,
                'excluded_column_names': []
            }
        
        tables[table_name]['columns'].append({
            'name': column_name,
            'type': data_type,
            'is_pk': is_pk
        })
        
        if is_pk:
            tables[table_name]['pk_count'] += 1
        
        if bool(re.search(r'ID|CODE|NUM|KEY|REF', str(column_name), re.IGNORECASE)) and not is_pk:
            tables[table_name]['key_like_columns'] += 1
        
        if bool(re.search(r'MONTANT|AMOUNT|VALEUR|VALUE|PRIX|PRICE|QTE|QTY|NOMBRE|COUNT|TOTAL|SUM|MT_|SOLD_|ENCR', 
                         str(column_name), re.IGNORECASE)):
            if data_type and re.search(r'NUMBER.*\([1-9][0-9]', str(data_type), re.IGNORECASE):
                tables[table_name]['measure_like_columns'] += 1
        
        if bool(re.search(r'NUMR_PERS|NUM_PERSONNE|ID_PERSONNE|NUMR_ENTT_TITL|NUMR_CART|IDNT_ADRS', str(column_name), re.IGNORECASE)):
            tables[table_name]['excluded_columns'] += 1
            tables[table_name]['excluded_column_names'].append(column_name)
            print(f"    Colonne interdite détectée : {column_name} dans {table_name}")
        
        if data_type:
            if re.search(r'NUMBER|INTEGER|INT|DECIMAL|FLOAT|NUMERIC', str(data_type), re.IGNORECASE):
                tables[table_name]['numeric_count'] += 1
            elif re.search(r'DATE|TIME|TIMESTAMP', str(data_type), re.IGNORECASE):
                tables[table_name]['date_count'] += 1
            elif re.search(r'CHAR|VARCHAR|VARCHAR2|TEXT|STRING', str(data_type), re.IGNORECASE):
                tables[table_name]['text_count'] += 1
    
    print(f"Nombre de tables détectées : {len(tables)}")
    
    fact_tables = []
    dimension_tables = []

    for table_name, table_info in tables.items():
        total_columns = len(table_info['columns'])
        if total_columns == 0:
            continue

        # Règle : si la table commence par CODE_ ou TYPE_, c'est une dimension
        if table_name.upper().startswith('CODE_') or table_name.upper().startswith('TYPE_'):
            dimension_tables.append({
                'name': table_name,
                'fact_score': 0,
                'dim_score': 10,
                'metrics': {
                    'columns': total_columns,
                    'pk_count': table_info['pk_count'],
                    'numeric_count': table_info['numeric_count'],
                    'text_count': table_info['text_count'],
                    'date_count': table_info['date_count'],
                    'key_like_columns': table_info['key_like_columns'],
                    'measure_like_columns': table_info['measure_like_columns']
                }
            })
            table_types[table_name] = 'DIMENSION'
            print(f"  → {table_name} classée comme TABLE DE DIMENSION (règle : commence par CODE_ ou TYPE_)")
            continue

        if table_name.upper() == 'PERS':
            print(f"Colonnes de la table PERS : {[col['name'] for col in table_info['columns']]}")
        
        numeric_ratio = table_info['numeric_count'] / total_columns
        text_ratio = table_info['text_count'] / total_columns
        date_ratio = table_info['date_count'] / total_columns
        fk_ratio = table_info['key_like_columns'] / total_columns
        
        fact_score = 0
        dim_score = 0
        
        if bool(re.search(r'FACT|FAIT|MESURE|FLUX|TRANSACTION|ANLS|EVNM|VENTE|ACHAT|MVT_|ECRT_|BALN_|JUST_|RAPP_|SOLD_', table_name, re.IGNORECASE)):
            fact_score += 4
            print(f"  {table_name}: +4 au score de fait (nom suggérant une table de fait)")
        
        if bool(re.search(r'DIM|REF|LOOKUP|DRIS|PERS|ORGN|CLIENT|PRODUIT|ARTICLE|TEMPS|PARM_', table_name, re.IGNORECASE)):
            dim_score += 5
            print(f"  {table_name}: +5 au score de dimension (nom suggérant une table de dimension)")
        
        if numeric_ratio > 0.3:
            fact_score += 2
            print(f"  {table_name}: +2 au score de fait (nombreuses colonnes numériques : {numeric_ratio:.1%})")
        
        if text_ratio > 0.4:
            dim_score += 2.5
            print(f"  {table_name}: +2.5 au score de dimension (nombreuses colonnes textuelles : {text_ratio:.1%})")
        
        if date_ratio > 0.15:
            fact_score += 1.5
            print(f"  {table_name}: +1.5 au score de fait (présence de dates : {date_ratio:.1%})")
        
        if fk_ratio > 0.25:
            fact_score += 1.5
            print(f"  {table_name}: +1.5 au score de fait (nombreuses clés étrangères : {fk_ratio:.1%})")
            if text_ratio > 0.4:
                dim_score += 1
                print(f"  {table_name}: +1 au score de dimension (clés étrangères avec texte élevé)")
        
        if table_info['pk_count'] == 1:
            dim_score += 2
            print(f"  {table_name}: +2 au score de dimension (unique clé primaire)")
        
        if table_info['pk_count'] > 1:
            fact_score += 0.5
            print(f"  {table_name}: +0.5 au score de fait (clé primaire composite)")
        
        if table_info['measure_like_columns'] > 0:
            fact_score += table_info['measure_like_columns'] * 1.5
            print(f"  {table_name}: +{table_info['measure_like_columns'] * 1.5} au score de fait (colonnes de mesure)")
        
        if total_columns > 20:
            fact_score += 2
            print(f"  {table_name}: +2 au score de fait (nombre élevé de colonnes : {total_columns})")
        elif total_columns < 10:
            dim_score += 2
            print(f"  {table_name}: +2 au score de dimension (peu de colonnes : {total_columns})")
        
        code_prefix_count = sum(
            1 for col in table_info['columns']
            if str(col['name']).upper().startswith('CODE')
        )

        if code_prefix_count >= 1:
            fact_score += 0.5
            print(f"  {table_name}: +0.5 au score de fait (au moins une colonne commence par 'CODE')")

        if code_prefix_count >= 2:
            dim_score += 1.5
            print(f"  {table_name}: +1.5 au score de dimension (plusieurs colonnes commencent par 'CODE')")
       
        # Règle supplémentaire : présence d'un couple code/libellé
        code_col = None
        libelle_col = None
        # On cherche CODE_<nom_table> et LIBL_<nom_table> 
        nom_table_sans_espace = re.sub(r'\W', '', table_name).upper()
        for col in table_info['columns']:
            colname = col['name'].replace(' ', '').upper()
            if colname == f"CODE_{nom_table_sans_espace}":
                code_col = col['name']
            if colname == f"LIBL_{nom_table_sans_espace}":
                libelle_col = col['name']
        if code_col and libelle_col:
            dim_score += 3  
            print(f"  {table_name}: +3 au score de dimension (présence d'un couple code/libellé : {code_col}, {libelle_col})")
        
        if table_info['excluded_columns'] > 0:
            print(f"  {table_name}: Forcée comme TABLE DE FAIT (contient {table_info['excluded_columns']} colonnes interdites : {', '.join(table_info['excluded_column_names'])})")
            fact_tables.append({
                'name': table_name,
                'fact_score': fact_score + 5,
                'dim_score': 0,
                'metrics': {
                    'columns': total_columns,
                    'pk_count': table_info['pk_count'],
                    'numeric_count': table_info['numeric_count'],
                    'text_count': table_info['text_count'],
                    'date_count': table_info['date_count'],
                    'key_like_columns': table_info['key_like_columns'],
                    'measure_like_columns': table_info['measure_like_columns']
                }
            })
            table_types[table_name] = 'FACT'
            continue
        
        table_result = {
            'name': table_name,
            'fact_score': fact_score,
            'dim_score': dim_score,
            'metrics': {
                'columns': total_columns,
                'pk_count': table_info['pk_count'],
                'numeric_count': table_info['numeric_count'],
                'text_count': table_info['text_count'],
                'date_count': table_info['date_count'],
                'key_like_columns': table_info['key_like_columns'],
                'measure_like_columns': table_info['measure_like_columns']
            }
        }
        
        if fact_score > dim_score:
            print(f"  → {table_name} classée comme TABLE DE FAIT (score fait : {fact_score:.1f}, score dim : {dim_score:.1f})")
            fact_tables.append(table_result)
            table_types[table_name] = 'FACT'
        else:
            print(f"  → {table_name} classée comme TABLE DE DIMENSION (score fait : {fact_score:.1f}, score dim : {dim_score:.1f})")
            dimension_tables.append(table_result)
            table_types[table_name] = 'DIMENSION'
    
    fact_tables.sort(key=lambda x: x['fact_score'], reverse=True)
    dimension_tables.sort(key=lambda x: x['dim_score'], reverse=True)
    
    # Ajouter la colonne Table_Type au DataFrame
    df['Table_Type'] = df[table_name_column].map(table_types).fillna('UNKNOWN')
    
    return {
        'fact_tables': fact_tables, 
        'dimension_tables': dimension_tables,
        'total_tables': len(tables)
    }, df

def print_results(results, file_name):
    """
    Affiche les résultats de la détection de façon formatée pour un fichier spécifique
    
    Args:
        results (dict): Résultats de la détection
        file_name (str): Nom du fichier traité
    """
    if not results:
        print(f"Aucun résultat à afficher pour {file_name}.")
        return
    
    fact_tables = results['fact_tables']
    dimension_tables = results['dimension_tables']
    
    seen_dim_tables = set()
    unique_dimension_tables = []
    for table in dimension_tables:
        table_name = table['name']
        if table_name not in seen_dim_tables:
            seen_dim_tables.add(table_name)
            unique_dimension_tables.append(table)
        else:
            print(f"Attention : Table de dimension en doublon détectée : {table_name}")
    
    print("\n" + "="*80)
    print(f"RÉSULTATS DE LA DÉTECTION POUR {file_name}")
    print("="*80)
    
    print(f"\nNombre total de tables analysées : {results['total_tables']}")
    print(f"Tables de fait identifiées : {len(fact_tables)}")
    print(f"Tables de dimension identifiées : {len(unique_dimension_tables)}")
    
    print("\n" + "-"*40)
    print("TABLES DE FAIT")
    print("-"*40)
    
    if not fact_tables:
        print("Aucune table de fait détectée.")
    else:
        for i, table in enumerate(fact_tables, 1):
            print(f"{i}. {table['name']} (Score : {table['fact_score']:.1f})")
            metrics = table['metrics']
            print(f"   - Colonnes : {metrics['columns']} (PK : {metrics['pk_count']}, Numériques : {metrics['numeric_count']}, Texte : {metrics['text_count']}, Date : {metrics['date_count']})")
            print(f"   - Clés étrangères potentielles : {metrics['key_like_columns']}")
            print(f"   - Mesures potentielles : {metrics['measure_like_columns']}")
            print()
    
    print("\n" + "-"*40)
    print("TABLES DE DIMENSION")
    print("-"*40)
    
    if not unique_dimension_tables:
        print("Aucune table de dimension détectée.")
    else:
        for i, table in enumerate(unique_dimension_tables, 1):
            print(f"{i}. {table['name']} (Score : {table['dim_score']:.1f})")
            metrics = table['metrics']
            print(f"   - Colonnes : {metrics['columns']} (PK : {metrics['pk_count']}, Numériques : {metrics['numeric_count']}, Texte : {metrics['text_count']}, Date : {metrics['date_count']})")
            print(f"   - Clés étrangères potentielles : {metrics['key_like_columns']}")
            print()
    
    print("\n" + "="*80)
    print(f"CONCLUSION POUR {file_name}")
    print("="*80)
    
    if fact_tables:
        main_fact = fact_tables[0]['name']
        print(f"Table de fait principale : {main_fact}")
        print("Cette table contient probablement les mesures et les métriques de votre modèle.")
    
    if unique_dimension_tables:
        print("\nDimensions principales :")
        for i, table in enumerate(unique_dimension_tables[:3], 1):
            print(f"{i}. {table['name']}")

def find_excel_files(directory="src1"):
    """
    Trouve tous les fichiers Excel dans le répertoire spécifié
    
    Args:
        directory (str): Chemin du répertoire à scanner (par défaut : 'src')
    
    Returns:
        list: Liste des chemins des fichiers Excel trouvés
    """
    excel_extensions = ['*.xlsx', '*.xls', '*.xlsm', '*.xlsb']
    excel_files = []
    
    for extension in excel_extensions:
        files = glob.glob(os.path.join(directory, extension))
        excel_files.extend(files)
    
    return excel_files

def main():
    """
    Fonction principale
    """
    source_directory = "src1"
    
    # Vérifier si le répertoire src existe
    if not os.path.exists(source_directory):
        print(f"Erreur : Le répertoire '{source_directory}' n'existe pas.")
        return
    
    # Trouver tous les fichiers Excel dans src
    excel_files = find_excel_files(source_directory)
    
    if not excel_files:
        print(f"Aucun fichier Excel trouvé dans le répertoire '{source_directory}'.")
        return
    
    print(f"Fichiers Excel trouvés dans '{source_directory}' : {len(excel_files)}")
    all_dfs = []
    
    for excel_file in excel_files:
        print(f"\nTraitement du fichier : {excel_file}")
        
        results, df_output = detect_fact_dimension_tables(excel_file)
        
        if results:
            print_results(results, os.path.basename(excel_file))
            
            # Ajouter une colonne Source_File
            df_output['Source_File'] = os.path.basename(excel_file)
            all_dfs.append(df_output)
    
    if all_dfs:
        # Combiner tous les DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        save_option = input("\nVoulez-vous enregistrer les résultats combinés dans un fichier ? (o/n) : ").strip().lower()
        if save_option == 'o' or save_option == 'oui':
            output_file = input("Nom du fichier de sortie (défaut : resultats_detection.xlsx) : ").strip()
            if not output_file:
                output_file = "resultats_detection.xlsx"
            elif not output_file.endswith('.xlsx'):
                output_file += '.xlsx'
            
            # Enregistrer le DataFrame combiné dans src
            os.makedirs("srcData", exist_ok=True)
            output_path = os.path.join("srcData/", output_file)
            combined_df.to_excel(output_path, index=False)
            print(f"Résultats combinés enregistrés dans {output_path}")
    else:
        print("\nAucun résultat valide à combiner.")

if __name__ == "__main__":
    main()