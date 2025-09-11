import oracledb
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
def get_oracle_connection():
    oracledb.init_oracle_client()
    username = os.getenv('ORACLE_USERNAME')
    password = os.getenv('ORACLE_PASSWORD')
    dsn = os.getenv('ORACLE_DSN')
    return oracledb.connect(user=username, password=password, dsn=dsn)

def fetch_oracle_labels(table_names):
    connection = get_oracle_connection()
    cursor = connection.cursor()
    results = {}
    for table in table_names:
        cursor.execute(
            "SELECT t.CODE_OBJT_TABL, d.LIBL_DETL_DESC "
            "FROM MTDO.MTDO_DICT_DESC@PSID11G d "
            "JOIN MTDO.MTDO_DICT_TABL@PSID11G t ON d.IDNT_OBJT_MODL_DETL = t.IDNT_DETL_DESC "
            "WHERE t.CODE_OBJT_TABL = :table_name",
            [table]
        )
        results[table] = [row[1] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    return results

def fetch_mpd_labels():
    connection = get_oracle_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT LIBL_OBJT_MODL FROM MTDO.MTDO_DICT_MODL@PSID11G")
    labels = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    return labels

def fetch_table_structure_by_mpd(mpd_label):
    connection = get_oracle_connection()
    cursor = connection.cursor()
    query = """
    SELECT
        A.TABLE_NAME        AS LIBELLE_DU_SEGMENT,
        A.TABLE_NAME        AS NORME_AFNOR_ET_ADD,
        ' '                 AS DESCRIPTION_DU_SEGMENT,
        ' '                 AS CODE_PROJET,
        A.COLUMN_NAME       AS NOM_EPURE_DE_LA_RUBRIQUE,
        A.COLUMN_NAME       AS NOM_DE_LA_COLONNE_NORME_ADD,
        ' '                 AS DESCRIPTION_RUBRIQUE,
        CASE
            WHEN A.data_type = 'NUMBER' AND A.data_scale > 0
                THEN A.data_type || '(' || A.data_precision || ',' || A.data_scale || ')'
            WHEN A.data_type = 'NUMBER'
                THEN A.data_type || '(' || A.data_precision || ')'
            WHEN A.data_type LIKE 'TIMESTAMP%' THEN A.data_type
            WHEN A.data_type LIKE '%INTERVAL%' THEN A.data_type
            WHEN A.data_type IN ('CLOB','BLOB','DATE','ROWID') THEN A.data_type
            ELSE A.data_type || '(' || A.data_length || ')'
        END AS TYPE_DONNEES_COLONNE,
        CASE WHEN pk_cols.COLUMN_NAME IS NOT NULL THEN 'O' ELSE 'N' END AS PK
    FROM all_tab_columns@psid11g  A
    JOIN all_objects@psid11g  O
      ON O.owner = A.owner AND O.object_name = A.table_name AND O.object_type = 'TABLE'
    LEFT JOIN (
        SELECT cc.OWNER, cc.TABLE_NAME, cc.COLUMN_NAME
        FROM all_cons_columns@psid11g  cc
        JOIN all_constraints@psid11g  c
          ON cc.CONSTRAINT_NAME = c.CONSTRAINT_NAME
         AND cc.OWNER          = c.OWNER
        WHERE c.constraint_type = 'U'
          AND c.owner = 'RODS3'
    ) pk_cols
      ON A.owner = pk_cols.OWNER
     AND A.table_name = pk_cols.TABLE_NAME
     AND A.column_name = pk_cols.COLUMN_NAME
    WHERE A.owner = 'RODS3'
      AND A.table_name IN (
            SELECT UPPER(TRIM(DICO_TABLE.CODE_OBJT_TABL))
            FROM MTDO.MTDO_DICT_TABL@psid11g  DICO_TABLE
            JOIN MTDO.MTDO_DICT_MODL@psid11g  DICO_MPD
              ON DICO_TABLE.IDNT_MODL_TABL = DICO_MPD.IDNT_OBJT_MODL
           WHERE DICO_MPD.LIBL_OBJT_MODL LIKE :mpd_label
      )
    ORDER BY A.table_name, A.COLUMN_ID
    """
    cursor.execute(query, [f"%{mpd_label}%"])
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return pd.DataFrame(rows, columns=columns)