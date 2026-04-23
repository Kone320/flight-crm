#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 07:51:50 2026

@author: konearounaromeo
"""

# preprocess_eda_sample.py — python preprocess_eda_sample.py
import sqlite3, datetime, os
import pandas as pd
import numpy as np

DB_PATH      = "data/batch_1.db"
OUTPUT_PATH  = "data/eda_sample.csv"
SAMPLE_SIZE  = 5_000
RANDOM_SEED  = 42

COUNTRY_MAPPING = {
    'GB':'Royaume-Uni','FR':'France','DE':'Allemagne','ES':'Espagne','IT':'Italie',
    'US':'États-Unis','JP':'Japon','CN':'Chine','IN':'Inde','BR':'Brésil',
    'AE':'Émirats arabes unis','QA':'Qatar','TR':'Turquie','RU':'Russie',
    'ZA':'Afrique du Sud','NG':'Nigeria','KE':'Kenya','ET':'Éthiopie','SN':'Sénégal',
    'MA':'Maroc','DZ':'Algérie','TN':'Tunisie','EG':'Égypte',
    'SA':'Arabie saoudite','IQ':'Irak','IR':'Iran','PK':'Pakistan',
    'BD':'Bangladesh','TH':'Thaïlande','VN':'Vietnam','ID':'Indonésie',
    'MY':'Malaisie','SG':'Singapour','PH':'Philippines','KR':'Corée du Sud',
    'AU':'Australie','NZ':'Nouvelle-Zélande','CA':'Canada','MX':'Mexique',
    'AR':'Argentine','CL':'Chili','CO':'Colombie','PE':'Pérou',
    'NL':'Pays-Bas','BE':'Belgique','CH':'Suisse','AT':'Autriche',
    'SE':'Suède','NO':'Norvège','DK':'Danemark','FI':'Finlande',
    'PL':'Pologne','CZ':'République tchèque','HU':'Hongrie','RO':'Roumanie',
    'PT':'Portugal','GR':'Grèce','BG':'Bulgarie','HR':'Croatie',
    'SK':'Slovaquie','SI':'Slovénie','LT':'Lituanie','LV':'Lettonie',
    'EE':'Estonie','BY':'Biélorussie','UA':'Ukraine','GE':'Géorgie',
    'AZ':'Azerbaïdjan','AM':'Arménie','KZ':'Kazakhstan','UZ':'Ouzbékistan',
    'MN':'Mongolie','HK':'Hong Kong','TW':'Taïwan','MO':'Macao',
    'LK':'Sri Lanka','NP':'Népal','MM':'Myanmar','KH':'Cambodge',
    'OM':'Oman','BH':'Bahreïn','KW':'Koweït','JO':'Jordanie',
    'LB':'Liban','IL':'Israël','SY':'Syrie',
    'TZ':'Tanzanie','UG':'Ouganda','ZM':'Zambie','ZW':'Zimbabwe',
    'AO':'Angola','MZ':'Mozambique','MG':'Madagascar','MU':'Maurice',
    'SC':'Seychelles','CV':'Cap-Vert','SL':'Sierra Leone','LR':'Libéria',
    'GH':'Ghana','BF':'Burkina Faso','NE':'Niger','MR':'Mauritanie',
    'SD':'Soudan','SS':'Soudan du Sud','CD':'République démocratique du Congo',
    'LU':'Luxembourg','IE':'Irlande','MT':'Malte','CY':'Chypre',
    'IS':'Islande','AL':'Albanie','RS':'Serbie','BA':'Bosnie-Herzégovine',
    'ME':'Monténégro','MK':'Macédoine du Nord','XK':'Kosovo',
    'LY':'Libye','BW':'Botswana','MV':'Maldives','BN':'Brunei',
    'KG':'Kirghizistan','TM':'Turkménistan','LU':'Luxembourg',
}


def format_heure(valeur):
    if pd.isnull(valeur):
        return None
    if isinstance(valeur, datetime.time):
        return valeur.hour
    try:
        valeur = str(valeur).strip()
        # Format HH:MM:SS ou HH:MM
        if ":" in valeur:
            return int(valeur.split(":")[0])
        # Format HHMM entier
        v = int(float(valeur))
        if v == 2400: v = 0
        return int(f"{v:04d}"[:2])
    except (ValueError, TypeError):
        return None


def main():
    print(f"📂 Connexion à {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    # ── 1. Charger les référentiels ───────────────────────────
    print("  Chargement aeroports & compagnies...")
    aeroports  = pd.read_sql_query("SELECT * FROM aeroports", conn)
    compagnies = pd.read_sql_query("SELECT * FROM compagnies", conn)

    aeroports["PAYS"] = aeroports["PAYS"].map(COUNTRY_MAPPING).fillna(aeroports["PAYS"])
    aeroports = aeroports.drop_duplicates(subset="CODE IATA", keep="first")

    # ── 2. Compter les vols totaux ────────────────────────────
    total = pd.read_sql_query("SELECT COUNT(*) as n FROM vols", conn).iloc[0,0]
    print(f"  Total vols dans la base : {total:,}")

    # ── 3. Échantillonnage stratifié par année ────────────────
    # On récupère d'abord les années disponibles
    years_df = pd.read_sql_query(
        "SELECT DISTINCT strftime('%Y', DATE) as yr FROM vols WHERE DATE IS NOT NULL",
        conn
    )
    years = sorted(years_df["yr"].dropna().tolist())

    if len(years) == 0:
        # Fallback si la date n'est pas parseable par SQLite
        print("  ⚠️  Dates non parsées par SQLite, lecture directe...")
        vols_raw = pd.read_sql_query(
            f"SELECT * FROM vols ORDER BY ROWID LIMIT {SAMPLE_SIZE * 3}", conn)
    else:
        print(f"  Années disponibles : {years}")
        per_year = max(SAMPLE_SIZE // len(years), 100)
        frames = []
        for yr in years:
            q = f"""SELECT * FROM vols
                    WHERE DATE LIKE '{yr}%'
                    ORDER BY RANDOM()
                    LIMIT {per_year}"""
            frames.append(pd.read_sql_query(q, conn))
        vols_raw = pd.concat(frames, ignore_index=True)
        print(f"  Lignes brutes récupérées : {len(vols_raw):,}")

    conn.close()

    # ── 4. Nettoyage colonnes inutiles ────────────────────────
    var_to_drop = [
        "HEURE D'ARRIVEE","ATTERRISSAGE",
        "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
        "TEMPS PASSE","TEMPS DE VOL","DETOURNEMENT","ANNULATION",
        "RETARD SYSTEM","RETARD SECURITE","RETARD COMPAGNIE",
        "RETARD AVION","RETARD METEO","RAISON D'ANNULATION",
        "NIVEAU DE SECURITE","DECOLLAGE","VOL","CODE AVION","IDENTIFIANT",
    ]
    vols_raw.drop(columns=var_to_drop, errors="ignore", inplace=True)

    # ── 5. Fusion compagnies ──────────────────────────────────
    df = vols_raw.merge(compagnies, left_on="COMPAGNIE AERIENNE",
                        right_on="CODE", how="left")
    df.rename(columns={"COMPAGNIE": "NOM_COMPAGNIE"}, inplace=True)
    df.drop(columns=["CODE"], errors="ignore", inplace=True)

    # ── 6. Fusion aéroports départ ────────────────────────────
    ap_dep = aeroports.rename(columns={
        "CODE IATA":"AEROPORT DEPART","NOM":"NOM_AP_DEP",
        "LIEU":"VILLE_DEP","PAYS":"PAYS_DEP",
        "LONGITUDE":"LON_DEP","LATITUDE":"LAT_DEP","HAUTEUR":"HAUTEUR_DEP",
        "PRIX RETARD PREMIERE 10 MINUTES":"COUT_10_DEP",
        "PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES":"COUT_SUP_DEP",
    })
    df = df.merge(ap_dep, on="AEROPORT DEPART", how="left")

    # ── 7. Fusion aéroports arrivée ───────────────────────────
    ap_arr = aeroports.rename(columns={
        "CODE IATA":"AEROPORT ARRIVEE","NOM":"NOM_AP_ARR",
        "LIEU":"VILLE_ARR","PAYS":"PAYS_ARR",
        "LONGITUDE":"LON_ARR","LATITUDE":"LAT_ARR","HAUTEUR":"HAUTEUR_ARR",
        "PRIX RETARD PREMIERE 10 MINUTES":"COUT_10_ARR",
        "PRIS RETARD POUR CHAQUE MINUTE APRES 10 MINUTES":"COUT_SUP_ARR",
    })
    df = df.merge(ap_arr, on="AEROPORT ARRIVEE", how="left")

    # ── 8. Renommage ──────────────────────────────────────────
    df.rename(columns={
        "RETARD A L'ARRIVEE"               : "RETARD_ARRIVEE",
        "RETART DE DEPART"                 : "RETARD_DEPART",
        "TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE": "TAXI_TIME",
        "TEMPS PROGRAMME"                  : "TEMPS_VOL",
        "DEPART PROGRAMME"                 : "DEPART_PROG",
        "HEURE DE DEPART"                  : "HEURE_DEP",
        "ARRIVEE PROGRAMMEE"               : "ARRIVEE_PROG",
        "COMPAGNIE AERIENNE"               : "COMPAGNIE",
        "AEROPORT DEPART"                  : "AP_DEP",
        "AEROPORT ARRIVEE"                 : "AP_ARR",
        "DISTANCE"                         : "DISTANCE",
    }, inplace=True)

    # ── 9. Typage numérique ───────────────────────────────────
    for c in ["RETARD_ARRIVEE","RETARD_DEPART","TAXI_TIME","TEMPS_VOL",
              "DISTANCE","COUT_10_DEP","COUT_SUP_DEP","COUT_10_ARR","COUT_SUP_ARR",
              "LON_DEP","LAT_DEP","HAUTEUR_DEP","LON_ARR","LAT_ARR","HAUTEUR_ARR"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 10. Variables temporelles ─────────────────────────────
    # Détection automatique du format de date
    sample_date = df["DATE"].dropna().iloc[0] if "DATE" in df.columns else ""
    date_formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]
    parsed_date  = None
    for fmt in date_formats:
        try:
            parsed_date = pd.to_datetime(df["DATE"], format=fmt, errors="coerce")
            if parsed_date.notna().mean() > 0.5:
                print(f"  Format date détecté : {fmt!r}")
                break
        except Exception:
            continue
    df["DATE"]      = parsed_date if parsed_date is not None else pd.NaT
    df["ANNEE"]     = df["DATE"].dt.year
    df["MOIS"]      = df["DATE"].dt.month
    df["JOUR_SEM"]  = df["DATE"].dt.dayofweek

    # Heure de départ → entier 0-23
    if "HEURE_DEP" in df.columns:
        raw = df["HEURE_DEP"].astype(str)
        df["HEURE_DEP_H"] = raw.apply(format_heure).astype("Int64")
    else:
        df["HEURE_DEP_H"] = pd.NA

    # ── 11. Variables dérivées ────────────────────────────────
    df["RETARD_BIN"]    = (pd.to_numeric(df["RETARD_ARRIVEE"], errors="coerce") > 15).astype("Int8")
    df["RETARD_SEVERE"] = (pd.to_numeric(df["RETARD_ARRIVEE"], errors="coerce") > 60).astype("Int8")
    df["COUT_TOTAL"]    = (
        df.get("COUT_10_DEP", pd.Series(0, index=df.index)).fillna(0)
      + df.get("COUT_SUP_DEP", pd.Series(0, index=df.index)).fillna(0)
      + df.get("COUT_10_ARR",  pd.Series(0, index=df.index)).fillna(0)
      + df.get("COUT_SUP_ARR", pd.Series(0, index=df.index)).fillna(0)
    )

    JOURS  = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
    MOIS_L = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

    df["NOM_MOIS"] = df["MOIS"].apply(
        lambda x: MOIS_L[int(x)-1] if pd.notna(x) and 1<=int(x)<=12 else "?")
    df["NOM_JOUR"] = df["JOUR_SEM"].apply(
        lambda x: JOURS[int(x)] if pd.notna(x) and 0<=int(x)<=6 else "?")

    df["CRENEAU"] = pd.cut(
        pd.to_numeric(df["HEURE_DEP_H"], errors="coerce"),
        bins=[-1,5,8,11,14,17,20,24],
        labels=["Nuit (0-5h)","Tôt matin (6-8h)","Matin (9-11h)",
                "Midi (12-14h)","Après-midi (15-17h)","Soirée (18-20h)","Nuit tardive (21-24h)"],
    )
    df["SEGMENT_DIST"] = pd.cut(
        pd.to_numeric(df["DISTANCE"], errors="coerce"),
        bins=[0,500,1500,3000,np.inf],
        labels=["Court (<500km)","Moyen (500-1500km)",
                "Long (1500-3000km)","Ultra-long (>3000km)"],
    )

    # ── 12. Nettoyage final & échantillon ─────────────────────
    df = df.dropna(subset=["RETARD_BIN"])
    df = df.reset_index(drop=True)

    # Sous-échantillon final stratifié par RETARD_BIN
    if len(df) > SAMPLE_SIZE:
        df = (df.groupby("RETARD_BIN", group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), int(SAMPLE_SIZE * len(g) / len(df))),
                    random_state=RANDOM_SEED))
                .reset_index(drop=True))

    # ── 13. Export CSV ────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n✅ Fichier sauvegardé : {OUTPUT_PATH}")
    print(f"   Shape finale : {df.shape}")
    print(f"   Taux retard  : {df['RETARD_BIN'].mean()*100:.1f}%")
    print(f"   Colonnes     : {df.columns.tolist()}")
    print(f"\n   Distribution RETARD_BIN :")
    print(f"   {df['RETARD_BIN'].value_counts().to_dict()}")
    print(f"\n   NaN par colonne clé :")
    key_cols = ["RETARD_BIN","ANNEE","MOIS","JOUR_SEM","HEURE_DEP_H",
                "CRENEAU","SEGMENT_DIST","NOM_COMPAGNIE","PAYS_DEP","LAT_DEP","LON_DEP"]
    for c in key_cols:
        if c in df.columns:
            print(f"   {c:<20} : {df[c].isna().sum()} NaN")

if __name__ == "__main__":
    main()