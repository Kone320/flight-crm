#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 07:47:37 2026

@author: konearounaromeo
"""

# debug_eda.py — à lancer dans le terminal : python debug_eda.py
import sqlite3, pickle, traceback, sys
import pandas as pd
import numpy as np

DB_PATH = "data/batch_1.db"

print("="*60)
print("ÉTAPE 1 — Connexion SQLite")
print("="*60)
try:
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"✅ Tables trouvées : {tables['name'].tolist()}")
except Exception as e:
    print(f"❌ ERREUR connexion : {e}")
    traceback.print_exc(); sys.exit(1)

print("\n" + "="*60)
print("ÉTAPE 2 — Lecture aeroports et compagnies")
print("="*60)
try:
    aeroports  = pd.read_sql_query("SELECT * FROM aeroports",  conn)
    compagnies = pd.read_sql_query("SELECT * FROM compagnies", conn)
    print(f"✅ aeroports  : {aeroports.shape}  | colonnes : {aeroports.columns.tolist()}")
    print(f"✅ compagnies : {compagnies.shape} | colonnes : {compagnies.columns.tolist()}")
except Exception as e:
    print(f"❌ ERREUR tables référentielles : {e}")
    traceback.print_exc(); sys.exit(1)

print("\n" + "="*60)
print("ÉTAPE 3 — Lecture vols (échantillon 200 lignes)")
print("="*60)
try:
    vols_sample = pd.read_sql_query(
        "SELECT * FROM vols ORDER BY ROWID LIMIT 200", conn)
    print(f"✅ vols sample : {vols_sample.shape}")
    print(f"   Colonnes : {vols_sample.columns.tolist()}")
    print(f"   Types :\n{vols_sample.dtypes.to_string()}")
    print(f"\n   Valeurs manquantes :\n{vols_sample.isnull().sum()[vols_sample.isnull().sum()>0].to_string()}")
except Exception as e:
    print(f"❌ ERREUR lecture vols : {e}")
    traceback.print_exc(); sys.exit(1)
finally:
    conn.close()

print("\n" + "="*60)
print("ÉTAPE 4 — Test format_heure sur les colonnes temporelles")
print("="*60)
import datetime

def format_heure(valeur):
    if pd.isnull(valeur): return np.nan
    if isinstance(valeur, datetime.time): return valeur
    try: valeur = int(valeur)
    except (ValueError, TypeError): return np.nan
    if valeur == 2400: valeur = 0
    s = f"{valeur:04d}"
    try: return datetime.time(int(s[:2]), int(s[2:]))
    except ValueError: return np.nan

time_cols = ["DEPART PROGRAMME","ARRIVEE PROGRAMMEE","HEURE DE DEPART"]
for col in time_cols:
    if col in vols_sample.columns:
        converted = vols_sample[col].apply(format_heure)
        n_nan = converted.isna().sum()
        print(f"  {col!r:45s} → {n_nan}/{len(converted)} NaN après conversion")
        if n_nan < len(converted):
            print(f"    Exemple valeur brute : {repr(vols_sample[col].dropna().iloc[0])}")
            print(f"    Exemple converti    : {converted.dropna().iloc[0]}")
    else:
        print(f"  ⚠️  Colonne {col!r} ABSENTE du dataset")

print("\n" + "="*60)
print("ÉTAPE 5 — Test RETARD A L'ARRIVEE et variable cible")
print("="*60)
cible_col = "RETARD A L'ARRIVEE"
if cible_col in vols_sample.columns:
    series = pd.to_numeric(vols_sample[cible_col], errors="coerce")
    print(f"  Valeurs non-null : {series.notna().sum()}/{len(series)}")
    print(f"  Min={series.min():.1f}  Max={series.max():.1f}  Médiane={series.median():.1f}")
    retard_bin = (series > 15).astype(int)
    print(f"  RETARD_BIN : {retard_bin.value_counts().to_dict()}")
else:
    print(f"  ❌ Colonne {cible_col!r} ABSENTE")
    print(f"  Colonnes disponibles : {[c for c in vols_sample.columns if 'RETARD' in c.upper()]}")

print("\n" + "="*60)
print("ÉTAPE 6 — Test du feature engineering complet sur 200 lignes")
print("="*60)
try:
    # Reproduire exactement ce que fait data_utils.py
    df = vols_sample.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y", errors="coerce")
    n_date_null = df["DATE"].isna().sum()
    print(f"  DATE parsée : {n_date_null} NaN")
    if n_date_null == len(df):
        # Essayer d'autres formats
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]:
            test = pd.to_datetime(vols_sample["DATE"], format=fmt, errors="coerce")
            if test.notna().sum() > 0:
                print(f"  ⚠️  Format correct pour DATE : {fmt!r}")
                df["DATE"] = test
                break

    df["JOUR_SEM"]   = df["DATE"].dt.weekday
    df["MOIS"]       = df["DATE"].dt.month
    df["ANNEE"]      = df["DATE"].dt.year

    # Heure de départ
    if "HEURE DE DEPART" in df.columns:
        raw = df["HEURE DE DEPART"].astype(str)
        # Cas HH:MM:SS
        if raw.str.contains(":").any():
            df["HEURE_DEP_H"] = pd.to_numeric(
                raw.str.split(":").str[0], errors="coerce")
        else:
            df["HEURE_DEP_H"] = pd.to_numeric(raw, errors="coerce") // 100
        print(f"  HEURE_DEP_H : {df['HEURE_DEP_H'].describe()}")
    else:
        print("  ⚠️  HEURE DE DEPART absente")
        df["HEURE_DEP_H"] = np.nan

    print("  ✅ Feature engineering de base OK")

except Exception as e:
    print(f"  ❌ ERREUR feature engineering : {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("ÉTAPE 7 — Résultat : sauvegarde d'un échantillon propre")
print("="*60)
try:
    df.to_pickle("data/debug_sample_200.pkl")
    print("  ✅ debug_sample_200.pkl sauvegardé")
    print(f"  Shape : {df.shape}")
    print(f"  Colonnes : {df.columns.tolist()}")
except Exception as e:
    print(f"  ❌ ERREUR sauvegarde : {e}")

print("\n✅ Diagnostic terminé.")