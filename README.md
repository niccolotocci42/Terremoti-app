"""
SISTEMA DIDATTICO PER LA LOCALIZZAZIONE DEI TERREMOTI
Versione con intersezione geometrica corretta su sfera
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import requests
from datetime import datetime, timedelta
import numpy as np
import os

# Configurazione pagina
st.set_page_config(
    page_title="Localizzazione Terremoti",
    page_icon="🌍",
    layout="wide"
)

# ============================================================================
# DATABASE STAZIONI PER STATO
# ============================================================================

STATIONS_BY_COUNTRY = {
    "Italia": pd.DataFrame([
        {"code": "R1BA8", "name": "Bagnoli (NA)", "latitude": 40.815, "longitude": 14.168},
        {"code": "R1A01", "name": "Roma", "latitude": 41.903, "longitude": 12.496},
        {"code": "R1A02", "name": "Napoli", "latitude": 40.852, "longitude": 14.268},
        {"code": "R1A03", "name": "Firenze", "latitude": 43.770, "longitude": 11.255},
        {"code": "R1A04", "name": "Bologna", "latitude": 44.494, "longitude": 11.342},
        {"code": "R1A05", "name": "Palermo", "latitude": 38.115, "longitude": 13.361},
        {"code": "R1A06", "name": "Catania", "latitude": 37.502, "longitude": 15.087},
        {"code": "R1A07", "name": "L'Aquila", "latitude": 42.350, "longitude": 13.399},
        {"code": "R1A08", "name": "Pescara", "latitude": 42.464, "longitude": 14.214},
        {"code": "R1A09", "name": "Ancona", "latitude": 43.616, "longitude": 13.518},
        {"code": "R1A10", "name": "Perugia", "latitude": 43.110, "longitude": 12.389},
        {"code": "R1A11", "name": "Cosenza", "latitude": 39.290, "longitude": 16.253},
        {"code": "R1A12", "name": "Bari", "latitude": 41.117, "longitude": 16.871},
        {"code": "R1A13", "name": "Cagliari", "latitude": 39.223, "longitude": 9.121},
        {"code": "R1A14", "name": "Trieste", "latitude": 45.649, "longitude": 13.776},
        {"code": "R1A15", "name": "Genova", "latitude": 44.405, "longitude": 8.946},
        {"code": "R1A16", "name": "Torino", "latitude": 45.070, "longitude": 7.686},
        {"code": "R1A17", "name": "Milano", "latitude": 45.464, "longitude": 9.190},
        {"code": "R1A18", "name": "Venezia", "latitude": 45.440, "longitude": 12.315},
        {"code": "R7906", "name": "Campobasso", "latitude": 41.560, "longitude": 14.660},
    ]),
    
    "Grecia": pd.DataFrame([
        {"code": "R1G01", "name": "Atene", "latitude": 37.983, "longitude": 23.727},
    ]),
    
    "Turchia": pd.DataFrame([
        {"code": "R1T01", "name": "Istanbul", "latitude": 41.008, "longitude": 28.978},
    ]),
    
    "Francia": pd.DataFrame([
        {"code": "R1F01", "name": "Parigi", "latitude": 48.856, "longitude": 2.352},
    ]),
    
    "Germania": pd.DataFrame([
        {"code": "R1D01", "name": "Berlino", "latitude": 52.520, "longitude": 13.405},
    ]),
    
    "Spagna": pd.DataFrame([
        {"code": "R1E01", "name": "Madrid", "latitude": 40.416, "longitude": -3.703},
    ]),
    
    "Svizzera": pd.DataFrame([
        {"code": "R1S01", "name": "Berna", "latitude": 46.948, "longitude": 7.447},
    ]),
    
    "Austria": pd.DataFrame([
        {"code": "R1A01", "name": "Vienna", "latitude": 48.208, "longitude": 16.373},
    ]),
}

COUNTRY_CENTERS = {
    "Italia": (42.0, 12.5),
    "Grecia": (39.0, 22.0),
    "Turchia": (39.0, 35.0),
    "Francia": (46.5, 2.5),
    "Germania": (51.0, 10.0),
    "Spagna": (40.0, -4.0),
    "Svizzera": (46.8, 8.2),
    "Austria": (47.5, 14.5),
}

# ============================================================================
# FUNZIONI DI CALCOLO GEOMETRICO
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcola la distanza in km tra due punti sulla sfera terrestre
    usando la formula dell'haversine.
    """
    R = 6371.0  # Raggio terrestre in km
    
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(min(1.0, a)))
    
    return R * c

def trilateration_geographic(point1, point2, point3, r1, r2, r3):
    """
    Calcola l'intersezione di tre cerchi sulla sfera terrestre.
    Utilizza un metodo iterativo di minimizzazione dell'errore.
    
    point1, point2, point3: tuple (lat, lon) in gradi
    r1, r2, r3: distanze in km
    """
    from scipy.optimize import minimize
    
    # Funzione di errore: somma degli errori quadratici delle distanze
    def error_function(x):
        lat, lon = x
        
        # Limita latitudine a valori validi
        if lat < -90 or lat > 90:
            return 1e10
        
        d1 = haversine_distance(lat, lon, point1[0], point1[1])
        d2 = haversine_distance(lat, lon, point2[0], point2[1])
        d3 = haversine_distance(lat, lon, point3[0], point3[1])
        
        return (d1 - r1)**2 + (d2 - r2)**2 + (d3 - r3)**2
    
    # Punto iniziale: media pesata per distanza inversa
    inv_sum = 1/r1 + 1/r2 + 1/r3
    init_lat = (point1[0]/r1 + point2[0]/r2 + point3[0]/r3) / inv_sum
    init_lon = (point1[1]/r1 + point2[1]/r2 + point3[1]/r3) / inv_sum
    
    # Ottimizzazione
    try:
        result = minimize(error_function, [init_lat, init_lon], method='Nelder-Mead')
        lat_est, lon_est = result.x
        
        # Normalizza longitudine
        lon_est = ((lon_est + 180) % 360) - 180
        
        # Verifica che il punto sia ragionevole
        if lat_est < -90 or lat_est > 90:
            raise ValueError("Latitudine fuori range")
        
        return lat_est, lon_est
    
    except Exception as e:
        # Fallback: media pesata
        lat_est = init_lat
        lon_est = init_lon
        return lat_est, lon_est

def trilateration_correct(circles_data):
    """Calcola l'intersezione geometrica di tre cerchi"""
    if len(circles_data) < 3:
        return None, None
    
    c1 = circles_data[0]
    c2 = circles_data[1]
    c3 = circles_data[2]
    
    return trilateration_geographic(
        (c1['lat'], c1['lon']),
        (c2['lat'], c2['lon']),
        (c3['lat'], c3['lon']),
        c1['distance'],
        c2['distance'],
        c3['distance']
    )

# ============================================================================
# FUNZIONE PER SCARICARE TERREMOTI
# ============================================================================

def get_earthquakes_from_ingv(days=7, min_magnitude=3.0):
    """Scarica terremoti dall'INGV"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://webservices.ingv.it/fdsnws/event/1/query"
    
    params = {
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "format": "text",
        "limit": 500,
        "orderby": "time"
    }
    
    try:
        response = requests.get(url, params=params, timeout=25)
        
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return None, "Nessun dato ricevuto"
        
        earthquakes = []
        for line in lines[1:]:
            parts = line.split('|')
            if len(parts) >= 12:
                try:
                    lon = float(parts[3])
                    lat = float(parts[2])
                    if -15 < lon < 45 and 30 < lat < 70:
                        mag = float(parts[10]) if parts[10] else float(parts[6])
                        earthquakes.append({
                            "time": parts[1].replace('T', ' ')[:19],
                            "latitude": lat,
                            "longitude": lon,
                            "depth": float(parts[4]),
                            "magnitude": mag,
                            "place": parts[11] if len(parts) > 11 else f"Lat: {lat:.2f}, Lon: {lon:.2f}"
                        })
                except:
                    continue
        
        if earthquakes:
            df = pd.DataFrame(earthquakes)
            df = df.sort_values('magnitude', ascending=False)
            return df, None
        else:
            return None, f"Nessun terremoto M>{min_magnitude} in Europa negli ultimi {days} giorni"
            
    except Exception as e:
        return None, str(e)[:100]

# ============================================================================
# FUNZIONI DI CALCOLO
# ============================================================================

def calculate_distance(dt_seconds):
    return dt_seconds * 8

def get_station_color(idx):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#a65628']
    return colors[idx % len(colors)]

def add_custom_station(code, name, lat, lon):
    """Aggiunge una stazione personalizzata"""
    if "Personalizzate" not in STATIONS_BY_COUNTRY:
        STATIONS_BY_COUNTRY["Personalizzate"] = pd.DataFrame()
    
    new_station = pd.DataFrame([{
        "code": code.upper(),
        "name": name if name else code.upper(),
        "latitude": lat,
        "longitude": lon
    }])
    
    STATIONS_BY_COUNTRY["Personalizzate"] = pd.concat([STATIONS_BY_COUNTRY["Personalizzate"], new_station], ignore_index=True)
    
    if "Personalizzate" not in COUNTRY_CENTERS:
        COUNTRY_CENTERS["Personalizzate"] = (lat, lon)
    
    return True

# ============================================================================
# INIZIALIZZAZIONE SESSIONE
# ============================================================================

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'selected_quake' not in st.session_state:
    st.session_state.selected_quake = None
if 'selected_stations' not in st.session_state:
    st.session_state.selected_stations = []
if 'circles_data' not in st.session_state:
    st.session_state.circles_data = []
if 'df_quakes' not in st.session_state:
    st.session_state.df_quakes = None
if 'last_error' not in st.session_state:
    st.session_state.last_error = None
if 'active_country' not in st.session_state:
    st.session_state.active_country = None
if 'current_stations' not in st.session_state:
    st.session_state.current_stations = pd.DataFrame()

def reset_app():
    st.session_state.step = 1
    st.session_state.selected_quake = None
    st.session_state.selected_stations = []
    st.session_state.circles_data = []
    st.session_state.df_quakes = None
    st.session_state.last_error = None
    st.session_state.active_country = None
    st.session_state.current_stations = pd.DataFrame()
    st.rerun()

def set_country(country_name):
    st.session_state.active_country = country_name
    st.session_state.current_stations = STATIONS_BY_COUNTRY.get(country_name, pd.DataFrame())
    st.session_state.selected_stations = []
    st.rerun()

# ============================================================================
# CREAZIONE MAPPA
# ============================================================================

def create_map():
    m = folium.Map(location=[42.0, 12.5], zoom_start=6, 
                   tiles='CartoDB positron', width='100%', height='650px')
    folium.plugins.Fullscreen(position='topright').add_to(m)
    
    # Terremoti
    if st.session_state.df_quakes is not None and not st.session_state.df_quakes.empty:
        for _, row in st.session_state.df_quakes.iterrows():
            mag = row['magnitude']
            if mag < 4.0:
                color = 'green'
            elif mag < 5.0:
                color = 'orange'
            else:
                color = 'red'
            radius = 6 + (mag - 3) * 2
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=f"🎯 {row['time'][:10]}<br>M{row['magnitude']:.1f}<br>{row['place'][:50]}",
                tooltip=f"M{row['magnitude']:.1f} - {row['time'][:10]}",
                color=color, fill=True, fillColor=color, fillOpacity=0.7
            ).add_to(m)
    
    # Epicentro selezionato
    if st.session_state.selected_quake:
        q = st.session_state.selected_quake
        folium.Marker(
            [q['latitude'], q['longitude']],
            popup=f"🌟 EPICENTRO REALE<br>M{q['magnitude']:.1f}<br>{q['time'][:10]}",
            icon=folium.Icon(color='darkgreen', icon='star', prefix='fa')
        ).add_to(m)
    
    # Pallini degli stati
    for country, (lat, lon) in COUNTRY_CENTERS.items():
        is_active = (st.session_state.active_country == country)
        color = 'red' if is_active else 'blue'
        fill_color = 'red' if is_active else 'blue'
        
        n_stations = len(STATIONS_BY_COUNTRY.get(country, pd.DataFrame()))
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=12,
            popup=f"<b>{country}</b><br>{n_stations} stazioni<br>Clicca per caricare",
            tooltip=f"{country} ({n_stations} stazioni)",
            color=color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)
        
        folium.Marker(
            [lat + 1.0, lon],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt; font-weight: bold; background: white; padding: 2px 5px; border-radius: 5px;">{country}</div>'),
            popup=country
        ).add_to(m)
    
    # Stazioni dello stato attivo
    if not st.session_state.current_stations.empty:
        marker_cluster = MarkerCluster(
            name=f"Stazioni {st.session_state.active_country}",
            options={"maxClusterRadius": 40, "disableClusteringAtZoom": 10}
        ).add_to(m)
        
        for _, row in st.session_state.current_stations.iterrows():
            is_selected = row['code'] in st.session_state.selected_stations
            icon_color = 'red' if is_selected else 'blue'
            
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"📡 {row['code']}<br>{row['name']}",
                tooltip=row['code'],
                icon=folium.Icon(color=icon_color, icon='microphone', prefix='fa')
            ).add_to(marker_cluster)
    
    # Cerchi di triangolazione
    if st.session_state.circles_data:
        for c in st.session_state.circles_data:
            folium.Circle(
                radius=c['distance'] * 1000,
                location=[c['lat'], c['lon']],
                color=c['color'],
                fill=False,
                weight=3,
                opacity=0.7,
                popup=f"{c['code']}<br>Δt: {c['dt']:.1f}s<br>Distanza: {c['distance']:.0f}km"
            ).add_to(m)
        
        # Calcola epicentro stimato con trilaterazione geometrica
        if len(st.session_state.circles_data) >= 3:
            est_lat, est_lon = trilateration_correct(st.session_state.circles_data[:3])
            
            if est_lat is not None and est_lon is not None:
                folium.Marker(
                    [est_lat, est_lon],
                    popup=f"⭐ EPICENTRO STIMATO<br>{est_lat:.3f}°N, {est_lon:.3f}°E",
                    icon=folium.Icon(color='red', icon='star', prefix='fa')
                ).add_to(m)
    
    return m

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

col_title, col_reset = st.columns([4, 1])
with col_title:
    st.title("🌍 Caccia al Terremoto")
    st.markdown("*Clicca su un pallino per caricare le stazioni di quel paese*")
with col_reset:
    if st.button("🔄 RESET", type="primary", use_container_width=True):
        reset_app()

col_map, col_controls = st.columns([2.2, 0.8])

# ============================================================================
# COLONNA SINISTRA: MAPPA
# ============================================================================

with col_map:
    st.subheader("🗺️ Mappa Interattiva")
    st.caption("👆 Clicca sui pallini per caricare le stazioni | Clicca sui terremoti per selezionarli")
    
    m = create_map()
    map_data = st_folium(m, width=950, height=650, key="map",
                          returned_objects=["last_object_clicked"])
    
    if map_data and map_data.get('last_object_clicked'):
        clicked = map_data['last_object_clicked']
        
        for country, (lat, lon) in COUNTRY_CENTERS.items():
            if np.sqrt((lat - clicked['lat'])**2 + (lon - clicked['lng'])**2) < 1.5:
                set_country(country)
                break
        
        if st.session_state.df_quakes is not None:
            for _, row in st.session_state.df_quakes.iterrows():
                if np.sqrt((row['latitude'] - clicked['lat'])**2 + 
                           (row['longitude'] - clicked['lng'])**2) < 0.2:
                    st.session_state.selected_quake = row.to_dict()
                    st.session_state.step = 2
                    st.rerun()
        
        if st.session_state.active_country and not st.session_state.current_stations.empty:
            for _, row in st.session_state.current_stations.iterrows():
                if np.sqrt((row['latitude'] - clicked['lat'])**2 + 
                           (row['longitude'] - clicked['lng'])**2) < 0.1:
                    code = row['code']
                    if code in st.session_state.selected_stations:
                        st.session_state.selected_stations.remove(code)
                    else:
                        st.session_state.selected_stations.append(code)
                    st.rerun()

# ============================================================================
# COLONNA DESTRA: CONTROLLI
# ============================================================================

with col_controls:
    
    if st.session_state.last_error:
        st.error(f"❌ {st.session_state.last_error}")
    
    # STEP 1: TERREMOTI
    st.subheader("📊 1. Carica Terremoti")
    st.caption("Magnitudo > 3.0")
    
    col_7gg, col_30gg, col_esempio = st.columns(3)
    with col_7gg:
        if st.button("🔍 7gg", use_container_width=True):
            with st.spinner("Scaricamento..."):
                df, err = get_earthquakes_from_ingv(days=7)
                if df is not None and not df.empty:
                    st.session_state.df_quakes = df
                    st.session_state.last_error = None
                    st.success(f"✅ {len(df)} terremoti")
                else:
                    st.session_state.last_error = err
                st.rerun()
    
    with col_30gg:
        if st.button("🔍 30gg", use_container_width=True):
            with st.spinner("Scaricamento..."):
                df, err = get_earthquakes_from_ingv(days=30)
                if df is not None and not df.empty:
                    st.session_state.df_quakes = df
                    st.session_state.last_error = None
                    st.success(f"✅ {len(df)} terremoti")
                else:
                    st.session_state.last_error = err
                st.rerun()
    
    with col_esempio:
        if st.button("📚 Esempio", use_container_width=True):
            st.session_state.df_quakes = pd.DataFrame([{
                "time": "2026-03-09 23:03:49",
                "latitude": 40.522,
                "longitude": 14.111,
                "magnitude": 5.9,
                "depth": 409.7,
                "place": "Golfo di Napoli e Capri (Napoli)"
            }])
            st.session_state.last_error = None
            st.success("✅ Terremoto esempio (M5.9, 9 marzo 2026)")
            st.rerun()
    
    if st.session_state.df_quakes is not None:
        st.info(f"📊 {len(st.session_state.df_quakes)} terremoti sulla mappa")
    
    # STEP 2: AGGIUNTA STAZIONE
    st.divider()
    st.subheader("➕ Aggiungi stazione")
    
    with st.expander("Aggiungi una stazione", expanded=False):
        col_code, col_lat, col_lon = st.columns(3)
        with col_code:
            custom_code = st.text_input("Codice", placeholder="R1234")
        with col_lat:
            custom_lat = st.number_input("Latitudine", value=41.0, format="%.6f")
        with col_lon:
            custom_lon = st.number_input("Longitudine", value=12.0, format="%.6f")
        
        custom_name = st.text_input("Nome", placeholder="Località")
        
        if st.button("➕ Aggiungi", use_container_width=True):
            if custom_code:
                add_custom_station(custom_code, custom_name, custom_lat, custom_lon)
                st.success(f"✅ Stazione {custom_code.upper()} aggiunta!")
                if not st.session_state.active_country:
                    set_country("Personalizzate")
                st.rerun()
    
    # STEP 3: STATO SELEZIONATO
    st.divider()
    st.subheader("📍 2. Seleziona un paese")
    
    if st.session_state.active_country:
        st.success(f"✅ Paese attivo: **{st.session_state.active_country}**")
        st.caption(f"Stazioni: {len(st.session_state.current_stations)}")
    
    # STEP 4: STAZIONI SELEZIONATE
    if st.session_state.selected_stations:
        st.divider()
        st.subheader("🔘 Stazioni selezionate")
        st.write(f"**{len(st.session_state.selected_stations)} / 3**")
        
        for code in st.session_state.selected_stations:
            station = st.session_state.current_stations[st.session_state.current_stations['code'] == code]
            if not station.empty:
                st.markdown(f"- **{code}** - {station.iloc[0]['name']}")
        
        if len(st.session_state.selected_stations) >= 3:
            if st.button("✅ Triangola", use_container_width=True, type="primary"):
                circles = []
                q = st.session_state.selected_quake
                for idx, code in enumerate(st.session_state.selected_stations[:3]):
                    s = st.session_state.current_stations[st.session_state.current_stations['code'] == code].iloc[0]
                    dist = haversine_distance(s['latitude'], s['longitude'], q['latitude'], q['longitude'])
                    circles.append({
                        'code': code,
                        'name': s['name'],
                        'lat': s['latitude'],
                        'lon': s['longitude'],
                        'dt': dist / 8,
                        'distance': dist,
                        'color': get_station_color(idx)
                    })
                st.session_state.circles_data = circles
                st.session_state.step = 3
                st.rerun()
        else:
            st.warning(f"Ne servono {3 - len(st.session_state.selected_stations)}")
    
    # STEP 5: INSERIMENTO Δt
    if st.session_state.circles_data:
        st.divider()
        st.subheader("⏱️ 3. Inserisci Δt (S-P)")
        
        updated = []
        for i, c in enumerate(st.session_state.circles_data):
            st.markdown(f"**{c['code']}** - {c['name']}")
            dt = st.number_input(
                "Δt (s)",
                value=float(c['dt']),
                step=0.5,
                key=f"dt_{i}"
            )
            dist = dt * 8
            st.caption(f"Distanza: {dist:.0f} km | Teorica: {c['dt']:.1f} s")
            updated.append({**c, 'dt': dt, 'distance': dist})
            st.divider()
        
        st.session_state.circles_data = updated
        
        if len(updated) >= 3:
            est_lat, est_lon = trilateration_correct(updated[:3])
            
            if est_lat is not None and est_lon is not None:
                q = st.session_state.selected_quake
                error = haversine_distance(est_lat, est_lon, q['latitude'], q['longitude'])
                
                st.success(f"⭐ Epicentro stimato: {est_lat:.3f}°N, {est_lon:.3f}°E")
                if error < 50:
                    st.balloons()
                    st.success(f"🎉 Errore: {error:.1f} km - Eccellente!")
                elif error < 150:
                    st.success(f"👍 Errore: {error:.1f} km - Buono!")
                else:
                    st.warning(f"⚠️ Errore: {error:.1f} km - Rivedi i Δt")
            else:
                st.warning("Impossibile calcolare l'intersezione. Prova con altre stazioni.")

# ============================================================================
# RIEPILOGO E GUIDA
# ============================================================================

with st.expander("📋 Riepilogo", expanded=False):
    if st.session_state.selected_quake:
        q = st.session_state.selected_quake
        st.write(f"**Terremoto:** M{q['magnitude']:.1f} - {q['time'][:10]}")
        st.write(f"**Epicentro reale:** {q['latitude']:.3f}°N, {q['longitude']:.3f}°E")
    
    if st.session_state.circles_data:
        df_summary = pd.DataFrame([{
            'Stazione': c['code'],
            'Δt (s)': c['dt'],
            'Distanza (km)': c['distance']
        } for c in st.session_state.circles_data])
        st.dataframe(df_summary, use_container_width=True)

with st.expander("📚 Guida Didattica"):
    st.markdown("**Distanza (km) = Δt (s) × 8**")
    st.markdown("")
    st.markdown("**Perché 8?**")
    st.markdown("Fattore = (Vp × Vs) / (Vp - Vs) = (6 × 3.5) / (6 - 3.5) = 21 / 2.5 = 8.4 ≈ 8")
    st.markdown("")
    st.markdown("**Come procedere:**")
    st.markdown("1. Carica i terremoti (7gg, 30gg o Esempio)")
    st.markdown("2. Clicca su un terremoto sulla mappa")
    st.markdown("3. Clicca su un pallino per caricare le stazioni")
    st.markdown("4. Clicca su 3 stazioni sulla mappa")
    st.markdown("5. Inizia la triangolazione")
    st.markdown("6. Modifica i Δt e osserva l'intersezione dei cerchi")

st.caption("🌍 INGV | Raspberry Shake | Intersezione geometrica dei cerchi (trilaterazione)")
