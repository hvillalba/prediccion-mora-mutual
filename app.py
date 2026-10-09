"""App de demo: score de riesgo de mora >60 días para un préstamo nuevo.
Ejecutar:  streamlit run app.py   (requiere model.pkl en la misma carpeta)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de mora — Mutual MSP y BS", page_icon="📊")
modelo = joblib.load("model.pkl")
enc = modelo.named_steps["pre"].named_transformers_["cat"]
opciones = dict(zip(["tipo_prestamo", "regional", "zona", "oficial"], [list(c) for c in enc.categories_]))

st.title("📊 Riesgo de mora a 12 meses")
st.caption("Score para priorizar cobranza preventiva. No reemplaza la evaluación del oficial de crédito.")

c1, c2 = st.columns(2)
with c1:
    tipo = st.selectbox("Tipo de préstamo", opciones["tipo_prestamo"],
                        index=opciones["tipo_prestamo"].index("EXTRAORDINARIO") if "EXTRAORDINARIO" in opciones["tipo_prestamo"] else 0)
    monto = st.number_input("Monto otorgado (Gs.)", 500_000, 150_000_000, 10_000_000, step=500_000)
    plazo = st.number_input("Plazo (meses)", 1, 72, 24)
    tasa = st.number_input("Tasa mensual (%)", 0.5, 5.0, 2.0, step=0.1)
with c2:
    regional = st.selectbox("Regional", opciones["regional"])
    zona = st.selectbox("Zona", opciones["zona"])
    oficial = st.selectbox("Oficial (código)", opciones["oficial"])
    nro_socio = st.number_input("Nº de socio", 1, 50_000, 15_000)
    previos = st.number_input("Préstamos vigentes del socio", 0, 10, 0)

cuota = monto / plazo + monto * tasa / 100          # cuota con interés flat, como calcula el sistema
st.write(f"Cuota estimada: **Gs. {cuota:,.0f}**".replace(",", "."))

if st.button("Calcular riesgo", type="primary"):
    x = pd.DataFrame([{
        "monto_otorgado": monto, "plazo_meses": plazo, "tasa_mensual": tasa, "cuota": cuota,
        "ratio_cuota_monto": cuota / monto, "nro_socio": nro_socio, "prestamos_vigentes_previos": previos,
        "meses_en_cartera": 12,                      # horizonte fijo de 12 meses
        "tipo_prestamo": tipo, "regional": regional, "zona": zona, "oficial": oficial,
    }])[modelo.columnas_]
    score = modelo.predict_proba(x)[0, 1]
    st.metric("Score de riesgo", f"{score:.2f}", help=f"Umbral de alerta: {modelo.umbral_:.2f}")
    if score >= modelo.umbral_:
        st.error("ALTO RIESGO: derivar a gestión de cobranza preventiva.")
    else:
        st.success("Riesgo normal.")
