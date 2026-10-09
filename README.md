# Predicción de mora en préstamos — Mutual Nacional de Funcionarios del MSP y BS

**Autor:** Héctor Villalba · Proyecto Final M7 · Diplomado en Machine Learning y Deep Learning (UNA)

## Problema
La mutual (~20.000 socios, 17 sucursales) otorga alrededor de 1.000 préstamos por mes. Una parte llega a más de 60 días de atraso, lo que obliga a constituir previsiones y genera pérdidas. Hoy la cobranza es reactiva: se actúa cuando el atraso ya existe.

## Objetivo
Asignar a cada préstamo un **score de riesgo de llegar a más de 60 días de atraso** para que el área de cobranzas priorice la gestión preventiva sobre el 20% de préstamos de mayor riesgo.

## Datos
- **Fuentes:** reportes del sistema de gestión de la mutual con corte al 30/09/2026: planilla de previsiones (una fila por préstamo vigente) cruzada por cédula con el detalle de atraso por regional.
- **Población:** 7.755 préstamos otorgados entre abril 2025 y marzo 2026, todavía dentro de su plazo, de socios activos.
- **Target:** `mora_60` = 1 si el préstamo tiene más de 60 días vencidos al corte (5,9% de positivos).
- **Variables (conocidas al otorgar):** tipo de préstamo, monto, plazo, tasa mensual, cuota, relación cuota/monto, nº de socio (proxy de antigüedad), préstamos vigentes previos del socio, regional, zona, oficial de crédito y meses en cartera (control del tiempo de exposición; se fija en 12 para predecir).
- **Privacidad:** se eliminaron cédula, nombre y número de préstamo; los oficiales se reemplazaron por códigos. `datos_muestra.csv` tiene 100 filas (20 morosos + 80 al día, enriquecida en positivos para que el notebook corra con la muestra).

### Decisiones para evitar sesgos y fuga de datos
- Se excluyeron saldos, intereses cobrados, fecha de último pago, categoría de riesgo y saldo de aportes: todos se conocen *después* del otorgamiento.
- Los préstamos viejos que siguen vigentes tienen más de 90% de mora (los buenos pagadores ya terminaron de pagar: **sesgo de supervivencia**), por eso se acotó la ventana de otorgamiento.
- **Fuga detectada y corregida:** el reporte por regional solo lista socios activos. Los socios ausentes (excluidos, cobro judicial) tenían 31,6% de mora; "regional desconocida" se volvió la variable más importante e inflaba el ROC-AUC de ~0,79 a ~0,84. Se excluyeron esos casos.

## Modelo
`Pipeline` de scikit-learn con `ColumnTransformer` (imputación + escalado en numéricas, One-Hot con agrupación de categorías poco frecuentes en categóricas). Se compararon con validación cruzada estratificada de 5 folds:

| Modelo | ROC-AUC (CV) | PR-AUC (CV) |
|---|---|---|
| Regresión logística (baseline) | 0,790 | 0,258 |
| **Random Forest (elegido)** | **0,785** | **0,262** |
| Gradient Boosting | 0,770 | 0,225 |

Métrica principal: **PR-AUC**, porque la clase positiva es minoritaria y el objetivo es encontrar a los morosos. El umbral se fija por **capacidad operativa** (top 20% de riesgo).

## Resultado (test, 1.551 préstamos)
- ROC-AUC 0,805 · PR-AUC 0,235 (un clasificador aleatorio da 0,059: **4 veces mejor que el azar**).
- Marcando el 19% de los préstamos se detecta el **56% de los que caen en mora**, con una concentración de morosos **3 veces** mayor que en la cartera general.
- El Random Forest no supera claramente al baseline logístico en test: con estas variables la señal es mayormente lineal. Ganar más requiere mejores datos, no un modelo más complejo.
- La salida es un **score para ordenar**, no una probabilidad calibrada (se entrenó con pesos balanceados).

### ROI estimado (por mes, ~1.000 préstamos nuevos)
Supuestos: pérdida del 30% del monto por préstamo en mora; la gestión preventiva evita 10% / 25% / 40% de las moras detectadas; costo de gestión por préstamo marcado Gs. 30.000 / 20.000 / 15.000.

| Escenario | Beneficio neto/mes | ROI |
|---|---|---|
| Conservador | ~Gs. 2,0 millones | 36% |
| Base | ~Gs. 15,4 millones | 410% |
| Optimista | ~Gs. 27,8 millones | 987% |

### Prueba con sueldo
Se consiguió un export parcial con el sueldo de los socios. Solo cubre 193 de los 7.755 préstamos (2,5%, 15 morosos). Agregar `sueldo` y la relación cuota/sueldo no cambió las métricas (ROC-AUC 0,792 → 0,791 en CV), así que no se incorporó al modelo final.

## Limitaciones
El target es una foto al corte, no un resultado a horizonte fijo; la exclusión de socios inactivos subestima la mora real; el sueldo solo está para el 2,5% de los préstamos y faltan edad, profesión y forma de cobro para toda la cartera.

## Próximos pasos
1. Extraer de la base el historial de cuotas para un target a 12 meses y validar con split temporal.
2. Exportar el sueldo de toda la cartera e incorporar edad, profesión y forma de cobro (giraduría).
3. Integrar el score al flujo de cobranza preventiva por WhatsApp y medir el impacto real con un grupo de control.

## Archivos y ejecución
- `notebook_final.ipynb`: pipeline completo, ejecutado. Si no encuentra el dataset completo usa `datos_muestra.csv`.
- `model.pkl`: pipeline entrenado (`joblib.load`), incluye `umbral_` y `columnas_`.
- `datos_muestra.csv`: 100 filas anonimizadas.
- `app.py`: demo en Streamlit → `streamlit run app.py`

Requisitos: `pip install pandas scikit-learn==1.8.0 matplotlib joblib streamlit openpyxl xlrd` (el modelo se guardó con scikit-learn 1.8.0; otra versión puede no cargarlo)

*Para la programación se usó un asistente de IA (Claude); los datos, el planteo y las conclusiones son del autor.*
