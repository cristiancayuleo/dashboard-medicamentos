import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Distribución de Medicamentos · Quinta Normal",
                   page_icon="💊", layout="wide")

DATA = "datos_preparados.xlsx"
IVA = 1.19
COL = px.colors.qualitative.Safe
MESES = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
         7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre",
         11: "Noviembre", 12: "Diciembre"}


@st.cache_data
def cargar():
    det = pd.read_excel(DATA, sheet_name="detalle")
    det["FECHA CRUCE"] = pd.to_datetime(det["FECHA CRUCE"])
    det["MONTO_IVA"] = det["VALORIZADO ENTREGADO"] * IVA
    det["MES_NUM"] = det["FECHA CRUCE"].dt.month
    return det


det = cargar()


def clp(x):
    return "$" + f"{x:,.0f}".replace(",", ".")


def miles(x):
    return f"{x:,.0f}".replace(",", ".")


def ayuda(texto):
    st.caption("ℹ️ " + texto)


def tabla_prioridad_proveedores(df):
    bp = df.groupby("NOMBRE PROVEEDOR")
    sem = bp.agg(monto_iva=("MONTO_IVA", "sum")).reset_index()
    sem["pct_susp"] = bp["ESTADO CENABAST"].apply(lambda s: (s == "SUSP. X DEUDA").mean() * 100).values
    fsusp = df[df["ESTADO CENABAST"] == "SUSP. X DEUDA"]
    med_susp = fsusp.groupby("NOMBRE PROVEEDOR")["CODIGO GENERICO"].nunique().rename("med_susp").reset_index()
    med_tot = df.groupby("NOMBRE PROVEEDOR")["CODIGO GENERICO"].nunique().rename("med_tot").reset_index()
    ms = fsusp.groupby("NOMBRE PROVEEDOR")["MONTO_IVA"].sum().rename("monto_susp").reset_index()
    sem = (sem.merge(med_susp, on="NOMBRE PROVEEDOR", how="left")
              .merge(med_tot, on="NOMBRE PROVEEDOR", how="left")
              .merge(ms, on="NOMBRE PROVEEDOR", how="left"))
    sem[["med_susp", "monto_susp"]] = sem[["med_susp", "monto_susp"]].fillna(0)
    sem["med_susp"] = sem["med_susp"].astype(int)
    sem["med_sin"] = (sem["med_tot"] - sem["med_susp"]).astype(int)
    total_det = sem["monto_susp"].sum()
    sem["peso_det"] = (sem["monto_susp"] / total_det * 100) if total_det else 0
    sem["Prioridad"] = sem["pct_susp"].map(
        lambda p: "🔴 Alta" if p >= 40 else ("🟡 Media" if p >= 15 else "🟢 Baja"))
    sem = sem[sem["monto_susp"] > 0].sort_values("monto_susp", ascending=False).head(15)
    return pd.DataFrame({
        "Prioridad": sem["Prioridad"], "Proveedor": sem["NOMBRE PROVEEDOR"],
        "Med. con susp.": sem["med_susp"], "Med. sin susp.": sem["med_sin"],
        "Monto detenido (+IVA)": sem["monto_susp"].apply(clp),
        "Peso detenido %": sem["peso_det"].round(1),
        "% Susp. deuda": sem["pct_susp"].round(0),
        "Monto total (+IVA)": sem["monto_iva"].apply(clp)})


def tabla_centros(df):
    rs = ["FALTANTE", "SUSP. X DEUDA"]
    sc = df.groupby("NOMBRE DESTINATARIO").agg(lineas=("ESTADO CENABAST", "size")).reset_index()
    sc["en_riesgo"] = df.groupby("NOMBRE DESTINATARIO")["ESTADO CENABAST"].apply(
        lambda s: s.isin(rs).sum()).values
    sc["pct_riesgo"] = (sc["en_riesgo"] / sc["lineas"] * 100).round(0)
    pr = (df[df["ESTADO CENABAST"].isin(rs)].groupby("NOMBRE DESTINATARIO")
          ["CODIGO GENERICO"].nunique().rename("prod_riesgo").reset_index())
    md = (df[df["ESTADO CENABAST"].isin(rs)].groupby("NOMBRE DESTINATARIO")
          ["MONTO_IVA"].sum().rename("monto_detenido").reset_index())
    sc = sc.merge(pr, on="NOMBRE DESTINATARIO", how="left").merge(md, on="NOMBRE DESTINATARIO", how="left")
    sc["prod_riesgo"] = sc["prod_riesgo"].fillna(0).astype(int)
    sc["monto_detenido"] = sc["monto_detenido"].fillna(0)
    sc["Estado"] = sc["pct_riesgo"].map(
        lambda p: "🔴 Crítico" if p >= 35 else ("🟡 Medio" if p >= 25 else "🟢 Estable"))
    sc = sc.sort_values("pct_riesgo", ascending=False)
    return pd.DataFrame({
        "Estado": sc["Estado"], "Centro": sc["NOMBRE DESTINATARIO"],
        "% en riesgo": sc["pct_riesgo"], "Productos en riesgo": sc["prod_riesgo"],
        "Monto detenido (+IVA)": sc["monto_detenido"].apply(clp)})


# ===================== Barra lateral =====================
st.sidebar.title("Distribución de medicamentos")
st.sidebar.caption("Quinta Normal · ICP")
pagina = st.sidebar.radio(
    "Sección",
    ["Inicio", "Panorama general", "Puntos de entrega", "Proveedores", "Productos", "Segmentación (ML)"])
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
anios = sorted(det["AÑO"].unique())
sel_anios = st.sidebar.multiselect("Año", anios, default=anios)
meses_disp = sorted(det["MES_NUM"].unique())
sel_meses_nom = st.sidebar.multiselect("Mes", [MESES[m] for m in meses_disp],
                                        default=[MESES[m] for m in meses_disp])
sel_meses = [m for m in meses_disp if MESES[m] in sel_meses_nom]
tipos = sorted(det["TIPO DE PRODUCTO"].unique())
sel_tipos = st.sidebar.multiselect("Tipo de producto", tipos, default=tipos)
destinos = sorted(det["NOMBRE DESTINATARIO"].unique())
sel_dest = st.sidebar.multiselect("Punto de entrega", destinos, default=destinos)
st.sidebar.caption("Los filtros se aplican a todas las secciones.")

f = det[
    det["AÑO"].isin(sel_anios)
    & det["MES_NUM"].isin(sel_meses)
    & det["TIPO DE PRODUCTO"].isin(sel_tipos)
    & det["NOMBRE DESTINATARIO"].isin(sel_dest)
].copy()

if f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()


# ===================== Inicio (resumen ejecutivo) =====================
if pagina == "Inicio":
    st.title("Resumen ejecutivo")
    st.caption("Una mirada rápida al estado del abastecimiento de medicamentos de la comuna: "
               "qué está en riesgo, qué ha faltado en el año y con qué proveedores conviene "
               "abrir conversaciones. Para el detalle, usa las secciones de la izquierda.")

    latest = f["FECHA CRUCE"].max()
    mes_lbl = f"{MESES[latest.month]} {latest.year}"
    fm = f[f["FECHA CRUCE"] == latest]
    riesgo_estados = ["FALTANTE", "SUSP. X DEUDA"]
    riesgo_mes = fm[fm["ESTADO CENABAST"].isin(riesgo_estados)]

    c = st.columns(4)
    c[0].metric("Monto del período (+IVA)", clp(f["MONTO_IVA"].sum()))
    c[1].metric("% Aprobado", f"{(f['ESTADO CENABAST']=='APROBADO').mean()*100:.0f}%")
    c[2].metric("% Susp. x deuda", f"{(f['ESTADO CENABAST']=='SUSP. X DEUDA').mean()*100:.0f}%")
    c[3].metric("Proveedores activos", f["NOMBRE PROVEEDOR"].nunique())

    st.markdown(f"**📅 Foco del último mes — {mes_lbl}**")
    d = st.columns(4)
    d[0].metric("Valorizado del mes (+IVA)", clp(fm["MONTO_IVA"].sum()))
    d[1].metric("Monto en riesgo del mes (+IVA)", clp(riesgo_mes["MONTO_IVA"].sum()))
    d[2].metric("Canasta afectada (productos)", riesgo_mes["CODIGO GENERICO"].nunique())
    d[3].metric("Unidades afectadas", miles(riesgo_mes["CANTIDAD UNITARIA A DESPACHAR"].sum()))

    st.divider()
    st.subheader(f"🚨 Medicamentos en riesgo de no llegar — {mes_lbl}")
    if riesgo_mes.empty:
        st.success("Sin medicamentos en riesgo para el último mes con los filtros actuales.")
    else:
        r = (riesgo_mes.groupby(["NOMBRE GENERICO CANONICO", "ESTADO CENABAST", "NOMBRE PROVEEDOR"])
             ["MONTO_IVA"].sum().reset_index().sort_values("MONTO_IVA", ascending=False)
             .rename(columns={"NOMBRE GENERICO CANONICO": "Medicamento", "ESTADO CENABAST": "Estado",
                              "NOMBRE PROVEEDOR": "Proveedor", "MONTO_IVA": "Monto (+IVA)"}))
        st.dataframe(r, use_container_width=True, hide_index=True, height=280)
    ayuda("Productos que este mes figuran como faltantes o suspendidos por deuda: los que "
          "podrían no llegar a los centros.")

    st.markdown(f"#### 🗓️ Acción inmediata — solo el último mes ({mes_lbl})")
    st.info("Estas dos tablas consideran **únicamente el último mes**, para gestión inmediata. "
            "Más abajo encontrarás las mismas tablas para **todo el período** seleccionado.")
    st.markdown("**🚦 Proveedores a priorizar este mes**")
    tpm = tabla_prioridad_proveedores(fm)
    if tpm.empty:
        st.success("Sin proveedores con suspensión por deuda este mes.")
    else:
        st.dataframe(tpm, use_container_width=True, hide_index=True)
    st.markdown("**🏥 Centros afectados este mes**")
    st.dataframe(tabla_centros(fm), use_container_width=True, hide_index=True)
    ayuda("Mismo criterio que las tablas del período (🔴/🟡/🟢), pero acotado al último mes. "
          "Útil para decidir con quién hablar y qué centro reforzar ahora.")

    anio_max = int(f["AÑO"].max())
    st.subheader(f"📉 Medicamentos que más han faltado en {anio_max}")
    fa = f[(f["AÑO"] == anio_max) & (f["ESTADO CENABAST"].isin(riesgo_estados))]
    if fa.empty:
        st.success(f"Sin faltas registradas en {anio_max} con los filtros actuales.")
    else:
        rec = (fa.groupby("NOMBRE GENERICO CANONICO")["MES_NUM"].nunique()
               .rename("Meses con falta").reset_index()
               .rename(columns={"NOMBRE GENERICO CANONICO": "Medicamento"})
               .sort_values("Meses con falta", ascending=False).head(12))
        st.plotly_chart(
            px.bar(rec.sort_values("Meses con falta"), x="Meses con falta", y="Medicamento",
                   orientation="h", color_discrete_sequence=COL,
                   title=f"Top 12 medicamentos con más meses de falta en {anio_max}"),
            use_container_width=True)
    ayuda("Cuántos meses del año cada medicamento estuvo faltante o suspendido. Mientras más "
          "meses, más recurrente es el problema de abastecimiento.")

    st.subheader("🏥 Centros más afectados por la falta")
    cen = f.groupby("NOMBRE DESTINATARIO").agg(lineas=("ESTADO CENABAST", "size")).reset_index()
    cen["en_riesgo"] = f.groupby("NOMBRE DESTINATARIO")["ESTADO CENABAST"].apply(
        lambda s: s.isin(riesgo_estados).sum()).values
    cen["pct_riesgo"] = (cen["en_riesgo"] / cen["lineas"] * 100).round(0)
    st.plotly_chart(
        px.bar(cen.sort_values("pct_riesgo"), x="pct_riesgo", y="NOMBRE DESTINATARIO",
               orientation="h", color="NOMBRE DESTINATARIO", color_discrete_sequence=COL,
               title="% de solicitudes en riesgo por centro",
               labels={"pct_riesgo": "% en riesgo", "NOMBRE DESTINATARIO": ""}),
        use_container_width=True)
    ayuda("Qué centro sufre más la falta: el % de sus solicitudes que quedan faltantes o "
          "suspendidas por deuda. Mientras más alto, más desabastecido queda ese centro (su "
          "detalle está en la sección 'Puntos de entrega').")

    st.subheader("🚦 Proveedores: prioridad de negociación (período completo)")
    st.dataframe(tabla_prioridad_proveedores(f), use_container_width=True, hide_index=True)
    ayuda("Semáforo según el % de solicitudes suspendidas por deuda: 🔴 alta (≥40%), "
          "🟡 media (15–40%), 🟢 baja (<15%). Ordenado por monto detenido: arriba están los "
          "proveedores con más dinero retenido, donde conviene enfocar las conversaciones.")


# ===================== Panorama general =====================
elif pagina == "Panorama general":
    st.title("Panorama general de la distribución")
    st.caption("Visión global de lo que se distribuye a la red de salud comunal. "
               "Usa los filtros de la izquierda para acotar por año, mes, tipo o punto de entrega.")

    with st.expander("📌 Cómo usar este tablero (guía rápida)"):
        st.markdown(
            "- **Panorama general:** la foto completa del gasto y la confiabilidad del abastecimiento.\n"
            "- **Puntos de entrega:** qué y cuánto llega a cada centro de la comuna.\n"
            "- **Proveedores:** en quién concentrar la gestión (peso y deuda).\n"
            "- **Productos:** qué se pide, con qué frecuencia y su evolución entre años.\n"
            "- **Segmentación (ML):** agrupa proveedores con comportamiento parecido.\n\n"
            "Cada gráfico tiene una nota **ℹ️** que explica cómo leerlo. Los **filtros** de la "
            "izquierda se aplican a todas las secciones.")

    c = st.columns(6)
    c[0].metric("Monto final (+IVA)", clp(f["MONTO_IVA"].sum()))
    c[1].metric("Unidades despachadas", miles(f["CANTIDAD UNITARIA A DESPACHAR"].sum()))
    c[2].metric("Proveedores", f["RUT PROVEEDOR"].nunique())
    c[3].metric("Productos (canasta)", f["CODIGO GENERICO"].nunique())
    c[4].metric("% Aprobado", f"{(f['ESTADO CENABAST']=='APROBADO').mean()*100:.0f}%")
    c[5].metric("% Susp. x deuda", f"{(f['ESTADO CENABAST']=='SUSP. X DEUDA').mean()*100:.0f}%")
    ayuda("Tarjetas resumen: el gasto total, cuánto se entregó y qué parte está aprobada vs. "
          "detenida por deuda (lo que la organización puede gestionar).")

    st.divider()
    a, b = st.columns(2)
    serie = f.groupby("FECHA CRUCE")["MONTO_IVA"].sum().reset_index()
    a.plotly_chart(
        px.area(serie, x="FECHA CRUCE", y="MONTO_IVA", title="Monto distribuido por mes (+IVA)",
                labels={"MONTO_IVA": "Monto (+IVA)", "FECHA CRUCE": "Mes"}),
        use_container_width=True)
    a.caption("ℹ️ Cómo leerlo: la evolución del gasto mensual. Sirve para ver estacionalidad "
              "y meses atípicos (ej. caídas en enero).")
    est = f["ESTADO CENABAST"].value_counts().reset_index()
    est.columns = ["Estado", "Líneas"]
    b.plotly_chart(
        px.bar(est, x="Estado", y="Líneas", color="Estado", color_discrete_sequence=COL,
               title="Estado de las solicitudes"),
        use_container_width=True)
    b.caption("ℹ️ Cómo leerlo: cuántas solicitudes hay en cada estado. 'Aprobado' y "
              "'Susp. x deuda' son las que dependen de la gestión interna.")

    st.subheader("Confiabilidad del abastecimiento en el tiempo")
    ts = f.groupby("FECHA CRUCE").agg(
        Aprobado=("ESTADO CENABAST", lambda s: (s == "APROBADO").mean() * 100),
        Suspendido=("ESTADO CENABAST", lambda s: (s == "SUSP. X DEUDA").mean() * 100),
    ).reset_index().melt("FECHA CRUCE", var_name="Estado", value_name="Porcentaje")
    st.plotly_chart(
        px.line(ts, x="FECHA CRUCE", y="Porcentaje", color="Estado", markers=True,
                color_discrete_sequence=COL, labels={"FECHA CRUCE": "Mes"}),
        use_container_width=True)
    ayuda("Cómo usarlo: si la línea de suspendido por deuda sube, el abastecimiento se vuelve "
          "menos confiable ese mes. Picos indican meses que requieren atención de gestión.")


# ===================== Puntos de entrega =====================
elif pagina == "Puntos de entrega":
    st.title("Puntos de entrega")
    st.caption("Qué llega a cada punto de dispensación de la comuna (el consumidor final). "
               "Permite comparar el peso y la confiabilidad entre puntos.")

    g = f.groupby("NOMBRE DESTINATARIO").agg(
        monto_iva=("MONTO_IVA", "sum"),
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        canasta=("CODIGO GENERICO", "nunique")).reset_index()
    g["pct_aprob"] = f.groupby("NOMBRE DESTINATARIO")["ESTADO CENABAST"].apply(
        lambda s: (s == "APROBADO").mean() * 100).values
    g["pct_susp"] = f.groupby("NOMBRE DESTINATARIO")["ESTADO CENABAST"].apply(
        lambda s: (s == "SUSP. X DEUDA").mean() * 100).values

    a, b = st.columns(2)
    a.plotly_chart(
        px.bar(g.sort_values("monto_iva"), x="monto_iva", y="NOMBRE DESTINATARIO",
               orientation="h", color="NOMBRE DESTINATARIO", color_discrete_sequence=COL,
               title="Monto entregado por punto (+IVA)",
               labels={"monto_iva": "Monto (+IVA)", "NOMBRE DESTINATARIO": ""}),
        use_container_width=True)
    a.caption("ℹ️ Cómo leerlo: qué punto concentra más gasto. El más largo es donde se "
              "destina más presupuesto.")
    b.plotly_chart(
        px.bar(g.sort_values("canasta"), x="canasta", y="NOMBRE DESTINATARIO",
               orientation="h", color="NOMBRE DESTINATARIO", color_discrete_sequence=COL,
               title="Canasta por punto (nº de productos distintos)",
               labels={"canasta": "Productos distintos", "NOMBRE DESTINATARIO": ""}),
        use_container_width=True)
    b.caption("ℹ️ Cómo leerlo: la variedad de productos que maneja cada punto. Una canasta "
              "amplia implica logística más compleja.")

    st.subheader("Resumen por punto de entrega")
    tg = g.rename(columns={"NOMBRE DESTINATARIO": "Punto", "monto_iva": "Monto (+IVA)",
                           "unidades": "Unidades", "canasta": "Canasta",
                           "pct_aprob": "% Aprobado", "pct_susp": "% Susp. deuda"})
    tg["% Aprobado"] = tg["% Aprobado"].round(0)
    tg["% Susp. deuda"] = tg["% Susp. deuda"].round(0)
    st.dataframe(tg.sort_values("Monto (+IVA)", ascending=False),
                 use_container_width=True, hide_index=True)
    ayuda("Compara puntos de un vistazo: monto, variedad y qué tan aprobado/suspendido está "
          "cada uno.")

    st.subheader("🚦 Confiabilidad por centro")
    riesgo_estados = ["FALTANTE", "SUSP. X DEUDA"]
    sc = f.groupby("NOMBRE DESTINATARIO").agg(lineas=("ESTADO CENABAST", "size")).reset_index()
    sc["en_riesgo"] = f.groupby("NOMBRE DESTINATARIO")["ESTADO CENABAST"].apply(
        lambda s: s.isin(riesgo_estados).sum()).values
    sc["pct_riesgo"] = (sc["en_riesgo"] / sc["lineas"] * 100).round(0)
    pr = (f[f["ESTADO CENABAST"].isin(riesgo_estados)].groupby("NOMBRE DESTINATARIO")
          ["CODIGO GENERICO"].nunique().rename("prod_riesgo").reset_index())
    md = (f[f["ESTADO CENABAST"].isin(riesgo_estados)].groupby("NOMBRE DESTINATARIO")
          ["MONTO_IVA"].sum().rename("monto_detenido").reset_index())
    sc = sc.merge(pr, on="NOMBRE DESTINATARIO", how="left").merge(md, on="NOMBRE DESTINATARIO", how="left")
    sc["prod_riesgo"] = sc["prod_riesgo"].fillna(0).astype(int)
    sc["monto_detenido"] = sc["monto_detenido"].fillna(0)

    def sem_c(p):
        if p >= 35:
            return "🔴 Crítico"
        if p >= 25:
            return "🟡 Medio"
        return "🟢 Estable"

    sc["Estado"] = sc["pct_riesgo"].map(sem_c)
    sc["monto_detenido"] = sc["monto_detenido"].apply(clp)
    tc = sc.sort_values("pct_riesgo", ascending=False).rename(columns={
        "NOMBRE DESTINATARIO": "Centro", "pct_riesgo": "% en riesgo",
        "prod_riesgo": "Productos en riesgo", "monto_detenido": "Monto detenido (+IVA)"})
    st.dataframe(tc[["Estado", "Centro", "% en riesgo", "Productos en riesgo",
                     "Monto detenido (+IVA)"]], use_container_width=True, hide_index=True)
    ayuda("Semáforo por centro según el % de solicitudes faltantes o suspendidas por deuda: "
          "🔴 crítico (≥35%), 🟡 medio (25–35%), 🟢 estable (<25%). Muestra qué centros quedan "
          "más desabastecidos y cuánto dinero en productos no les llega.")

    st.subheader("¿Qué se entrega en cada punto?")
    punto = st.selectbox("Elige un punto de entrega", destinos)
    fp = f[f["NOMBRE DESTINATARIO"] == punto]
    if not fp.empty:
        top = (fp.groupby("NOMBRE GENERICO CANONICO")["MONTO_IVA"].sum()
               .sort_values(ascending=False).head(12).reset_index())
        st.plotly_chart(
            px.bar(top.sort_values("MONTO_IVA"), x="MONTO_IVA", y="NOMBRE GENERICO CANONICO",
                   orientation="h", title=f"Top 12 productos en {punto} (Monto +IVA)",
                   labels={"MONTO_IVA": "Monto (+IVA)", "NOMBRE GENERICO CANONICO": ""}),
            use_container_width=True)
        ayuda("Cómo usarlo: los productos donde más presupuesto se destina en ese punto; "
              "útil para priorizar negociación y control de stock.")


# ===================== Proveedores =====================
elif pagina == "Proveedores":
    st.title("Proveedores que abastecen la red")
    st.caption("Peso de cada proveedor y dónde se concentra el foco de gestión. "
               "Pasa el mouse sobre los puntos para ver el nombre de cada proveedor.")

    base = f.groupby(["RUT PROVEEDOR", "NOMBRE PROVEEDOR", "GRUPO"])
    g = base.agg(
        monto_iva=("MONTO_IVA", "sum"),
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        n_productos=("CODIGO GENERICO", "nunique")).reset_index()
    g["pct_susp"] = base["ESTADO CENABAST"].apply(
        lambda s: (s == "SUSP. X DEUDA").mean()).values * 100
    # Monto detenido por deuda por proveedor
    ms = (f[f["ESTADO CENABAST"] == "SUSP. X DEUDA"]
          .groupby(["RUT PROVEEDOR", "NOMBRE PROVEEDOR", "GRUPO"])["MONTO_IVA"].sum()
          .rename("monto_susp").reset_index())
    g = g.merge(ms, on=["RUT PROVEEDOR", "NOMBRE PROVEEDOR", "GRUPO"], how="left")
    g["monto_susp"] = g["monto_susp"].fillna(0)
    g["peso_%"] = g["monto_iva"] / g["monto_iva"].sum() * 100

    grupos = sorted(g["GRUPO"].unique())
    sel_g = st.multiselect("Filtrar por grupo de proveedor", grupos, default=grupos)
    g = g[g["GRUPO"].isin(sel_g)]
    topn = st.slider("Mostrar top N proveedores", 5, 30, 10)

    a, b = st.columns(2)
    top = g.sort_values("monto_iva", ascending=False).head(topn)
    a.plotly_chart(
        px.bar(top.sort_values("monto_iva"), x="monto_iva", y="NOMBRE PROVEEDOR",
               orientation="h", color="GRUPO", color_discrete_sequence=COL,
               title=f"Top {topn} proveedores por monto (+IVA)",
               labels={"monto_iva": "Monto (+IVA)", "NOMBRE PROVEEDOR": ""}),
        use_container_width=True)
    a.caption("ℹ️ Cómo leerlo: el peso de cada proveedor en el gasto total. Concentración alta "
              "= mayor dependencia de pocos proveedores.")
    tops = g.sort_values("monto_susp", ascending=False).head(topn)
    b.plotly_chart(
        px.bar(tops.sort_values("monto_susp"), x="monto_susp", y="NOMBRE PROVEEDOR",
               orientation="h", color="GRUPO", color_discrete_sequence=COL,
               title=f"Top {topn} por monto detenido por deuda (+IVA)",
               labels={"monto_susp": "Monto suspendido (+IVA)", "NOMBRE PROVEEDOR": ""}),
        use_container_width=True)
    b.caption("ℹ️ Cómo usarlo: dónde se concentra el dinero retenido por deuda. Estos "
              "proveedores son el foco prioritario para resolver y gestionar.")

    st.subheader("Mapa de proveedores: peso vs. deuda")
    med = g["monto_iva"].median()
    fig = px.scatter(g, x="monto_iva", y="pct_susp", size="n_productos", color="GRUPO",
                     color_discrete_sequence=COL, hover_name="NOMBRE PROVEEDOR",
                     labels={"monto_iva": "Monto (+IVA)", "pct_susp": "% Susp. x deuda"})
    fig.add_vline(x=med, line_dash="dash", line_color="gray")
    fig.add_hline(y=30, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    ayuda("Cómo usarlo: el cuadrante superior derecho (alto monto + alta suspensión por deuda) "
          "reúne a los proveedores prioritarios para gestionar. El tamaño del punto es la "
          "variedad de productos que entrega.")

    st.subheader("Detalle de proveedores")
    tabla = g.sort_values("monto_iva", ascending=False).rename(columns={
        "NOMBRE PROVEEDOR": "Proveedor", "n_productos": "Canasta", "unidades": "Unidades",
        "monto_iva": "Monto (+IVA)", "monto_susp": "Monto susp. deuda",
        "peso_%": "Peso %", "pct_susp": "% Susp. deuda"})
    tabla["Peso %"] = tabla["Peso %"].round(2)
    tabla["% Susp. deuda"] = tabla["% Susp. deuda"].round(0)
    st.dataframe(tabla[["Proveedor", "GRUPO", "Canasta", "Unidades", "Monto (+IVA)",
                        "Monto susp. deuda", "Peso %", "% Susp. deuda"]],
                 use_container_width=True, height=360, hide_index=True)


# ===================== Productos =====================
elif pagina == "Productos":
    st.title("Catálogo de productos y frecuencia")
    st.caption("Volumen, monto y cada cuánto se pide cada producto. Abajo puedes analizar un "
               "producto a través de los años.")

    cc1, cc2 = st.columns(2)
    solo_ctrl = cc1.checkbox("Solo controlados (psicotrópicos)", value=False)
    clasif = sorted(f["Columna1"].dropna().unique())
    sel_cl = cc2.multiselect("Clasificación", clasif, default=clasif)

    pf = f[f["Columna1"].isin(sel_cl)].copy()
    if solo_ctrl:
        pf = pf[pf["CONTROLADO"]]
    if pf.empty:
        st.warning("No hay productos con esos filtros.")
        st.stop()

    pg = pf.groupby(["CODIGO GENERICO", "NOMBRE GENERICO CANONICO"]).agg(
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        monto_iva=("MONTO_IVA", "sum"),
        meses_pedido=("FECHA CRUCE", "nunique")).reset_index()

    k = st.columns(3)
    k[0].metric("Productos", pg.shape[0])
    k[1].metric("Unidades", miles(pg["unidades"].sum()))
    k[2].metric("Monto (+IVA)", clp(pg["monto_iva"].sum()))

    topn = st.slider("Mostrar top N productos", 5, 30, 12)
    a, b = st.columns(2)
    tu = pg.sort_values("monto_iva", ascending=False).head(topn)
    a.plotly_chart(
        px.bar(tu.sort_values("monto_iva"), x="monto_iva", y="NOMBRE GENERICO CANONICO",
               orientation="h", title=f"Top {topn} por monto (+IVA)",
               labels={"monto_iva": "Monto (+IVA)", "NOMBRE GENERICO CANONICO": ""}),
        use_container_width=True)
    a.caption("ℹ️ Cómo leerlo: los productos que concentran más presupuesto. Foco para control "
              "de costos y negociación.")
    tf = pg.sort_values("meses_pedido", ascending=False).head(topn)
    b.plotly_chart(
        px.bar(tf.sort_values("meses_pedido"), x="meses_pedido", y="NOMBRE GENERICO CANONICO",
               orientation="h", title=f"Top {topn} más frecuentes (meses con pedido)",
               labels={"meses_pedido": "Meses con pedido", "NOMBRE GENERICO CANONICO": ""}),
        use_container_width=True)
    b.caption("ℹ️ Cómo leerlo: productos de demanda recurrente (se piden casi todos los meses). "
              "Son los más críticos de no quebrar stock.")

    st.subheader("Un producto a través de los años")
    prod_sel = st.selectbox("Elige un producto", sorted(pf["NOMBRE GENERICO CANONICO"].unique()))
    pp = pf[pf["NOMBRE GENERICO CANONICO"] == prod_sel]
    por_anio = pp.groupby("AÑO").agg(
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        pct_aprob=("ESTADO CENABAST", lambda s: (s == "APROBADO").mean() * 100),
        pct_falta=("ESTADO CENABAST", lambda s: (s.isin(["FALTANTE", "SUSP. X DEUDA"])).mean() * 100),
    ).reset_index()
    por_anio["AÑO"] = por_anio["AÑO"].astype(str)
    a2, b2 = st.columns(2)
    a2.plotly_chart(
        px.bar(por_anio, x="AÑO", y="unidades", title=f"Unidades entregadas por año — {prod_sel}",
               labels={"unidades": "Unidades entregadas", "AÑO": "Año"}),
        use_container_width=True)
    b2.plotly_chart(
        px.line(por_anio, x="AÑO", y="pct_falta", markers=True,
                title="% no entregado (faltante o susp. por deuda)",
                labels={"pct_falta": "% no entregado", "AÑO": "Año"}),
        use_container_width=True)
    ayuda("Cómo usarlo: compara la demanda real (lo entregado) entre años. Ojo: si un producto "
          "se sigue pidiendo pero figura como faltante o suspendido (línea de la derecha alta), "
          "su 'demanda' aparente puede no reflejar el consumo real. Comparar años ayuda a "
          "distinguir demanda genuina de pedidos arrastrados que no se cumplieron.")

    st.subheader("Detalle de productos")
    det_tab = pg.sort_values("monto_iva", ascending=False).rename(columns={
        "NOMBRE GENERICO CANONICO": "Producto", "CODIGO GENERICO": "Código",
        "unidades": "Unidades", "monto_iva": "Monto (+IVA)", "meses_pedido": "Meses pedido"})
    st.dataframe(det_tab, use_container_width=True, height=360, hide_index=True)


# ===================== Segmentación (ML) =====================
elif pagina == "Segmentación (ML)":
    st.title("Segmentación de proveedores (Machine Learning)")
    st.caption("Un modelo agrupa automáticamente a los proveedores con comportamiento parecido "
               "(monto, unidades, variedad y % suspendido por deuda), revelando perfiles para "
               "tratar a cada grupo de forma distinta.")

    base = f.groupby(["RUT PROVEEDOR", "NOMBRE PROVEEDOR", "GRUPO"])
    prov = base.agg(
        monto_iva=("MONTO_IVA", "sum"),
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        n_productos=("CODIGO GENERICO", "nunique")).reset_index()
    prov["pct_susp"] = base["ESTADO CENABAST"].apply(
        lambda s: (s == "SUSP. X DEUDA").mean()).values

    if len(prov) < 6:
        st.warning("Muy pocos proveedores con los filtros actuales para segmentar.")
        st.stop()

    X = prov[["monto_iva", "unidades", "n_productos", "pct_susp"]].copy()
    X["monto_iva"] = np.log1p(X["monto_iva"])
    X["unidades"] = np.log1p(X["unidades"])
    Xs = StandardScaler().fit_transform(X)

    kmax = min(8, len(prov) - 1)
    sils = {}
    for kk in range(2, kmax + 1):
        lab = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(Xs)
        sils[kk] = silhouette_score(Xs, lab)
    k_sug = max(sils, key=sils.get)

    c1, c2 = st.columns([2, 1])
    k = c1.slider("Número de grupos", 2, kmax, k_sug)
    c2.metric("Sugerencia del modelo", f"{k_sug} grupos")
    ayuda("El modelo sugiere el número de grupos que mejor separa a los proveedores. Puedes "
          "moverlo para explorar más o menos grupos.")

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    prov["cluster"] = km.labels_
    perfil = prov.groupby("cluster").agg(
        monto_iva=("monto_iva", "mean"), unidades=("unidades", "mean"),
        n_productos=("n_productos", "mean"), pct_susp=("pct_susp", "mean"))

    def nombrar(perfil):
        nom, vmed = {}, perfil["monto_iva"].median()
        for c, r in perfil.iterrows():
            if r["n_productos"] >= perfil["n_productos"].max() and r["n_productos"] > 100:
                t = "Distribuidor amplio (gran canasta)"
            elif r["monto_iva"] >= vmed and r["pct_susp"] >= 0.30:
                t = "Grande con alta deuda"
            elif r["monto_iva"] >= vmed:
                t = "Grande (alto monto)"
            elif r["pct_susp"] >= 0.30:
                t = "Con deuda relevante"
            else:
                t = "Menor / bajo volumen"
            nom[c] = f"G{c}: {t}"
        return nom

    nombres = nombrar(perfil)
    prov["Perfil"] = prov["cluster"].map(nombres)

    st.plotly_chart(
        px.scatter(prov, x="monto_iva", y="pct_susp", size="n_productos", color="Perfil",
                   color_discrete_sequence=COL, hover_name="NOMBRE PROVEEDOR",
                   title="Grupos de proveedores",
                   labels={"monto_iva": "Monto (+IVA)", "pct_susp": "% Susp. x deuda"}),
        use_container_width=True)
    ayuda("Cómo leerlo: cada color es un perfil de proveedor. Los grupos de alto monto con "
          "alta suspensión por deuda son los que más conviene atender de cerca.")

    st.subheader("Perfil de cada grupo (promedios)")
    tp = perfil.copy()
    tp.index = tp.index.map(nombres)
    tp["pct_susp"] = (tp["pct_susp"] * 100).round(0)
    tp = tp.round(0)
    tp.insert(0, "n_proveedores", prov.groupby("cluster").size().rename(index=nombres))
    tp = tp.rename(columns={"monto_iva": "Monto prom (+IVA)", "unidades": "Unid. prom",
                            "n_productos": "Canasta prom", "pct_susp": "% Susp. deuda"})
    st.dataframe(tp, use_container_width=True)

    with st.expander("¿Cómo se eligió el número de grupos? (silhouette)"):
        sdf = pd.DataFrame({"k": list(sils), "silhouette": [round(v, 3) for v in sils.values()]})
        st.plotly_chart(
            px.line(sdf, x="k", y="silhouette", markers=True,
                    title="Calidad de separación por número de grupos (más alto = mejor)"),
            use_container_width=True)

    st.subheader("Proveedores con su grupo asignado")
    out = prov.sort_values(["cluster", "monto_iva"], ascending=[True, False])[
        ["NOMBRE PROVEEDOR", "GRUPO", "Perfil", "monto_iva", "unidades", "n_productos", "pct_susp"]]
    st.dataframe(out, use_container_width=True, height=360, hide_index=True)
    st.download_button("Descargar grupos (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "proveedores_segmentados.csv", "text/csv")
