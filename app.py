import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Distribución de Medicamentos · Cenabast Comunal",
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
st.sidebar.caption("Cenabast Comunal")
pagina = st.sidebar.radio(
    "Sección",
    ["Inicio", "Panorama general", "Puntos de entrega", "Proveedores", "Productos", "Negociación inteligente (ML)"])
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
    st.subheader(f"🚨 Medicamentos con suspensión por deuda Cenabast — {mes_lbl}")
    if riesgo_mes.empty:
        st.success("Sin medicamentos en riesgo para el último mes con los filtros actuales.")
    else:
        r = (riesgo_mes.groupby(["NOMBRE GENERICO CANONICO", "ESTADO CENABAST", "NOMBRE PROVEEDOR"])
             ["MONTO_IVA"].sum().reset_index().sort_values("MONTO_IVA", ascending=False)
             .rename(columns={"NOMBRE GENERICO CANONICO": "Medicamento", "ESTADO CENABAST": "Estado",
                              "NOMBRE PROVEEDOR": "Proveedor", "MONTO_IVA": "Monto (+IVA)"}))
        r["Monto (+IVA)"] = r["Monto (+IVA)"].apply(clp)
        st.dataframe(r, use_container_width=True, hide_index=True, height=280)
    ayuda("Productos que este mes figuran como faltantes o suspendidos por deuda: los que "
          "podrían no llegar a los centros.")

    st.markdown(f"#### 🗓️ Acción inmediata — mes ({mes_lbl})")
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

    st.subheader("🚦 Proveedores: prioridad de negociación (filtrar en caso de querer algún año y/o mes en particular)")
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
            "- **Negociación inteligente (ML):** clasificación ABC, grupos de proveedores y simulador de ahorro.\n\n"
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
    est.columns = ["Estado", "Cantidad Productos"]
    b.plotly_chart(
        px.bar(est, x="Estado", y="Cantidad Productos", color="Estado", color_discrete_sequence=COL,
               title="Estado de las solicitudes (filtrar en caso de querer algún año y/o mes en particular)"),
        use_container_width=True)
    b.caption("ℹ️ Cómo leerlo: cuántas solicitudes hay en cada estado. 'Aprobado' y "
              "'Susp. x deuda' son las que dependen de la gestión interna.")

    st.subheader("Confiabilidad del abastecimiento en el tiempo (Este gráfico deja fuera a los productos Suspendidos , Faltantes, Eliminados y Pendientes, puesto que, dichos estados son exclusivamente potestad de Cenabast)")
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
    tg = tg.sort_values("Monto (+IVA)", ascending=False)
    tg["% Aprobado"] = tg["% Aprobado"].round(0)
    tg["% Susp. deuda"] = tg["% Susp. deuda"].round(0)
    tg["Monto (+IVA)"] = tg["Monto (+IVA)"].apply(clp)
    tg["Unidades"] = tg["Unidades"].apply(miles)
    st.dataframe(tg, use_container_width=True, hide_index=True)
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
    tabla["Unidades"] = tabla["Unidades"].apply(miles)
    tabla["Monto (+IVA)"] = tabla["Monto (+IVA)"].apply(clp)
    tabla["Monto susp. deuda"] = tabla["Monto susp. deuda"].apply(clp)
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
    det_tab["Unidades"] = det_tab["Unidades"].apply(miles)
    det_tab["Monto (+IVA)"] = det_tab["Monto (+IVA)"].apply(clp)
    st.dataframe(det_tab, use_container_width=True, height=360, hide_index=True)


# ===================== Negociación inteligente (ML + ABC) =====================
elif pagina == "Negociación inteligente (ML)":
    st.title("Negociación inteligente: ABC + Machine Learning")
    st.caption("Prioriza dónde negociar combinando tres miradas: clasificación ABC (qué productos "
               "concentran el gasto), segmentación de proveedores con un modelo de ML, y un "
               "simulador de ahorro por negociación.")

    HOLDINGS = {"OPKO CHILE S.A.": "OPKO–ARAMA (holding)",
                "ARAMA NATURAL PRODUCTS DISTRIBUIDOR": "OPKO–ARAMA (holding)"}
    consolidar = st.checkbox(
        "Consolidar holdings (tratar OPKO y ARAMA como un solo proveedor para negociar)", value=True)
    fx = f.copy()
    fx["PROVEEDOR_NEG"] = (fx["NOMBRE PROVEEDOR"].map(lambda x: HOLDINGS.get(x, x))
                           if consolidar else fx["NOMBRE PROVEEDOR"])

    # ---------- 1) Clasificación ABC ----------
    st.subheader("1️⃣ Clasificación ABC de productos (costeo ABC)")
    abc = (fx.groupby("NOMBRE GENERICO CANONICO")
           .agg(monto=("MONTO_IVA", "sum"), unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"))
           .sort_values("monto", ascending=False).reset_index())
    abc["acum_%"] = abc["monto"].cumsum() / abc["monto"].sum() * 100
    abc["Clase"] = abc["acum_%"].map(lambda a: "A" if a <= 80 else ("B" if a <= 95 else "C"))
    res = abc.groupby("Clase").agg(productos=("NOMBRE GENERICO CANONICO", "count"),
                                   monto=("monto", "sum")).reset_index()
    res["% del gasto"] = (res["monto"] / res["monto"].sum() * 100).round(0)
    clase_map = abc.set_index("NOMBRE GENERICO CANONICO")["Clase"]
    fx_cl = fx.assign(Clase=fx["NOMBRE GENERICO CANONICO"].map(clase_map))
    emp = fx_cl.groupby("Clase")["PROVEEDOR_NEG"].nunique().rename("empresas").reset_index()
    res = res.merge(emp, on="Clase", how="left")
    cc = st.columns(3)
    for i, cls in enumerate(["A", "B", "C"]):
        row = res[res["Clase"] == cls]
        if not row.empty:
            cc[i].metric(f"Clase {cls}", f"{int(row['productos'].iloc[0])} productos",
                         f"{row['% del gasto'].iloc[0]:.0f}% del gasto")
    rest = res.copy()
    rest["monto"] = rest["monto"].apply(clp)
    rest = rest.rename(columns={"productos": "Productos", "empresas": "Empresas",
                                "monto": "Monto (+IVA)"})
    st.dataframe(rest[["Clase", "Productos", "Empresas", "Monto (+IVA)", "% del gasto"]],
                 use_container_width=True, hide_index=True)
    ayuda("Ojo: en esta tabla una misma empresa puede contarse en varias clases (vende productos "
          "A, B y C). La tabla siguiente la cuenta una sola vez.")

    st.markdown("**Proveedores por su clase de mayor prioridad (sin doble conteo)**")
    rank = {"A": 1, "B": 2, "C": 3}
    inv = {1: "A", 2: "B", 3: "C"}
    fx_cl["rk"] = fx_cl["Clase"].map(rank)
    pb = fx_cl.groupby("PROVEEDOR_NEG").agg(rk=("rk", "min"),
                                            monto=("MONTO_IVA", "sum")).reset_index()
    pb["Clase proveedor"] = pb["rk"].map(inv)
    res2 = (pb.groupby("Clase proveedor")
            .agg(empresas=("PROVEEDOR_NEG", "count"), monto=("monto", "sum"))
            .reset_index().sort_values("Clase proveedor"))
    res2["% del gasto"] = (res2["monto"] / res2["monto"].sum() * 100).round(0)
    res2["monto"] = res2["monto"].apply(clp)
    st.dataframe(res2.rename(columns={"empresas": "Empresas", "monto": "Monto total (+IVA)"})[
        ["Clase proveedor", "Empresas", "Monto total (+IVA)", "% del gasto"]],
        use_container_width=True, hide_index=True)
    ayuda("Aquí cada proveedor cuenta UNA sola vez, en su clase más alta: si tiene algún producto "
          "clase A es 'proveedor A' (prioridad), porque al negociar con él se incluyen también "
          "sus productos B y C. Así ves cuántos proveedores son realmente prioritarios y cuánto "
          "gasto representan.")
    figp = px.bar(abc.head(20), x="NOMBRE GENERICO CANONICO", y="monto", color="Clase",
                  color_discrete_map={"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c"},
                  title="Top 20 productos por gasto (Pareto)",
                  labels={"monto": "Monto (+IVA)", "NOMBRE GENERICO CANONICO": "Producto"})
    figp.update_xaxes(showticklabels=False)
    st.plotly_chart(figp, use_container_width=True)
    ayuda("La clase A son los pocos productos que concentran ~80% del gasto: ahí cada peso "
          "negociado rinde más. En productos básicos de alto volumen, una rebaja pequeña por "
          "unidad genera grandes ahorros (ver simulador abajo).")

    st.markdown("**Proveedores más importantes según ABC**")
    abcp = (fx.groupby("PROVEEDOR_NEG")
            .agg(monto=("MONTO_IVA", "sum"), unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
                 canasta=("CODIGO GENERICO", "nunique"))
            .sort_values("monto", ascending=False).reset_index())
    abcp["% del gasto"] = (abcp["monto"] / abcp["monto"].sum() * 100)
    abcp["acum_%"] = abcp["% del gasto"].cumsum()
    abcp["Clase"] = abcp["acum_%"].map(lambda a: "A" if a <= 80 else ("B" if a <= 95 else "C"))
    tabc = pd.DataFrame({
        "Clase": abcp["Clase"],
        "Proveedor": abcp["PROVEEDOR_NEG"],
        "Monto (+IVA)": abcp["monto"].apply(clp),
        "% del gasto": abcp["% del gasto"].round(1),
        "% acumulado": abcp["acum_%"].round(0),
        "Unidades": abcp["unidades"].apply(miles),
        "Canasta": abcp["canasta"],
    }).head(20)
    st.dataframe(tabc, use_container_width=True, hide_index=True, height=360)
    ayuda("Los proveedores que concentran la mayor parte del gasto (clase A) son los de mayor "
          "impacto: negociar mejores precios con ellos mueve la aguja del presupuesto. Con el "
          "holding consolidado, OPKO–ARAMA sube como uno de los más relevantes.")

    # ---------- 2) Segmentación de proveedores (K-Means) ----------
    st.subheader("2️⃣ Segmentación de proveedores (modelo de ML · K-Means)")
    st.caption("El modelo agrupa proveedores con comportamiento parecido (no predice: descubre "
               "patrones). Cada grupo se gestiona de forma distinta.")

    with st.expander("📘 ¿Qué es esto y cómo me sirve? (explicación con ejemplo)"):
        st.markdown(
            "**El problema.** Tenemos ~180 proveedores. Revisarlos uno por uno para decidir a "
            "quién priorizar es inviable.\n\n"
            "**Qué hace el modelo.** *K-Means* es un modelo de Machine Learning que los **agrupa "
            "automáticamente** según se parezcan en cuatro cosas: cuánto dinero mueven, cuántas "
            "unidades, qué tan variada es su canasta y qué % tienen suspendido por deuda. "
            "Nosotros no le damos las reglas: el modelo descubre solo los grupos.\n\n"
            "**Para qué sirve.** En vez de 180 casos sueltos, quedan unos pocos **tipos de "
            "proveedor**, y a cada tipo le aplicas una estrategia: a los *grandes con alta deuda* "
            "→ negociación prioritaria; a los *menores* → seguimiento liviano.\n\n"
            "**Cómo leer el gráfico.** Cada punto es un proveedor. Más a la **derecha** = mueve "
            "más dinero. Más **arriba** = más % suspendido por deuda. El **tamaño** del punto es "
            "la variedad de productos. El **color** es el grupo que el modelo le asignó.\n\n"
            "**Ejemplo.** Dos proveedores que venden productos totalmente distintos, pero que "
            "ambos mueven mucho dinero y tienen alta deuda, caen en el **mismo grupo** — porque "
            "para la gestión piden la misma acción: sentarse a negociar la deuda. Ese grupo "
            "(arriba a la derecha) es tu foco.\n\n"
            "*El número de grupos lo sugiere el propio modelo; puedes moverlo para ver más o "
            "menos detalle.*")

    base = fx.groupby("PROVEEDOR_NEG")
    prov = base.agg(monto_iva=("MONTO_IVA", "sum"),
                    unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
                    n_productos=("CODIGO GENERICO", "nunique")).reset_index()
    prov["pct_susp"] = base["ESTADO CENABAST"].apply(lambda s: (s == "SUSP. X DEUDA").mean()).values

    if len(prov) >= 6:
        X = prov[["monto_iva", "unidades", "n_productos", "pct_susp"]].copy()
        X["monto_iva"] = np.log1p(X["monto_iva"])
        X["unidades"] = np.log1p(X["unidades"])
        Xs = StandardScaler().fit_transform(X)
        kmax = min(8, len(prov) - 1)
        sils = {kk: silhouette_score(Xs, KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(Xs))
                for kk in range(2, kmax + 1)}
        k_sug = max(sils, key=sils.get)
        k = st.slider("Número de grupos", 2, kmax, k_sug)
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        prov["cluster"] = km.labels_
        perfil = prov.groupby("cluster").agg(monto_iva=("monto_iva", "mean"),
                                             pct_susp=("pct_susp", "mean"),
                                             n_productos=("n_productos", "mean"))

        def nombrar(perfil):
            nom, vmed = {}, perfil["monto_iva"].median()
            for c, r in perfil.iterrows():
                if r["n_productos"] >= perfil["n_productos"].max() and r["n_productos"] > 100:
                    t = "Distribuidor amplio"
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
                       color_discrete_sequence=COL, hover_name="PROVEEDOR_NEG",
                       title="Grupos de proveedores",
                       labels={"monto_iva": "Monto (+IVA)", "pct_susp": "% Susp. x deuda"}),
            use_container_width=True)
        ayuda("Cada color es un perfil. Los grupos de alto monto con alta suspensión por deuda son "
              "los que más conviene atender (ej.: el holding OPKO–ARAMA si está consolidado).")

        st.markdown("**Perfil de cada grupo (promedios)** — qué representa cada color")
        tp = perfil.copy()
        tp.index = tp.index.map(nombres)
        tp = pd.DataFrame({
            "Proveedores": prov.groupby("cluster").size().rename(index=nombres),
            "Monto prom. (+IVA)": tp["monto_iva"].apply(clp),
            "% Susp. deuda prom.": (tp["pct_susp"] * 100).round(0),
            "Canasta prom.": tp["n_productos"].round(0),
        })
        st.dataframe(tp, use_container_width=True)
        ayuda("Lee cada fila como un 'tipo' de proveedor: cuántos hay, cuánto mueven en promedio, "
              "qué % de deuda y qué tan amplia es su canasta. Así sabes qué hacer con cada grupo.")
    else:
        st.info("Muy pocos proveedores con los filtros actuales para segmentar.")

    # ---------- 3) Simulador de ahorro ----------
    st.subheader("3️⃣ Simulador de ahorro por negociación")
    st.caption("En productos básicos de alto volumen, una rebaja pequeña por unidad se multiplica. "
               "Simula cuánto se podría ahorrar y cuánta deuda cubriría.")
    prov_sel = st.selectbox("Proveedor / holding a negociar", sorted(fx["PROVEEDOR_NEG"].unique()))
    fps = fx[fx["PROVEEDOR_NEG"] == prov_sel]
    unidades = int(fps["CANTIDAD UNITARIA A DESPACHAR"].sum())
    detenido = fps[fps["ESTADO CENABAST"] == "SUSP. X DEUDA"]["MONTO_IVA"].sum()
    ahorro_u = st.slider("Ahorro negociado por unidad ($)", 0, 50, 5)
    ahorro_total = unidades * ahorro_u
    s = st.columns(3)
    s[0].metric("Unidades del proveedor", miles(unidades))
    s[1].metric("Ahorro por unidad", clp(ahorro_u))
    s[2].metric("Ahorro potencial total", clp(ahorro_total))
    if detenido > 0:
        cob = ahorro_total / detenido * 100
        st.markdown(f"Ese ahorro cubriría el **{cob:.0f}%** del monto hoy detenido por deuda con "
                    f"este proveedor ({clp(detenido)}).")
    ayuda("Ejemplo: si el proveedor entrega 2.000.000 de unidades y negocias $30 menos por unidad, "
          "el ahorro es $60.000.000 — recursos que pueden destinarse a reducir deuda.")
