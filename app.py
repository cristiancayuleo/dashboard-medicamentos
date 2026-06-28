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


@st.cache_data
def cargar():
    det = pd.read_excel(DATA, sheet_name="detalle")
    det["FECHA CRUCE"] = pd.to_datetime(det["FECHA CRUCE"])
    det["MONTO_IVA"] = det["VALORIZADO ENTREGADO"] * IVA
    det["MES_NUM"] = det["FECHA CRUCE"].dt.month
    return det


MESES = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
         7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre",
         11: "Noviembre", 12: "Diciembre"}


det = cargar()


def clp(x):
    return "$" + f"{x:,.0f}".replace(",", ".")


def miles(x):
    return f"{x:,.0f}".replace(",", ".")


# ===================== Barra lateral =====================
st.sidebar.title("Distribución de medicamentos")
st.sidebar.caption("Quinta Normal · ICP")
pagina = st.sidebar.radio(
    "Sección",
    ["Panorama general", "Puntos de entrega", "Proveedores", "Productos", "Segmentación (ML)"],
)
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

f = det[
    det["AÑO"].isin(sel_anios)
    & det["MES_NUM"].isin(sel_meses)
    & det["TIPO DE PRODUCTO"].isin(sel_tipos)
    & det["NOMBRE DESTINATARIO"].isin(sel_dest)
].copy()

if f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()


# ===================== Panorama general =====================
if pagina == "Panorama general":
    st.title("Panorama general de la distribución")
    st.caption("Visión global de lo que se distribuye a la red de salud comunal.")

    c = st.columns(6)
    c[0].metric("Monto final (+IVA)", clp(f["MONTO_IVA"].sum()))
    c[1].metric("Unidades despachadas", miles(f["CANTIDAD UNITARIA A DESPACHAR"].sum()))
    c[2].metric("Proveedores", f["RUT PROVEEDOR"].nunique())
    c[3].metric("Productos (canasta)", f["CODIGO GENERICO"].nunique())
    c[4].metric("% Aprobado", f"{(f['ESTADO CENABAST']=='APROBADO').mean()*100:.0f}%")
    c[5].metric("% Susp. x deuda", f"{(f['ESTADO CENABAST']=='SUSP. X DEUDA').mean()*100:.0f}%")

    st.divider()
    a, b = st.columns(2)
    serie = f.groupby("FECHA CRUCE")["MONTO_IVA"].sum().reset_index()
    a.plotly_chart(
        px.area(serie, x="FECHA CRUCE", y="MONTO_IVA",
                title="Monto distribuido por mes (+IVA)",
                labels={"MONTO_IVA": "Monto (+IVA)", "FECHA CRUCE": "Mes"}),
        use_container_width=True)
    est = f["ESTADO CENABAST"].value_counts().reset_index()
    est.columns = ["Estado", "Líneas"]
    b.plotly_chart(
        px.bar(est, x="Estado", y="Líneas", color="Estado",
               color_discrete_sequence=COL, title="Estado de las solicitudes"),
        use_container_width=True)

    st.subheader("Confiabilidad del abastecimiento en el tiempo")
    st.caption("Qué porcentaje de lo solicitado se aprueba vs se suspende por deuda, mes a mes.")
    ts = f.groupby("FECHA CRUCE").agg(
        Aprobado=("ESTADO CENABAST", lambda s: (s == "APROBADO").mean() * 100),
        Suspendido=("ESTADO CENABAST", lambda s: (s == "SUSP. X DEUDA").mean() * 100),
    ).reset_index().melt("FECHA CRUCE", var_name="Estado", value_name="Porcentaje")
    st.plotly_chart(
        px.line(ts, x="FECHA CRUCE", y="Porcentaje", color="Estado", markers=True,
                color_discrete_sequence=COL,
                labels={"FECHA CRUCE": "Mes"}),
        use_container_width=True)


# ===================== Puntos de entrega =====================
elif pagina == "Puntos de entrega":
    st.title("Puntos de entrega")
    st.caption("Qué llega a cada punto de dispensación de la comuna (el consumidor final).")

    g = f.groupby("NOMBRE DESTINATARIO").agg(
        monto_iva=("MONTO_IVA", "sum"),
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        canasta=("CODIGO GENERICO", "nunique"),
    ).reset_index()
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
    b.plotly_chart(
        px.bar(g.sort_values("canasta"), x="canasta", y="NOMBRE DESTINATARIO",
               orientation="h", color="NOMBRE DESTINATARIO", color_discrete_sequence=COL,
               title="Canasta por punto (nº de productos distintos)",
               labels={"canasta": "Productos distintos", "NOMBRE DESTINATARIO": ""}),
        use_container_width=True)

    st.subheader("Resumen por punto de entrega")
    tg = g.rename(columns={"NOMBRE DESTINATARIO": "Punto", "monto_iva": "Monto (+IVA)",
                           "unidades": "Unidades", "canasta": "Canasta",
                           "pct_aprob": "% Aprobado", "pct_susp": "% Susp. deuda"})
    tg["% Aprobado"] = tg["% Aprobado"].round(0)
    tg["% Susp. deuda"] = tg["% Susp. deuda"].round(0)
    st.dataframe(tg.sort_values("Monto (+IVA)", ascending=False),
                 use_container_width=True, hide_index=True)

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


# ===================== Proveedores =====================
elif pagina == "Proveedores":
    st.title("Proveedores que abastecen la red")
    st.caption("Peso y comportamiento de cada proveedor en el abastecimiento.")

    base = f.groupby(["RUT PROVEEDOR", "NOMBRE PROVEEDOR", "GRUPO"])
    g = base.agg(
        monto_iva=("MONTO_IVA", "sum"),
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        n_productos=("CODIGO GENERICO", "nunique"),
    ).reset_index()
    g["pct_susp"] = base["ESTADO CENABAST"].apply(
        lambda s: (s == "SUSP. X DEUDA").mean()).values
    g["peso_%"] = g["monto_iva"] / g["monto_iva"].sum() * 100

    grupos = sorted(g["GRUPO"].unique())
    sel_g = st.multiselect("Filtrar por grupo de proveedor", grupos, default=grupos)
    g = g[g["GRUPO"].isin(sel_g)]
    topn = st.slider("Mostrar top N proveedores", 5, 30, 10)
    top = g.sort_values("monto_iva", ascending=False).head(topn)

    a, b = st.columns(2)
    a.plotly_chart(
        px.bar(top.sort_values("monto_iva"), x="monto_iva", y="NOMBRE PROVEEDOR",
               orientation="h", color="GRUPO", color_discrete_sequence=COL,
               title=f"Top {topn} proveedores por monto (+IVA)",
               labels={"monto_iva": "Monto (+IVA)", "NOMBRE PROVEEDOR": ""}),
        use_container_width=True)
    b.plotly_chart(
        px.scatter(g, x="monto_iva", y="pct_susp", size="n_productos", color="GRUPO",
                   color_discrete_sequence=COL, hover_name="NOMBRE PROVEEDOR",
                   title="Perfil: monto vs % suspendido por deuda",
                   labels={"monto_iva": "Monto (+IVA)", "pct_susp": "% Susp. x deuda"}),
        use_container_width=True)

    st.subheader("Detalle de proveedores")
    tabla = g.sort_values("monto_iva", ascending=False).rename(columns={
        "NOMBRE PROVEEDOR": "Proveedor", "n_productos": "Canasta",
        "unidades": "Unidades", "monto_iva": "Monto (+IVA)", "peso_%": "Peso %"})
    tabla["pct_susp"] = (tabla["pct_susp"] * 100).round(0)
    tabla["Peso %"] = tabla["Peso %"].round(2)
    tabla = tabla.rename(columns={"pct_susp": "% Susp. deuda"})
    st.dataframe(tabla[["Proveedor", "GRUPO", "Canasta", "Unidades", "Monto (+IVA)",
                        "Peso %", "% Susp. deuda"]],
                 use_container_width=True, height=380, hide_index=True)


# ===================== Productos =====================
elif pagina == "Productos":
    st.title("Catálogo de productos y frecuencia")
    st.caption("Volumen, monto y cada cuánto se pide cada producto.")

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
        meses_pedido=("FECHA CRUCE", "nunique"),
    ).reset_index()

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
    tf = pg.sort_values("meses_pedido", ascending=False).head(topn)
    b.plotly_chart(
        px.bar(tf.sort_values("meses_pedido"), x="meses_pedido", y="NOMBRE GENERICO CANONICO",
               orientation="h", title=f"Top {topn} más frecuentes (meses con pedido)",
               labels={"meses_pedido": "Meses con pedido", "NOMBRE GENERICO CANONICO": ""}),
        use_container_width=True)

    st.subheader("Detalle de productos")
    det_tab = pg.sort_values("monto_iva", ascending=False).rename(columns={
        "NOMBRE GENERICO CANONICO": "Producto", "CODIGO GENERICO": "Código",
        "unidades": "Unidades", "monto_iva": "Monto (+IVA)", "meses_pedido": "Meses pedido"})
    st.dataframe(det_tab, use_container_width=True, height=380, hide_index=True)


# ===================== Segmentación (ML) =====================
elif pagina == "Segmentación (ML)":
    st.title("Segmentación de proveedores (Machine Learning)")
    st.caption("Un modelo de clustering agrupa a los proveedores según su comportamiento: "
               "monto, unidades, diversidad de canasta y % suspendido por deuda.")

    base = f.groupby(["RUT PROVEEDOR", "NOMBRE PROVEEDOR", "GRUPO"])
    prov = base.agg(
        monto_iva=("MONTO_IVA", "sum"),
        unidades=("CANTIDAD UNITARIA A DESPACHAR", "sum"),
        n_productos=("CODIGO GENERICO", "nunique"),
    ).reset_index()
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
