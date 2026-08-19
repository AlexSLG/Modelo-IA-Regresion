"""Aplicación principal — Modelos de Inteligencia Artificial.

Menú de casos de estudio. Los casos implementados se importan desde
``cases.registry``; los pendientes se muestran como "Próximamente".
"""

import streamlit as st

from cases.registry import CASES, render_case

st.set_page_config(
    page_title="Modelos de Inteligencia Artificial",
    page_icon="📊",
    layout="wide",
)


def render_home() -> None:
    st.title("MODELOS DE INTELIGENCIA ARTIFICIAL")
    st.subheader("Seleccione un caso de estudio")
    st.write(
        "Aplicación de modelos de regresión aplicados a diferentes "
        "problemas de predicción."
    )

    for case in CASES:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### {case['title']}")
                st.write(case["description"])
            with c2:
                if case["status"] == "available":
                    if st.button(case["button"], key=case["id"], use_container_width=True):
                        st.session_state["page"] = case["id"]
                        st.rerun()
                else:
                    st.markdown("**:gray[Próximamente]**")
                    if st.button("Ingresar al Caso", key=case["id"], disabled=True,
                                 use_container_width=True):
                        pass


def main() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    page = st.session_state["page"]
    if page == "home":
        render_home()
    else:
        try:
            render_case(page)
        except Exception as exc:  # noqa: BLE001
            st.error(f"El caso '{page}' no pudo cargarse: {exc}")
            if st.button("Volver al menú principal"):
                st.session_state["page"] = "home"
                st.rerun()


main()