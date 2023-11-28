import streamlit as st

from cache.dataset import load_full_data, load_recos
from cache.model import get_feat_importance, load_model
from utils.html_formatter import format_etiquettes_recos
from utils.utils import (
    Factor,
)

st.set_page_config(
    page_title="Recommendations",
    page_icon="💡"
)

dict_data = load_full_data()
if "dict_data" not in st.session_state:
    dict_data = load_full_data()
    st.session_state["dict_data"] = dict_data

pipeline = load_model()
df_importance = get_feat_importance(_pipeline=pipeline)

# load recommendations
df_reco = load_recos()

# =================== Recommendations ===================

st.subheader("Recommendations", divider=True)
st.write(
    "These recommendations aim at supporting data-driven decisions to "
    "reduce employee turnover. Generated with ChatGPT."
)

factors_to_select = sorted(Factor.get_factor_formatted())
factors_selected = st.multiselect(
    label="Choose 1 or more factors",
    options=factors_to_select,
    default=None,
)

if factors_selected:
    # df_reco
    # get factor ids
    factor_ids = Factor.ids_from_formatted(factors_selected)
    # select targeted reco lines only
    mask = df_reco["id_factor"].isin(factor_ids)
    reco_selected = df_reco[mask]
    reco_selected["factor_fmt"] = reco_selected["id_factor"].apply(lambda id: Factor.id_to_formatted(id))
    reco_selected

    # format etiquettes for each factor
    etiquettes = format_etiquettes_recos(df_reco=reco_selected, df_importance=df_importance)
    # show etiquettes
    for e in etiquettes:
        st.markdown(e, unsafe_allow_html=True)
