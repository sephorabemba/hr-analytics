import numpy as np
import streamlit as st

from cache.dataset import load_full_data
from cache.model import load_model
from utils.html_formatter import format_etiquettes
from utils.turnover_calculator import turnover_by_segment
from utils.utils import (
    segments,
)

st.set_page_config(
    page_title="Risk By Segment",
    page_icon="⚠️"
)

dict_data = load_full_data()
if "dict_data" not in st.session_state:
    dict_data = load_full_data()
    st.session_state["dict_data"] = dict_data

df_test = dict_data["df_test"]
pipeline = load_model()

# =================== RISK BY SEGMENT ===================
# segments P1: department, tenure
st.subheader("Risk by segment", divider=True)

# department
departments = df_test["department_fmt"].unique().tolist()
departments = sorted(departments)

# segments
segments_to_select = segments.keys()
select_segment = st.selectbox(
    label="Select the segment to analyze",
    options=segments_to_select
)

if select_segment:
    # get corresponding enum object
    for s in segments_to_select:
        if select_segment == s:
            # get predicted turnovers by department
            segment_enum = segments[s]
            tr_by_segment = turnover_by_segment(segment_enum=segment_enum, df=df_test)
            tr_by_segment

            # display turnovers tn by dept
            etiquettes = format_etiquettes(tr_by_segment=tr_by_segment, segment_enum=segment_enum,
                                           col_delta="delta_prc")
            nb_etiquettes = len(etiquettes)
            nb_cols = 2
            nb_rows = int(np.ceil(nb_etiquettes / nb_cols))
            # create nb_cols columns nb_rows times
            etiquette_pos = 0
            for row_index in range(nb_rows):
                col1, col2 = st.columns(2)
                if etiquette_pos < nb_etiquettes:
                    col1.markdown(etiquettes[etiquette_pos], unsafe_allow_html=True)
                etiquette_pos += 1
                if etiquette_pos < nb_etiquettes:
                    col2.markdown(etiquettes[etiquette_pos], unsafe_allow_html=True)
                    etiquette_pos += 1
