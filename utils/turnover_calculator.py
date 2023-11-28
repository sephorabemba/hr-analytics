import streamlit as st


def compute_turnover(y_labels):
    """

    :return:
    """
    turnover_nb = y_labels.sum()
    turnover_prc = turnover_nb / len(y_labels) * 100
    return turnover_prc, turnover_nb


def turnover_by_segment(segment_enum, df):
    # eg segment_str = Department, segment_enum = class Dept

    # loop through

    # count
    # segment_group = df.groupby([segment_enum._col_name, "left"])[segment_enum._col_name, "left"].agg( {"left": "sum", segment_enum._col_name: "count"})

    # segment_group = df.groupby([segment_enum._col_name, "left"]).agg(
    #   {"left": "count", segment_enum._col_name: "count"})

    # isolate group by multi-index series with counts. Cf. L0 +++
    gb = df.groupby([segment_enum._col_name, "left"]).size()
    # create df with counts only
    df_segment = gb.to_frame("count")
    # add proportions using the multi-index series of counts -> degroup up to the salary group level then get sum()
    df_segment["proportion"] = round(gb / gb.groupby(level=0).sum() * 100, 2)
    df_segment = df_segment.reset_index()

    # keep turnover data only
    df_turnover = df_segment[df_segment["left"] == 1]
    df_turnover["mean_segment"] = df_turnover["proportion"].sum() / len(df_turnover)

    # compute delta compared to average segment turnover
    df_turnover["delta_prc"] = round(-(1 - df_turnover["proportion"] / df_turnover["mean_segment"]) * 100, 2)

    return df_turnover
