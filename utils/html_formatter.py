import pandas as pd

from utils.utils import Factor


def build_etiquette_style(risky=None, top=None, section="Segmentation"):
    # common to all sections
    if risky is not None:
        color = "red" if risky else "green"
    else:
        color = "green" if top else "black"
    etiquette_body = """
                border: 2px solid;
                border-color: {border_color};
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 2px;
    """

    segment_cat_body = """
                        font-weight: bold;
                        color: {cat_color};
    """
    segment_cat_body = segment_cat_body.format(cat_color=color)

    end = """
                    }
                </style>
            """

    # for segmentation section only
    etiquette_body = etiquette_body.format(border_color=color)
    if section == "Segmentation":
        start = """
            <style>
                .etiquette_risky {
    """ if risky else """
            <style>
                .etiquette_safe {
    """
        cat_label = """
                }
                .segment_cat_risky {
        """ if risky else """
                }
                .segment_cat_safe {
        """
        return start + etiquette_body + cat_label + segment_cat_body + end

    # for Recommendations section only
    # build style for Reco etiquette
    start = """
                <style>
                    .etiquette_top {
        """ if top else """
                <style>
                    .etiquette_basic {
        """
    cat_label = """
            }
            .segment_cat_top {
    """ if top else """
            }
            .segment_cat_basic {
    """
    return start + etiquette_body + cat_label + segment_cat_body + end


def format_etiquette(segment_cat, delta=None, rank=None, reco_block=None):
    """Format a single etiquette

    :param segment_cat:
    :param delta:
    :param top:
    :param reco_block:
    :return:
    """
    #
    comp = None
    style = None
    top_msg = ""
    if delta is not None:
        risky = delta > 0
        if risky:
            # risky
            comp = "more"
            segment_cat_class = "segment_cat_risky"
            etiquette_class = "etiquette_risky"
        else:
            # safe
            comp = "less"
            delta = -delta

            segment_cat_class = "segment_cat_safe"
            etiquette_class = "etiquette_safe"
        style = build_etiquette_style(risky=risky)
        # prepare message body
        msg = f"<p>{delta}% {comp} than the average turnover rate</p>"
    elif rank is not None:
        top = rank <= 3
        if top:
            segment_cat_class = "segment_cat_top"
            etiquette_class = "etiquette_top"
        else:
            segment_cat_class = "segment_cat_basic"
            etiquette_class = "etiquette_basic"
        style = build_etiquette_style(risky=None, top=top, section="Reco")
        msg = reco_block
        top_msg = f" - Top #{rank} factor"
    else:
        print("Error, delta or top must be not null")
    # build etiquette block
    etiquette_txt = f"""
<div class= '{etiquette_class}'>
    <p class='{segment_cat_class}'>{segment_cat}{top_msg}</p>{msg}
</div>
"""
    etiquette_fmt = style + etiquette_txt
    etiquette_fmt = etiquette_fmt.strip()
    return etiquette_fmt


def format_etiquettes_recos(df_reco, df_importance):
    # get unique factor ids
    ids = df_reco["id_factor"].unique().tolist()
    # get corresponding formatted names
    fmt_factors = Factor.ids_to_formatted(ids)

    start = "<ul>"
    end = "</ul>"
    etiquettes = list()
    for fmt_factor in fmt_factors:
        # get all recos for given factor as a list
        recos = df_reco.loc[df_reco["factor_fmt"] == fmt_factor, "reco"].tolist()
        reco_html = start
        # add recos as html
        for reco in recos:
            reco_html += "<li>" + reco + "</li>"
        reco_html += end

        # get single factor importance rank
        # df_importance.apply(lambda row: row["Factor"])
        # rank = df_reco[df_importance["Factor"] == fmt_factor]["rank"].values.tolist()[0]
        rank = df_importance[df_importance["Factor"] == fmt_factor]["rank"].values.tolist()[0]
        # add 1 for display
        rank += 1
        # format etiquette
        etiquette = format_etiquette(segment_cat=fmt_factor, delta=None, rank=rank, reco_block=reco_html)

        # add to block list
        etiquettes.append(etiquette)

    return etiquettes


def format_etiquettes(tr_by_segment, segment_enum, col_delta):
    """

    :return:
    """
    col_segment = segment_enum._col_name
    df_cols = ["category", "delta_prc", "etiquette"]
    df_list = list()
    for segment in segment_enum:
        # get category and delta value
        category = segment.formatted
        tr_by_segment.loc[tr_by_segment[col_segment] == category, col_delta].values.tolist()
        delta = tr_by_segment.loc[tr_by_segment[col_segment] == category, col_delta].values.tolist()[0]
        etiquette = format_etiquette(segment_cat=category, delta=delta)
        df_list.append(pd.DataFrame([[category, delta, etiquette]], columns=df_cols, index=[0]))

    # order etiquettes by delta
    df_etiquettes = pd.concat(df_list, axis=0)
    df_etiquettes = df_etiquettes.sort_values(by="delta_prc", ascending=False)
    etiquettes = df_etiquettes["etiquette"].values.tolist()
    return etiquettes
