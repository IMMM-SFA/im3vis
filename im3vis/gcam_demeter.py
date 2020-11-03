import seaborn as sns

from matplotlib import pyplot as plt


def gcam_demeter_region(df, target_year, figure_size=(12, 8), metric_id_col='metric_id', region_col='region',
                        landclass_col='landclass', landclass_list=None, font_scale=1.5):
    """Create a bar plot of GCAM land allocation per each geopolitical region.  Optionally, do
    so for a specific land class.

    :param df:                      Input data frame generated from the GCAM projected input for Demeter.
    :type df:                       data frame

    :param target_year:             Target year column name to examine as a four digit year (e.g., 2000) cast as a string
    :type target_year:              str

    :param figure_size:             x and y integer size for the resulting figure
    :type figure_size:              tuple

    :param metric_id_col:           name of the metric id column in the input data frame
    :type metric_id_col:            str

    :param region_col:              name of the region name column in the input data frame
    :type metric_id_col:            str

    :param landclass_col:           name of the landclass name column in the input data frame
    :type metric_id_col:            str

    :param landclass_list:          a list of land class names to aggregate by region
    :type landclass_list:           list

    :param font_scale:              font scaling factor for seaborn set class
    :type font_scale:               float

    :return:                        axis object

    """
    sns.set(font_scale=font_scale)

    # create a copy of the input data frame
    gcam_reg_df = df.copy()

    # set up plot with custom size
    fig, ax = plt.subplots(figsize=figure_size)

    # extract specific land classes
    if landclass_list is None:
        title_adder = ''
    else:
        gcam_reg_df = gcam_reg_df.loc[gcam_reg_df[landclass_col].isin(landclass_list)]
        title_adder = f"for land classes {','.join(landclass_list)}"

    # drop unneeded fields
    gcam_reg_df.drop(metric_id_col, inplace=True, axis=1)

    # sum by region
    gcam_reg_df = gcam_reg_df.groupby(region_col).sum()

    # sort the data frame by allocation
    gcam_reg_df.sort_values(by=[target_year], inplace=True)

    # generate plot
    g = sns.barplot(x=gcam_reg_df.index, y=target_year, data=gcam_reg_df[[target_year]])
    g.set(xlabel='GCAM Geopolitical Region',
          ylabel=f"{target_year} land allocation (thous km)",
          title=f"GCAM land allocation by region{title_adder}")
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    return g
