import polars as pl


def move_col(df: pl.DataFrame, col_from_id: int, col_to_id: id):
    """Moves a column by id, to reorder the columns of a dataframe

    :param df: the dataframe
    :type df: pl.DataFrame
    :param col_from_id: the id of the column to be moved
    :type col_from_id: int
    :param col_to_id: the new id the column will be moved to
    :type col_to_id: int
    :return: the dataframe with reordered columns
    :rtype: pl.DataFrame
    """
    columns: list = df.columns
    last_col = columns.pop(col_from_id)
    columns.insert(col_to_id, last_col)
    return df.select(columns)
