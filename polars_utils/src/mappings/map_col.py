from typing import Union, List, Dict, Any, Callable

import polars as pl


def map_col(
        df: pl.DataFrame,
        map_from: Union[str, List[str]], map_to: Union[str, List[str]],
        mapping: Union[Dict[Any, Any], Callable[[Any], Any]],
        default: Any = None) -> pl.DataFrame:
    """
    Map values in column to new values using dictionary or mapping function

    When mapping with a function, the function should only take a single positional
    argument `args`. The function should not have any side effects.

    Avoid mapping with a function if possible, as it is much slower than mapping with a
    dictionary. Therefore, avoid using functions as the one shown below.

    ```def map_func(x):
        return dictionary[x]```


    When mapping multiple columns with a function, the function should not take
    positional or keyword arguments, but use `args`. Indexing can be
    used to access the values of the columns to be mapped. Make sure to pass the
    columns in the correct order.

    Example:
    ```def map_func(*args):
        return 2*args[0] + args[1] / args[2]```

    :param df: The dataframe
    :type df: pl.DataFrame
    :param map_from: the name or names of the column(s) to map from (in correct order)
    :type map_from: Union[str, List[str]]
    :param map_to:  the name or names of the column(s) to be created as a result of the mapping
    :type map_to: Union[str, List[str]]
    :param mapping: a dictionary or function to mapping with
    :type mapping: Union[Dict[Any, Any], Callable[[...], Any]]
    :param default: the default value to use if no match is found in the dictionary
    :type default: Any
    :return: the dataframe with the new column
    :rtype: pl.DataFrame
    :raises: ValueError: if map_from is a list and mapping is a dictionary
    """
    if isinstance(mapping, dict):
        if isinstance(map_from, list):  # multiple columns
            raise ValueError('When mapping with a dictionary, map_from must be a '
                             'single column and not a list of columns.')
        return _map_col_dict(df=df,
                             col_name=map_from, new_col_name=map_to,
                             dictionary=mapping, default=default
                             )
    elif callable(mapping):
        if isinstance(map_from, str):  # 1 to ...
            if isinstance(map_to, str):  # 1 to 1
                return _map_1to1_function(df=df,
                                          col_name=map_from, new_col_name=map_to,
                                          function=mapping
                                          )
            elif isinstance(map_to, list):  # 1 to n
                return _map_1ton_function(df=df,
                                          col_name=map_from, new_col_names=map_to,
                                          function=mapping
                                          )

        elif isinstance(map_from, list):  # n to ...
            if isinstance(map_to, str):  # n to 1
                return _map_nto1_function(df=df,
                                          col_names=map_from, new_col_name=map_to,
                                          function=mapping
                                          )
            elif isinstance(map_to, list):  # m to n
                return _map_ntom_function(df=df,
                                          col_names=map_from, new_col_names=map_to,
                                          function=mapping
                                          )


def _map_col_dict(df: pl.DataFrame,
                  col_name: str, new_col_name: str,
                  dictionary: Dict[Any, Any], default: Any = None) -> pl.DataFrame:
    """
    Map values in column to new values using dictionary

    Helper function for map_col

    :param df: The dataframe
    :type df: pl.DataFrame
    :param col_name: the name of the column to map from
    :type col_name: str
    :param new_col_name: name of the new column to be created as a result of the mapping
    :type new_col_name: str
    :param dictionary: a dictionary to mapping with
    :type dictionary: Dict[Any, Any]
    :param default: the default value to use if no match is found in the dictionary
    :type default: Any
    :return: the dataframe with the new column
    :rtype: pl.DataFrame
    """
    return df.with_columns(
        pl.col(col_name).map_dict(dictionary, default=default).alias(new_col_name)
    )


def _map_1to1_function(df: pl.DataFrame,
                       col_name: str, new_col_name: str,
                       function: Callable[[...], Any]) -> pl.DataFrame:
    """
    Map values in column to new values using function

    Helper function for map_col

    :param df: The dataframe
    :type df: pl.DataFrame
    :param col_name: the name of the column to map from
    :type col_name: str
    :param new_col_name: name of the new column to be created as a result of the mapping
    :type new_col_name: str
    :param function: a function to mapping with
    :type function: Callable[[...], Any]
    :return: the dataframe with the new column
    :rtype: pl.DataFrame
    """
    return df.with_columns(
        pl.col(col_name).apply(function).alias(new_col_name)
    )


def _map_1ton_function(df: pl.DataFrame,
                       col_name: str, new_col_names: List[str],
                       function: Callable[[...], Any]) -> pl.DataFrame:
    """
    Map values in column to new values using function

    Helper function for map_col

    :param df: The dataframe
    :type df: pl.DataFrame
    :param col_name: the name of the column to map from
    :type col_name: str
    :param new_col_names: names of the new columns to be created as a result of the mapping
    :type new_col_names: List[str]
    :param function: a function to mapping with
    :type function: Callable[[...], Any]
    :return: the dataframe with the new column
    :rtype: pl.DataFrame
    """
    results = df[col_name].map_elements(function)
    new_columns = {new_col_names[i]: [res[i] for res in results] for i in range(len(new_col_names))}

    return df.with_columns([pl.Series(name, values) for name, values in new_columns.items()])


def _map_nto1_function(df: pl.DataFrame,
                       col_names: List[str], new_col_name: str,
                       function: Callable[[...], Any]) -> pl.DataFrame:
    """
    Map values in multiple columns to new values using function

    Helper function for map_col

    :param df: The dataframe
    :type df: pl.DataFrame
    :param col_names: the names of the columns to map from
    :type col_names: List[str]
    :param new_col_name: name of the new column to be created as a result of the mapping
    :type new_col_name: str
    :param function: a function to mapping with
    :type function: Callable[[...], Any]
    :return: the dataframe with the new column
    :rtype: pl.DataFrame
    """
    return df.with_columns(
        pl.struct(col_names).apply(function).alias(new_col_name)
    )


def _map_ntom_function(df: pl.DataFrame,
                       col_names: List[str], new_col_names: List[str],
                       function: Callable[[...], Any]) -> pl.DataFrame:
    """
    Map values in multiple columns to new values using function

    Helper function for map_col

    :param df: The dataframe
    :type df: pl.DataFrame
    :param col_names: the names of the columns to map from
    :type col_names: List[str]
    :param new_col_names: names of the new columns to be created as a result of the mapping
    :type new_col_names: List[str]
    :param function: a function to mapping with
    :type function: Callable[[...], Any]
    :return: the dataframe with the new column
    :rtype: pl.DataFrame
    """
    raise NotImplementedError
