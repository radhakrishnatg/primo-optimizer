#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

# Standard libs
import logging
from typing import Union

# Installed libs
import pandas as pd
from pyomo.common.config import Bool

# User-defined libs
from primo.data_parser.well_data_column_names import WellDataColumnNames
from primo.data_parser.input_options_well_data import WELL_DATA_CONFIG
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


# pylint: disable = too-many-instance-attributes
# pylint: disable = trailing-whitespace, protected-access
# pylint: disable = logging-fstring-interpolation
class WellData:
    """
    Reads, processes, and anlyzes well data.
    """

    __slots__ = (
        "config",  # ConfigDict containing input options
        "data",  # DataFrame containing well data
        # Private variables
        "_col_names",  # Pointer to WellDataColumnNames object
        "_removed_rows",  # dict containing list of rows removed
        "_num_wells_stats",  # dict containing statistics on num. wells
    )

    def __init__(
        self,
        filename: str,
        column_names: WellDataColumnNames,
        **kwargs,
    ) -> None:
        """
        Reads well data and performs initial high-level data processing.

        Parameters
        ----------
        filename : str
            Name of the file containing the well data. Currently, only
            .xlsx, .xls, and .csv formats are supported.

        column_names : WellDataColumnNames
            A WellDataColumnNames object containing the names of various columns
        """
        # Import columns whose column names have been provided.
        LOGGER.info("Reading the well data from the input file.")
        if filename.split(".")[-1] in ["xlsx", "xls"]:
            self.data = pd.read_excel(
                filename,
                sheet_name=kwargs.pop("sheet", 0),
                usecols=column_names.values(),
                # Store well ids as strings
                dtype={column_names.well_id: str},
            )

        elif filename.split(".")[-1] == "csv":
            self.data = pd.read_csv(
                filename,
                usecols=column_names.values(),
                # Store well ids as strings
                dtype={column_names.well_id: str},
            )

        else:
            raise_exception(
                "Unsupported input file format. Only .xlsx, .xls, and .csv are supported.",
                TypeError,
            )

        # Updating the `index` to keep it consistent with the row numbers in
        # xlsx, xls, or csv files
        self.data.index += 2

        # Read input options
        self.config = WELL_DATA_CONFIG(kwargs)
        self._removed_rows = {}
        self._col_names = column_names
        # Store number of wells in the input data
        num_wells_input = self.data.shape[0]
        self._num_wells_stats = {"input_data": num_wells_input}

        LOGGER.info("Finished reading the well data. Beginning to process it.")

        # Start processing input data
        # self._process_input_data()  # Uncomment in the next PR

        num_wells_processed = self.data.shape[0]
        self._num_wells_stats["initial_processing"] = num_wells_processed
        LOGGER.info("Finished preliminary processing.")

    def __contains__(self, val):
        """Checks if a column is available in the well data"""
        return val in self.data.columns

    def __iter__(self):
        """Iterate over all rows of the well data"""
        return iter(self.data.index)

    @property
    def get_removed_wells(self):
        """Yields the list of wells removed from the data set"""
        row_list = []

        for val in self._removed_rows.values():
            row_list += val

        row_list.sort()

        return row_list

    @property
    def get_removed_wells_with_reason(self):
        """
        Yields the list of wells removed from the data set.
        Keys contain the reason, and the values contain the list of rows removed.
        """
        return self._removed_rows

    @property
    def get_flag_columns(self):
        """Returns all the columns containing"""
        return [col for col in self.data.columns if "_flag" in col]

    @property
    def get_priority_score_columns(self):
        """Returns all the columns containing metric scores"""
        return [col for col in self.data.columns if " Score " in col]

    def has_incomplete_data(self, col_name: str):
        """
        Checks if a column contains empty cells.

        Parameters
        ----------
        col_name : str
            Name of the column

        Returns
        -------
        flag : bool
            True, if the column has empty cells; False, otherwise

        empty_cells : list
            A list of rows containing empty cells
        """
        flag = False

        empty_cells = self.data[self.data[col_name].isna()].index
        if len(empty_cells) > 0:
            LOGGER.warning(f"Found empty cells in column {col_name}")
            flag = True

        return flag, list(empty_cells)

    def drop_incomplete_data(self, col_name: str, dict_key: str):
        """
        Removes rows(wells) if the cell in a specific column is empty.

        Parameters
        ----------
        col_name : str
            Name of the column

        dict_key : str
            A key/identifier to store the reason for removal

        Returns
        -------
        None
        """
        # NOTE: Can avoid reassignment by passing the argument inplace=True.
        # But this is not recommended.
        new_well_data = self.data.dropna(subset=col_name)

        if new_well_data.shape[0] == self.data.shape[0]:
            # There is no incomplete data in col_list, so return
            return

        LOGGER.warning(
            f"Removed a few wells because {col_name} information is not available for them."
        )
        removed_rows = [i for i in self.data.index if i not in new_well_data.index]
        if dict_key in self._removed_rows:
            self._removed_rows[dict_key].append(removed_rows)
        else:
            self._removed_rows[dict_key] = removed_rows

        # Replace the pointer to the new DataFrame
        self.data = new_well_data

        return

    def fill_incomplete_data(
        self,
        col_name: str,
        value: float,
        flag_col_name: Union[None, str] = None,
    ):
        """
        Fill empty cells in a column with a constant value

        Parameters
        ----------
        col_name : str
            Name of the column

        value : float
            Empty cells in the column will be filled with `value`

        flag_col_name : str
            If specified, creates a new column called `flag_col_name`,
            and flags those wells with empty cells in column `col_name`.
        """

        if flag_col_name is not None:
            _, empty_cells = self.has_incomplete_data(col_name)
            self.flag_wells(rows=empty_cells, col_name=flag_col_name)

        LOGGER.warning(
            f"Filling any empty cells in column {col_name} with value {value}."
        )

        self.data[col_name] = self.data[col_name].fillna(value)

    def replace_data(self, col_name: str, from_to_map: dict):
        """
        Method to replace specific data with new values.

        Parameters
        ----------
        col_name : str
            Name of the column

        from_to_map : dict
            {key = current data, and value = new data}
        """
        with pd.option_context("future.no_silent_downcasting", True):
            self.data[col_name] = (
                self.data[col_name].replace(from_to_map).infer_objects()
            )

    def convert_data_to_binary(self, col_name: str):
        """
        Converts 1/True/"Yes"/"Y" -> 1 and 0/False/"No"/"N" -> 0.
        The strings "Yes", "Y", "No", and "N" are case-insensitive i.e.,
        "yes", "y", "no", "n" are also acceptable.

        Parameters
        ----------
        col_name : str
            Name of the column
        """
        # First convert the data to True or False
        try:
            self.data[col_name] = self.data[col_name].apply(Bool)
        except ValueError as excp:
            raise ValueError(
                f"Column {col_name} is expected to contain boolean-type "
                f"data. Received a non-boolean value for some/all rows."
            ) from excp

        # Now convert the data to binary
        self.data[col_name] = self.data[col_name].apply(int)

    def check_data_in_range(self, col_name: str, lb: float, ub: float):
        """
        Utility to check if all the values in a column are within
        a valid interval.

        Parameters
        ----------
        col_name : str
            Name of the column
        lb : float
            Lower bound for the value
        ub : float
            Upper bound for the value

        Returns
        -------
        invalid_data : list
            List of rows in which the value lies outside the interval

        Raises
        ------
        ValueError
            If one or more values in the column lie outside the valid range
        """
        # Check the validity of latitude and longitude
        valid_data = self.data[col_name].apply(lambda val: lb <= val <= ub)
        invalid_data = list(self.data[~valid_data].index)

        if len(invalid_data) > 0:
            msg = (
                f"Values in column {col_name} are expected to be in the interval "
                f"{[lb, ub]}. However, the value in one or more rows "
                f"lies outside the interval."
            )
            raise_exception(msg, ValueError)

    def is_data_numeric(self, col_name: str):
        """
        Checks if all the cells in a column are numeric.

        Parameters
        ----------
        col_name : str
            Name of the column
        """
        # There are two ways to do this:
        # The current approach is a safe option since it also works for mixed
        # data types, but it is a bit messy.
        # Alternatively, one can use pd.api.types.is_numeric_dtype() method.
        # This is quite compact, but this has risks. In the future, pandas will not
        # allow modification of cells in a column to a different dtype. Then, it will
        # be safe to use the pd.api.types.is_numeric_dtype() method.
        flag = True
        non_numeric_rows = []
        for i in self:
            # Record the row number if it is an empty cell
            if pd.isnull(self.data.loc[i, col_name]):
                non_numeric_rows.append(i)

            # Record the row number if it does not contain a numeric value
            elif not pd.api.types.is_number(self.data.loc[i, col_name]):
                non_numeric_rows.append(i)

        if len(non_numeric_rows) > 0:
            flag = False

        return flag, non_numeric_rows

    def flag_wells(self, rows: list, col_name: str):
        """
        Utility to flag a specific set of wells. Useful to record
        wells for which a specific data/metric is estimated.

        wells: list
            List of rows(wells) in the DataFrame

        col_name : str
            Name of the new column that contains the flag information
        """
        if len(rows) == 0:
            # Nothing to flag, so return.
            return

        self.data[col_name] = 0
        for row in rows:
            self.data.loc[row, col_name] = 1

    def add_new_columns(self):
        """
        Adds new columns to the DataFrame. Read -> Remove deleted rows -> Join columns
        """
        # NOTE: Can avoid using this method by importing all the required columns
        # at the beginning. Just have to register unsupported columns in the
        # WellDataColumnNames object using obj.register_new_columns() method before
        # creating the WellData object.
        # NOTE: Not supporting it currently, since it is not crucial. This can be
        # supported in the future.
        msg = (
            "This method is not supported currently. Ensure that all the required columns "
            "are specified in the WellDataColumnNames object. To read unsupported data, "
            "register the corresponding column using the `register_new_columns` method."
        )
        raise_exception(msg, NotImplementedError)
