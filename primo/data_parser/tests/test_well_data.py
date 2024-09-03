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
import os

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.data_parser.well_data_column_names import WellDataColumnNames

LOGGER = logging.getLogger(__name__)


INCOMPLETE_ROWS = {
    "API Well Number": [2, 3, 4, 5, 6],
    "x": [7, 8, 9, 10, 11],
    "y": [12, 13, 14, 15, 16],
    "Operator": [17, 18, 19, 20, 21],
    "Unknown Operator": [22, 23, 24, 25, 26],
    "Age": [27, 28, 29, 30, 31],
    "Depth": [32, 33, 34, 35, 36],
    "Leak": [37, 38, 39, 40, 41],
    "Violation": [42, 43, 44, 45, 46],
    "Incident": [47, 48, 49, 50, 51],
    "Compliance": [52, 53, 54, 55, 56],
    "Oil": [57, 58, 59, 60, 61],
    "Gas": [62, 63, 64, 65, 66],
    "Hospitals": [67, 68, 69, 70, 71],
    "Schools": [72, 73, 74, 75, 76],
    "Life Gas Fill": [77, 78, 79, 80, 81],
    "Life Oil Fill": [82, 83, 84, 85, 86],
    "Life Gas Remove": [87, 88, 89, 90, 91],
    "Life Oil Remove": [92, 93, 94, 95, 96],
}


# pylint: disable = missing-function-docstring, protected-access
# pylint: disable = unused-variable
@pytest.fixture(name="get_column_names", scope="function")
def get_column_names_fixture():
    """Returns well data from a csv file"""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        violation="Violation [Yes/No]",
        incident="Incident [Yes/No]",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        hospitals="Hospitals",
        schools="Schools",
        life_gas_production="Lifelong Gas [Mcf]",
        life_oil_production="Lifelong Oil [bbl]",
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data.csv",
    )

    return filename, col_names


def test_well_data(caplog, get_column_names):
    """Returns well data from a csv file"""

    filename, col_names = get_column_names
    wd = WellData(
        filename=filename,
        column_names=col_names,
        fill_age=99,
        fill_depth=999,
        fill_life_gas_production=1.5,
        fill_life_oil_production=1.5,
        threshold_gas_production=2,
        threshold_oil_production=2,
    )

    # Well ID checks
    assert (
        f"Removed a few wells because {col_names.well_id} "
        f"information is not available for them."
    ) in caplog.text
    assert wd._removed_rows["well_id"] == INCOMPLETE_ROWS["API Well Number"]

    # Latitue checks
    assert (
        f"Removed a few wells because {col_names.latitude} "
        f"information is not available for them."
    ) in caplog.text
    assert wd._removed_rows["latitude"] == INCOMPLETE_ROWS["x"]

    # Longitude checks
    assert (
        f"Removed a few wells because {col_names.longitude} "
        f"information is not available for them."
    ) in caplog.text
    assert wd._removed_rows["longitude"] == INCOMPLETE_ROWS["y"]

    # Operator name checks
    assert (
        f"Removed a few wells because {col_names.operator_name} "
        f"information is not available for them."
    ) in caplog.text
    assert wd._removed_rows["operator_name"] == INCOMPLETE_ROWS["Operator"]

    assert (
        "Owner name for some wells is listed as unknown."
        "Treating these wells as if the owner name is not provided, "
        "so removing them from the dataset."
    ) in caplog.text
    assert wd._removed_rows["unknown_owner"] == INCOMPLETE_ROWS["Unknown Operator"]

    # Age checks
    assert (
        "Assigning the age of the well as 99 "
        "years, if it is missing. To change this number, pass fill_age "
        "argument while instantiating the WellData object."
    ) in caplog.text

    # Ensure that wells are not deleted because of missing age
    assert "age" not in wd._removed_rows
    for row in wd:
        if row in INCOMPLETE_ROWS["Age"]:
            assert wd.data.loc[row, "age_flag"] == 1
            assert wd.data.loc[row, col_names.age] == 99
        else:
            assert wd.data.loc[row, "age_flag"] == 0

    # Depth checks
    assert (
        "Assigning the depth of the well as 999 "
        "ft, if it is missing. To change this number, pass fill_depth "
        "argument while instantiating the WellData object."
    ) in caplog.text

    # Ensure that wells are not deleted because of missing depth
    assert "depth" not in wd._removed_rows
    for row in wd:
        if row in INCOMPLETE_ROWS["Depth"]:
            assert wd.data.loc[row, "depth_flag"] == 1
            assert wd.data.loc[row, col_names.depth] == 999
        else:
            assert wd.data.loc[row, "depth_flag"] == 0

    # Filter volume checks
    assert (
        wd.data.loc[INCOMPLETE_ROWS["Life Gas Fill"], col_names.life_gas_production]
        == 1.5
    ).all()  # Check if the missing data is filled correctly
    assert (
        wd.data.loc[INCOMPLETE_ROWS["Life Oil Fill"], col_names.life_oil_production]
        == 1.5
    ).all()  # Check if the missing data is filled correctly
    assert wd._removed_rows["production_volume"] == (
        INCOMPLETE_ROWS["Life Gas Remove"] + INCOMPLETE_ROWS["Life Oil Remove"]
    )
    assert (
        "Some wells have been removed based on the lifelong production volume."
    ) in caplog.text

    # Because of missing data, categorization based on depth is not performed
    assert wd._well_types["shallow"] is None
    assert wd._well_types["deep"] is None

    # Data needed to classify oil and gas wells is available
    assert isinstance(wd._well_types["oil"], set)
    assert isinstance(wd._well_types["gas"], set)

    # Warning on number of wells removed
    assert (
        "Preliminary processing removed 35 "
        "wells (11.67% wells in the input data set) because of missing "
        "information. To include these wells in the analysis, please provide the missing "
        "information. List of wells removed can be queried using the get_removed_wells "
        "and get_removed_wells_with_reason properties."
    ) in caplog.text


def test_age_depth_remove(caplog, get_column_names):
    """Tests the remove method for missing age and depth"""
    filename, col_names = get_column_names

    wd = WellData(
        filename=filename,
        column_names=col_names,
        missing_age="remove",
        missing_depth="remove",
    )

    assert (
        f"Removed a few wells because {col_names.age} "
        f"information is not available for them."
    ) in caplog.text
    assert wd._removed_rows["age"] == INCOMPLETE_ROWS["Age"]

    assert (
        f"Removed a few wells because {col_names.depth} "
        f"information is not available for them."
    ) in caplog.text
    assert wd._removed_rows["depth"] == INCOMPLETE_ROWS["Depth"]


def test_age_depth_estimation(caplog, get_column_names):
    """Tests the estimate method for missing age and depth"""
    filename, col_names = get_column_names

    with pytest.raises(
        NotImplementedError,
        match="Age estimation feature is not supported currently.",
    ):
        wd = WellData(
            filename=filename,
            column_names=col_names,
            missing_age="estimate",
        )

    assert (
        "Estimating the age of a well if it is missing. The estimation "
        "approach assumes that all the wells in the input file are "
        "arranged in chronological order of their commission. DO NOT USE "
        "this utility if that is not the case!."
    ) in caplog.text

    with pytest.raises(
        NotImplementedError,
        match="Depth estimation feature is not supported currently.",
    ):
        wd = WellData(
            filename=filename,
            column_names=col_names,
            missing_depth="estimate",
        )

    assert (
        "Estimating the depth of a well, if it is missing, from its "
        "nearest neighbors."
    ) in caplog.text
