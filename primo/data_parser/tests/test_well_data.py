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
from primo.data_parser import ImpactMetrics, WellData, WellDataColumnNames

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
# pylint: disable = unused-variable, no-member
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
        threshold_depth=2000,
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

    # Threshold depth is specified, so the processing step categorizes wells as
    # shallow and deep
    assert isinstance(wd._well_types["shallow"], set)
    assert isinstance(wd._well_types["deep"], set)

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


def test_no_warnings(caplog, get_column_names):
    """
    This function:
    1. Checks if no warnings are thrown if there are no issues with data.
    2. Passing the data as DataFrame instead of a file.
    3. No categorization is performed if the requied data is not
        available.
    4. A warning is printed if the required input for filter production
        volume function is not available
    5. Raises an error if the Well API is not unique
    """
    filename, col_names = get_column_names

    # Doing this to avoid oil-gas classification
    col_names.ann_gas_production = None
    col_names.ann_oil_production = None
    col_names.life_gas_production = None
    col_names.life_oil_production = None

    wd_df = pd.read_csv(filename, usecols=col_names.values())
    # Removing 100 wells with incomplete data
    wd_df = wd_df.drop(index=list(range(100)))
    assert wd_df.shape[0] == 200

    wd = WellData(
        filename=wd_df,
        column_names=col_names,
        fill_age=99,
        fill_depth=999,
        fill_life_gas_production=1.5,
        fill_life_oil_production=1.5,
        threshold_gas_production=2,
        threshold_oil_production=2,
    )

    # Check no warning messages related to empty are printed
    assert "Found empty cells in column" not in caplog.text
    assert (
        "information. To include these wells in the analysis, please provide the missing "
        "information. List of wells removed can be queried using the get_removed_wells "
        "and get_removed_wells_with_reason properties."
    ) not in caplog.text

    # Filter production volume is active, but the required data is not available
    # This should print a warning.
    assert (
        "Unable to filter wells based on production volume because both lifelong gas "
        "production [in Mcf] and lifelong oil production [in bbl] are missing in the "
        "input data. Please provide at least one of the columns via attributes "
        "life_gas_production and life_oil_production in the WellDataColumnNames object."
    ) in caplog.text

    # Categorization is not performed because data is not available
    assert wd._well_types["oil"] is None
    assert wd._well_types["gas"] is None
    assert wd._well_types["shallow"] is None
    assert wd._well_types["deep"] is None

    assert wd.get_gas_oil_wells is None
    assert (
        "Insufficient information for well categorization. Either specify "
        "well_type in the input data, or provide both ann_gas_production "
        "ann_oil_production"
    ) in caplog.text

    assert wd.get_shallow_deep_wells is None
    assert (
        "Insufficient information for well categorization. Either specify "
        "well_type_by_depth in the input data, or specify threshold_depth "
        "while instantiating the WellData object."
    ) in caplog.text

    assert wd.full_partitioned_data is None
    assert "Insufficient information for well categorization." in caplog.text

    # Raises an error if the Well API is not unique
    wd_df.loc[103, col_names.well_id] = wd_df.loc[102, col_names.well_id]
    with pytest.raises(
        ValueError,
        match="Well ID must be unique. Found multiple wells with the same Well ID.",
    ):
        wd = WellData(filename=wd_df, column_names=col_names)


def test_age_depth_remove(caplog, get_column_names):
    """Tests the remove method for missing age and depth"""
    filename, col_names = get_column_names

    wd = WellData(
        filename=filename,
        column_names=col_names,
        missing_age="remove",
        missing_depth="remove",
    )

    assert f"Found empty cells in column {col_names.age}" in caplog.text
    assert (
        f"Removed a few wells because {col_names.age} "
        f"information is not available for them."
    ) in caplog.text
    assert wd._removed_rows["age"] == INCOMPLETE_ROWS["Age"]

    assert f"Found empty cells in column {col_names.depth}" in caplog.text
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


def test_well_partitioning(tmp_path, get_column_names):
    filename, col_names = get_column_names
    # Doing this to avoid oil-gas classification
    col_names.ann_gas_production = None
    col_names.ann_oil_production = None
    col_names.life_gas_production = None
    col_names.life_oil_production = None

    # Add well types to the input data
    wd_df = pd.read_csv(filename, usecols=col_names.values())
    # Removing 100 wells with incomplete data
    wd_df = wd_df.drop(index=list(range(100))).reset_index()
    assert wd_df.shape[0] == 200

    # Initialize columns
    col_names.well_type = "Well Type"
    col_names.well_type_by_depth = "Well Type by depth"
    wd_df[col_names.well_type] = "OIL"
    wd_df[col_names.well_type_by_depth] = "Deep"

    for i in range(100):
        wd_df.loc[i, col_names.well_type] = "Gas"

    for i in range(50):
        wd_df.loc[i, col_names.well_type_by_depth] = "Shallow"

    for i in range(100, 150):
        wd_df.loc[i, col_names.well_type_by_depth] = "Shallow"

    # Construct the WellData object
    wd = WellData(filename=wd_df, column_names=col_names)

    assert wd._well_types["oil"] == set(range(100, 200))
    assert wd._well_types["gas"] == set(range(100))
    assert wd._well_types["shallow"] == set(range(50)).union(set(range(100, 150)))
    assert wd._well_types["deep"] == set(range(50, 100)).union(set(range(150, 200)))

    gas_oil_wells = wd.get_gas_oil_wells
    assert isinstance(gas_oil_wells["oil"], WellData)
    assert gas_oil_wells["oil"].data.shape[0] == 100
    assert isinstance(gas_oil_wells["gas"], WellData)
    assert gas_oil_wells["gas"].data.shape[0] == 100
    assert len(gas_oil_wells) == 2

    shallow_deep_wells = wd.get_shallow_deep_wells
    assert isinstance(shallow_deep_wells["shallow"], WellData)
    assert shallow_deep_wells["shallow"].data.shape[0] == 100
    assert isinstance(shallow_deep_wells["deep"], WellData)
    assert shallow_deep_wells["deep"].data.shape[0] == 100
    assert len(shallow_deep_wells) == 2

    full_partition_wells = wd.full_partitioned_data
    assert len(full_partition_wells) == 4
    assert isinstance(full_partition_wells["oil_deep"], WellData)
    assert full_partition_wells["oil_deep"].data.shape[0] == 50
    assert list(full_partition_wells["oil_deep"].data.index) == list(range(150, 200))

    assert isinstance(full_partition_wells["oil_shallow"], WellData)
    assert full_partition_wells["oil_shallow"].data.shape[0] == 50
    assert list(full_partition_wells["oil_shallow"].data.index) == list(range(100, 150))

    assert isinstance(full_partition_wells["gas_deep"], WellData)
    assert full_partition_wells["gas_deep"].data.shape[0] == 50
    assert list(full_partition_wells["gas_deep"].data.index) == list(range(50, 100))

    assert isinstance(full_partition_wells["gas_shallow"], WellData)
    assert full_partition_wells["gas_shallow"].data.shape[0] == 50
    assert list(full_partition_wells["gas_shallow"].data.index) == list(range(50))

    # Write the data files to the temp directory
    oil_deep_file = str(tmp_path / "oil_deep.xlsx")
    full_partition_wells["oil_deep"].save_to_file(oil_deep_file)
    gas_shallow_file = str(tmp_path / "gas_shallow.csv")
    full_partition_wells["gas_shallow"].save_to_file(gas_shallow_file)

    assert os.path.exists(oil_deep_file)
    assert os.path.exists(gas_shallow_file)

    gas_deep_file = str(tmp_path / "gas_deep.foo")
    with pytest.raises(ValueError, match="Format .foo is not supported."):
        full_partition_wells["gas_deep"].save_to_file(gas_deep_file)

    # Check unrecognized well-type erros
    wd_df.loc[0, col_names.well_type] = "Foo"
    with pytest.raises(
        ValueError,
        match=("Well-type must be either oil or gas. Received " "Foo in row 0."),
    ):
        wd = WellData(filename=wd_df, column_names=col_names)

    wd_df.loc[0, col_names.well_type] = "gas"
    wd_df.loc[0, col_names.well_type_by_depth] = "Foo"
    with pytest.raises(
        ValueError,
        match=(
            "Well-type by depth must be either shallow or deep. Received "
            "Foo in row 0."
        ),
    ):
        wd = WellData(filename=wd_df, column_names=col_names)


def test_compute_priority_scores(caplog, get_column_names):
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

    im_metrics = ImpactMetrics()
    im_metrics.set_weight(
        primary_metrics={
            "ch4_emissions": 30,
            "dac_impact": 5,
            "sensitive_receptors": 20,
            "ann_production_volume": 20,
            "well_age": 15,
            "well_count": 10,
        },
        submetrics={
            "ch4_emissions": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "dac_impact": {
                "fed_dac": 100,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )

    wd.compute_priority_scores(impact_metrics=im_metrics)

    # Check warning messages
    assert f"Found empty cells in column {col_names.leak}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.leak} with value False."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.compliance}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.compliance} with value True."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.violation}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.violation} with value False."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.incident}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.incident} with value False."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.hospitals}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.hospitals} with value 0."
        in caplog.text
    )

    assert f"Found empty cells in column {col_names.schools}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.schools} with value 0."
        in caplog.text
    )

    # Annual gas and oil production data has been filled in the preliminary step
    assert f"Found empty cells in column {col_names.ann_gas_production}" in caplog.text
    assert f"Found empty cells in column {col_names.ann_oil_production}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.ann_gas_production} "
        f"with value 0.0"
    ) in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.ann_oil_production} "
        f"with value 0.0"
    ) in caplog.text

    # All cells in the leak column must be zero
    assert (wd.data[im_metrics.leak.score_col_name] == 0).all()
