#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the DOE's National Emission Reduction
# Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for
# itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit others to do so.
#
#################################################################################

# Standard libs
import logging

# Installed libs
import pytest

# User-defined libs
from primo.data_parser import ImpactMetrics, WellDataColumnNames

LOGGER = logging.getLogger(__name__)


# pylint: disable = missing-function-docstring, no-member
# pylint: disable = f-string-without-interpolation
def test_well_data_column_names():
    wcn = WellDataColumnNames(
        well_id="Well API",
        latitude="Latitude",
        longitude="Longitude",
        age="Age [Years]",
        depth="Depth [ft]",
    )

    cols = ["well_id", "latitude", "longitude", "age", "depth"]

    assert wcn.well_id == "Well API"
    assert wcn.latitude == "Latitude"
    assert wcn.longitude == "Longitude"
    assert wcn.age == "Age [Years]"
    assert wcn.depth == "Depth [ft]"

    # Testing contains and keys methods
    for key in cols:
        assert key in wcn
        assert key in wcn.keys()

    # Testing items method
    for key, val in wcn.items():
        if key not in cols:
            assert val is None

    # Testing iter method
    for key in wcn:
        if key not in cols:
            assert getattr(wcn, key) is None

    # Testing values method
    assert "Well API" in wcn.values()
    assert "Latitude" in wcn.values()
    assert "Longitude" in wcn.values()
    assert "Age [Years]" in wcn.values()
    assert "Depth [ft]" in wcn.values()

    # Test register new column method
    wcn.register_new_columns({"new_col_1": "New Column 1"})
    assert wcn.new_col_1 == "New Column 1"

    with pytest.raises(
        AttributeError,
        match=(
        f"Attribute new_col_1 is already defined. Use a different name."
        )
    ):
        wcn.register_new_columns({"new_col_1": "New Column 1"})


@pytest.fixture(scope="module")
def get_well_data_cols():
    im_mt = ImpactMetrics()
    # Work with fewer metrics for convenience
    im_mt.delete_metric("environment")
    im_mt.delete_metric("other_emissions")
    im_mt.delete_metric("well_age")
    im_mt.delete_metric("well_count")
    im_mt.delete_submetric("violation")
    im_mt.delete_submetric("incident")
    im_mt.delete_submetric("buildings_near")
    im_mt.delete_submetric("buildings_far")

    # Now, the object has five metrics
    # Set weights for all metrics
    im_mt.set_value(
        {
            "ch4_emissions": 20,
            "dac_impact": 20,
            "sensitive_receptors": 20,
            "production_volume": 20,
            "well_integrity": 20,

            # submetrics
            "leak": 50,
            "compliance": 50,
            "fed_dac": 40,
            "state_dac": 60,
            "hospitals": 50,
            "schools": 50,
            "ann_production_volume": 100,
            "five_year_production_volume": 0,
        }
    )

    wcn = WellDataColumnNames(
        well_id="Well API",
        latitude="Latitude",
        longitude="Longitude",
        age="Age [Years]",
        depth="Depth [ft]",
        operator_name="Operator Name",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        state_dac="State DAC Score",
        hospitals="Num Hospitals Nearby",
        schools="Num Schools nearby",
        ann_gas_production="Gas [Mcf/yr]",
        ann_oil_production="Oil [bbl/yr]",
        well_integrity="Well Integrity Status",
    )

    return im_mt, wcn


def test_no_warnings_case(get_well_data_cols):
    """
    Checks if the `check_columns_available` method works as expected
    """
    im_mt, wcn = get_well_data_cols
    assert im_mt.check_validity() is None

    # The object has all the required inputs, so this should
    # not raise any warnings or errors.
    assert wcn.check_columns_available(im_mt) is None
    # This should register `the data_col_name` attribute
    assert im_mt.leak.data_col_name == wcn.leak
    assert im_mt.compliance.data_col_name == wcn.compliance
    assert im_mt.fed_dac.data_col_name is None
    assert im_mt.state_dac.data_col_name == wcn.state_dac
    assert im_mt.hospitals.data_col_name == wcn.hospitals
    assert im_mt.schools.data_col_name == wcn.schools
    assert im_mt.ann_production_volume.data_col_name is None
    assert im_mt.five_year_production_volume.data_col_name is None


def test_unsupported_metric_warn(caplog, get_well_data_cols):

    im_mt, wcn = get_well_data_cols
    # Test unsupported metric warning
    im_mt.register_new_metric(name="metric_1", full_name="Metric One")
    wcn.check_columns_available(im_mt)

    # assert len(record) == 1
    assert (
        "Metric/submetric metric_1/Metric One is not supported. "
        "Users are required to process the data for this metric, and "
        "assign the name of the column to the attribute `data_col_name` "
        "in the metric/submetric metric_1 object."
    ) in caplog.text


def test_missing_col_error(get_well_data_cols):
    im_mt, wcn = get_well_data_cols

    # Suppose well_integrity data is not provided
    wcn.well_integrity = None

    with pytest.raises(
        AttributeError,
        match=(
            "Weight of the metric well_integrity is nonzero, so attribute "
            "well_integrity is an essential input in the "
            "WellDataColumnNames object."
        ),
    ):
        wcn.check_columns_available(im_mt)

    # Repeat the test with a submetric
    wcn.well_integrity = "Well Integrity Status"
    wcn.hospitals = None

    with pytest.raises(
        AttributeError,
        match=(
            "Weight of the metric hospitals is nonzero, so attribute "
            "hospitals is an essential input in the "
            "WellDataColumnNames object."
        ),
    ):
        wcn.check_columns_available(im_mt)

    wcn.hospitals = "Hospitals"
    # Now test the list of columns error message
    wcn.ann_gas_production = None

    with pytest.raises(
        AttributeError,
        match=(
            "Weight of the metric ann_production_volume is nonzero, so attribute "
            "ann_gas_production is an essential input in the WellDataColumnNames object."
        ),
    ):
        wcn.check_columns_available(im_mt)

    wcn.ann_gas_production = "Gas [Mcf/yr]"
    wcn.ann_oil_production = None

    with pytest.raises(
        AttributeError,
        match=(
            "Weight of the metric ann_production_volume is nonzero, so attribute "
            "ann_oil_production is an essential input in the WellDataColumnNames object."
        ),
    ):
        wcn.check_columns_available(im_mt)

    wcn.ann_oil_production = "Oil [bbl/yr]"

    im_mt.set_value(
        {
            "ann_production_volume": 50,
            "five_year_production_volume": 50,
        }
    )
    assert im_mt.check_validity() is None
    wcn.five_year_gas_production = "Five-Year Gas [Mcf]"

    with pytest.raises(
        AttributeError,
        match=(
            "Weight of the metric five_year_production_volume is nonzero, so attribute "
            "five_year_oil_production is an essential input in the WellDataColumnNames object."
        ),
    ):
        wcn.check_columns_available(im_mt)
