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
from dataclasses import dataclass
from typing import Union

# This file should contain all the constants
FEASIBILITY_TOLERANCE = 1e-6  # Optimization Feasibility tolerance
EARTH_RADIUS = 3959.0  # Earth's radius in Miles
CENSUS_YEAR = 2020
CONVERSION_FACTOR = 5.614583  # convert bbl of oil to mcf of gas
BING_MAPS_BASE_URL = "http://dev.virtualearth.net/REST/V1/Routes/Driving"
# Fixed start coordinates in elevation util to get nearest road point
START_COORDINATES = (40.44, -79.94)


# Set of supported impact metrics along with
# the required data for the analysis.
@dataclass()
class _SupportedContent:
    name: str
    full_name: str
    has_submetrics: bool = False
    is_submetric: bool = False
    parent_metric: Union[str, None] = None
    # required_data: keys in WellDataColumnNames class
    # This will be used to check if the input data has
    # required columns or not.
    required_data: Union[str, list, None] = None


SUPP_IMPACT_METRICS = {
    # Primary metrics
    "ch4_emissions": _SupportedContent(
        name="ch4_emissions",
        full_name="Methane Emissions (Proxies)",
        has_submetrics=True,
    ),
    "dac_impact": _SupportedContent(
        name="dac_impact",
        full_name="Disadvantaged Community Impact",
        has_submetrics=True,
    ),
    "sensitive_receptors": _SupportedContent(
        name="sensitive_receptors",
        full_name="Sensitive Receptors",
        has_submetrics=True,
    ),
    "production_volume": _SupportedContent(
        name="production_volume",
        full_name="Production Volume",
        has_submetrics=True,
    ),
    "well_age": _SupportedContent(
        name="well_age",
        full_name="Well Age",
        required_data="age",
    ),
    "well_count": _SupportedContent(
        name="well_count",
        full_name="Owner Well Count",
        required_data="owner_name",
    ),
    "other_emissions": _SupportedContent(
        name="other_emissions",
        full_name="Other Emissions",
        has_submetrics=True,
    ),
    "well_integrity": _SupportedContent(
        name="well_integrity",
        full_name="Hazards due to Well Integrity",
        required_data="well_integrity",
    ),
    "environment": _SupportedContent(
        name="environment",
        full_name="Environment",
        has_submetrics=True,
    ),
    # Submetrics of ch4_emissions
    "leak": _SupportedContent(
        name="leak",
        full_name="Leak [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="leak",
    ),
    "compliance": _SupportedContent(
        name="compliance",
        full_name="Compliance [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="compliance",
    ),
    "violation": _SupportedContent(
        name="violation",
        full_name="Violation [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="violation",
    ),
    "incident": _SupportedContent(
        name="incident",
        full_name="Incident [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="incident",
    ),
    # Submetrics of dac_impact
    "fed_dac": _SupportedContent(
        name="fed_dac",
        full_name="Justice40 Federal DAC",
        is_submetric=True,
        parent_metric="dac_impact",
        # WellData class computes this information
        required_data=None,
    ),
    "state_dac": _SupportedContent(
        name="state_dac",
        full_name="State DAC",
        is_submetric=True,
        parent_metric="dac_impact",
        required_data="state_dac",
    ),
    # Submetrics of sensitive_recptors
    "hospitals": _SupportedContent(
        name="hospitals",
        full_name="Hospitals",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="hospitals",
    ),
    "schools": _SupportedContent(
        name="schools",
        full_name="Schools",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="schools",
    ),
    "buildings_near": _SupportedContent(
        name="buildings_near",
        full_name="Buildings (Close Range)",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="buildings_near",
    ),
    "buildings_far": _SupportedContent(
        name="buildings_far",
        full_name="Buildings (Distant Range)",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="buildings_far",
    ),
    # Submetrics of environment
    "fed_wetlands_near": _SupportedContent(
        name="fed_wetlands_near",
        full_name="Federal Wetlands (Close Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="fed_wetlands_near",
    ),
    "fed_wetlands_far": _SupportedContent(
        name="fed_wetlands_far",
        full_name="Federal Wetlands (Distant Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="fed_wetlands_far",
    ),
    "state_wetlands_near": _SupportedContent(
        name="state_wetlands_near",
        full_name="State Wetlands (Close Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="state_wetlands_near",
    ),
    "state_wetlands_far": _SupportedContent(
        name="state_wetlands_far",
        full_name="State Wetlands (Distant Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="state_wetlands_far",
    ),
    # Submetrics of production_volume
    "ann_production_volume": _SupportedContent(
        name="ann_production_volume",
        full_name="Annual Production Volume",
        is_submetric=True,
        parent_metric="production_volume",
        required_data=[
            "ann_gas_production",
            "ann_oil_production",
        ],
    ),
    "five_year_production_volume": _SupportedContent(
        name="five_year_production_volume",
        full_name="Five-year Production Volume",
        is_submetric=True,
        parent_metric="production_volume",
        required_data=[
            "five_year_gas_production",
            "five_year_oil_production",
        ],
    ),
    # Submetrics of other_emissions
    "h2s_leak": _SupportedContent(
        name="h2s_leak",
        full_name="H2S Leak [Yes/No]",
        is_submetric=True,
        parent_metric="other_emissions",
        required_data="h2s_leak",
    ),
    "brine_leak": _SupportedContent(
        name="brine_leak",
        full_name="Brine Leak [Yes/No]",
        is_submetric=True,
        parent_metric="other_emissions",
        required_data="brine_leak",
    ),
}

SUPP_EFF_METRICS = {
    "num_wells": _SupportedContent(
        name="num_wells",
        full_name="Number of Wells",
    ),
    "num_unique_owners": _SupportedContent(
        name="num_unique_owners",
        full_name="Number of Unique Owners",
    ),
    "dist_centroid": _SupportedContent(
        name="dist_centroid",
        full_name="Distance to Centroid [miles]",
    ),
    "elevation_delta": _SupportedContent(
        name="elevation_delta",
        full_name="Average Elevation Delta [m]",
    ),
    "age_range": _SupportedContent(
        name="age_range",
        full_name="Age Range [Years]",
    ),
    "depth_range": _SupportedContent(
        name="depth_range",
        full_name="Depth Range [ft]",
    ),
    "record_completeness": _SupportedContent(
        name="record_completeness",
        full_name="Record Completeness",
    ),
}
