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

"""
This module defines all the input options for objects of WellData class
"""

# Installed libs
from pyomo.common.config import (
    Bool,
    ConfigDict,
    ConfigValue,
    In,
    NonNegativeFloat,
    NonNegativeInt,
)

# User libs
from primo.data_parser.default_data import SUPP_IMPACT_METRICS
from primo.utils.domain_validators import InRange


# This dictionary contains default values to fill with,
# if the data is not provided.
binary_type_metrics = {
    "leak": False,  # Well is not leaking
    "compliance": True,  # Well is compliant
    "violation": False,  # Well is not in violation
    "incident": False,  # Incident is assumed to be False
    "h2s_leak": False,  # H2S Leak is absent
    "brine_leak": False,  # Brine leak is absent
    "buildings_near": False,  # No buildings nearby
    "buildings_far": False,  # No distant buildings
    "fed_wetlands_near": False,  # No federal wetlands nearby
    "fed_wetlands_far": False,  # No distant federal wetlands
    "state_wetlands_near": False,  # No state wetlands nearby
    "state_wetlands_far": False,  # No distant state wetlands
    # Assuming that well integrity is good, if it is not provided
    "well_integrity": False,
}

# Input options for well data class
CONFIG = ConfigDict()
CONFIG.declare(
    "census_year",
    ConfigValue(
        default=2020,
        domain=In(list(range(2020, 2101, 10))),
        doc="Year for collecting census data",
    ),
)
CONFIG.declare(
    "ignore_operator_name",
    ConfigValue(
        default=True,
        domain=Bool,
        doc="Remove well if operator name is not provided",
    ),
)
CONFIG.declare(
    "missing_age",
    ConfigValue(
        default="fill",
        domain=In(["fill", "estimate", "remove"]),
        doc="Method for processing missing age information",
    ),
)
CONFIG.declare(
    "missing_depth",
    ConfigValue(
        default="fill",
        domain=In(["fill", "estimate", "remove"]),
        doc="Method for processing missing depth information",
    ),
)
CONFIG.declare(
    "fill_age",
    ConfigValue(
        default=100,
        # Assuming that no well is older than 350 years
        domain=InRange(0, 350),
        doc="Value to fill with, if the age is missing",
    ),
)
CONFIG.declare(
    "fill_depth",
    ConfigValue(
        default=1000,
        # Assuming that no well is deeper than 20,000 ft
        domain=InRange(0, 20000),
        doc="Value to fill with, if the depth is missing",
    ),
)

for key, val in binary_type_metrics.items():
    CONFIG.declare(
        "fill_" + key,
        ConfigValue(
            default=val,
            domain=Bool,
            doc=(
                f"Value to fill with, if {SUPP_IMPACT_METRICS[key].full_name} "
                f"information is missing"
            ),
        ),
    )

CONFIG.declare(
    "fill_hospitals",
    ConfigValue(
        default=0,  # Assuming no hospitals nearby
        domain=NonNegativeInt,
        doc="Value to fill with, if number of Hospitals nearby is not specified",
    ),
)
CONFIG.declare(
    "fill_schools",
    ConfigValue(
        default=0,  # Assuming no hospitals nearby
        domain=NonNegativeInt,
        doc="Value to fill with, if number of Schools nearby is not specified",
    ),
)
CONFIG.declare(
    "fill_state_dac",
    ConfigValue(
        default=0.0,
        domain=InRange(0, 100),
        doc="Value to fill with, if the state DAC information is not specified",
    ),
)
CONFIG.declare(
    "fill_ann_gas_production",
    ConfigValue(
        default=0.0,  # No gas production, if not specified
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the annual gas production [in Mcf/yr] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "fill_ann_oil_production",
    ConfigValue(
        default=0.0,  # No oil production, if not specified
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the annual oil production [in bbl/yr] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "fill_five_year_gas_production",
    ConfigValue(
        default=0.0,  # No gas production, if not specified
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the five-year gas production [in Mcf] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "fill_five_year_oil_production",
    ConfigValue(
        default=0.0,  # No oil production, if not specified
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the five-year oil production [in bbl] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "fill_life_gas_production",
    ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the lifelong gas production [in Mcf] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "fill_life_oil_production",
    ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the lifelong oil production [in bbl] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "threshold_production_volume",
    ConfigValue(
        domain=NonNegativeFloat,
        doc=(
            "If specified, wells whose lifelong production volume is "
            "above the threshold production volume will be removed from the dataset."
        ),
    ),
)

WELL_DATA_CONFIG = CONFIG
