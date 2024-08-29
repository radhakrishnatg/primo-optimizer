"""
These checks are used by stagedfright to determine whether a staged file is cleared for commit 
or not. The check suite should be optimized and optimized based on the features of the specific 
sensitive data formats and features being handled.
For more information, see the README.md file in this directory.
"""

import ast
import numbers
from pathlib import Path
from typing import List, Dict

import pytest
from stagedfright import StagedFile, AllowFile, PyContent


def test_allowfile_matches_if_present(staged: StagedFile, allowfile: AllowFile):
    assert (
        staged.fingerprint == allowfile.fingerprint
    ), "An allowfile must contain a matching fingerprint for a staged file to be cleared for commit"


MAP_PATH_EXPECTED_HARDCODED_DATA_COUNT = {
    "primo/data_parser/data_model.py": 8,
    "docs/conf.py": 4,
    "primo/opt_model/base_model.py": 15,
    "primo/opt_model/opt_model.py": 15,
    "primo/opt_model/tests/test_base_model.py": 13,
    "primo/opt_model/tests/test_opt_model.py": 19,
    "primo/utils/__init__.py": 6,
    "primo/utils/age_depth_estimation.py": 18,
    "primo/utils/census_utils.py": 12,
    "primo/utils/config_utils.py": 15,
    "primo/utils/demo_utils.py": 30,
    "primo/utils/elevation_utils.py": 14,
    "primo/utils/estimation_utils.py": 14,
    "primo/utils/geo_utils.py": 20,
    "primo/utils/kpi_utils.py": 7,
    "primo/utils/map_utils.py": 40,
    "primo/utils/opt_utils.py": 5,
    "primo/utils/override_utils.py": 5,
    "primo/utils/proximity_to_sensitive_receptors.py": 4,
    "primo/utils/proximity_utils.py": 5,
    "primo/utils/setup_arg_parser.py": 9,
    "primo/utils/setup_logger.py": 3,
    "primo/utils/tests/test_config_utils.py": 67,
    "primo/utils/tests/test_demo_utils.py": 92,
    "primo/utils/tests/test_elevation_utils.py": 35,
    "primo/utils/tests/test_estimation_utils.py": 181,
    "primo/utils/tests/test_geo_utils.py": 21,
    "primo/utils/tests/test_kpi_utils.py": 118,
    "primo/utils/tests/test_map_utils.py": 124,
    "primo/utils/tests/test_opt_utils.py": 55,
    "primo/utils/tests/test_proximity_utils.py": 26,
    "primo/utils/welldata_clustering_functions.py": 14,
}

# this is meta, two levels deep:
# - this file is also included checked by stagedfright,
#   so the hardcoded data used to define this mapping must be counted,
# - the extra number accounts for constants defined below,
#   plus 1 for the fact that this number itself is defined using a constant
MAP_PATH_EXPECTED_HARDCODED_DATA_COUNT[".stagedfright/checks.py"] = (
    len(MAP_PATH_EXPECTED_HARDCODED_DATA_COUNT) + 3
)


@pytest.mark.usefixtures("skip_if_matching_allowfile")
class TestIsClearedForCommit:
    def test_has_py_path_suffix(self, staged: StagedFile):
        assert staged.suffix in {
            ".py"
        }, "Only files with a 'py' extension may be cleared for commit"

    def test_is_text_file(self, staged: StagedFile):
        assert (
            staged.text_content is not None
        ), "Only source (text) files may be cleared for commit"

    def test_has_meaningful_python_code(self, staged_pycontent: PyContent):
        assert len(staged_pycontent.ast_nodes.essential) >= 2

    @pytest.fixture
    def hardcoded_data_definitions(self, staged_pycontent: PyContent) -> List[ast.AST]:
        return [
            n
            for n in staged_pycontent.ast_nodes.literal
            if isinstance(n.value, numbers.Number) and n.value != 0
        ]

    @pytest.fixture
    def hardcoded_data_count(self, hardcoded_data_definitions: List[ast.AST]) -> int:
        return len(hardcoded_data_definitions)

    @pytest.fixture
    def _mapping_with_normalized_paths(self) -> Dict[str, int]:
        # paths must be normalized (here, by using pathlib.Path objects instead of str)
        # so that the same key in the mapping can be matched on both UNIX and Windows
        return {Path(k): v for k, v in MAP_PATH_EXPECTED_HARDCODED_DATA_COUNT.items()}

    @pytest.fixture
    def expected_hardcoded_data_count(
        self,
        staged: StagedFile,
        _mapping_with_normalized_paths: Dict[str, int],
    ) -> int:
        key = Path(staged)
        return int(_mapping_with_normalized_paths.get(key, 0))

    def test_py_module_has_no_unexpected_hardcoded_data(
        self,
        hardcoded_data_count: int,
        expected_hardcoded_data_count: int,
        max_added_count=2,
    ):
        assert hardcoded_data_count <= (expected_hardcoded_data_count + max_added_count)
