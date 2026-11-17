from importlib.metadata import version

import fair_credit_scoring


def test_project_is_installed():
    assert fair_credit_scoring.__file__ is not None
    assert version("fairness-credit-scoring-replication")
