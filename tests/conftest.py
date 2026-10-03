import pytest

from vehicle_emissions.data import add_binary_features, load_epa_test_cars, load_fe_guide, load_fe_guide_raw


@pytest.fixture(scope="session")
def fe_raw():
    return load_fe_guide_raw()


@pytest.fixture(scope="session")
def fe():
    return add_binary_features(load_fe_guide())


@pytest.fixture(scope="session")
def epa():
    return load_epa_test_cars()
