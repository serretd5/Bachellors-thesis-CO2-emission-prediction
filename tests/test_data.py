"""Tests de ingesta y transformación de la Fuel Economy Guide."""

import pandas as pd

from vehicle_emissions.data import FE_SENTINEL_THRESHOLD, clean_fe_guide, model_family, vehicle_signature


def test_raw_shape(fe_raw):
    assert fe_raw.shape == (964, 162)


def test_clean_keeps_every_vehicle(fe_raw, fe):
    assert len(fe) == len(fe_raw.dropna(how="all"))


def test_clean_is_deterministic(fe_raw):
    pd.testing.assert_frame_equal(clean_fe_guide(fe_raw), clean_fe_guide(fe_raw))


def test_targets_preserved_row_by_row(fe_raw, fe):
    """La limpieza no reordena filas: el objetivo coincide con el original posición a posición."""
    original = fe_raw["Comb FE (Guide) - Conventional Fuel"].to_numpy()
    assert (fe["Comb FE (Guide)"].astype(int).to_numpy() == original).all()


def test_guzzler_encoding_matches_source(fe_raw, fe):
    assert fe["Guzzler?"].sum() == (fe_raw["Guzzler?"].astype(str).str.strip() == "G").sum() == 55


def test_binary_features_semantics(fe):
    manual = fe["Trans Desc"] == "Manual"
    assert (fe.loc[manual, "Automatic"] == 0).all() and (fe.loc[~manual, "Automatic"] == 1).all()
    diesel = fe["Fuel Usage"].str.contains("Diesel")
    assert (fe.loc[diesel, "Gasoline"] == 0).all() and diesel.sum() == 20
    assert set(fe.loc[fe["AWD/4WD"] == 0, "Drive Desc"]) == {"2-Wheel Drive, Front", "2-Wheel Drive, Rear"}


def test_hybrid_flag_excludes_mild_hybrids(fe):
    assert not ((fe["Hybrid"] == 1) & (fe["Mild Hybrid"] == 1)).any()
    assert fe["Hybrid"].sum() == 64


def test_model_family_merges_drive_variants():
    df = pd.DataFrame({"Division": ["BMW", "BMW", "BMW"],
                       "Carline": ["X3 sDrive30i", "X3 xDrive30i AWD", "X5 xDrive40i"]})
    fam = model_family(df)
    assert fam.nunique() == 3  # el sufijo AWD se elimina, pero sDrive/xDrive siguen siendo modelos distintos
    df2 = pd.DataFrame({"Division": ["Toyota", "Toyota"], "Carline": ["SEQUOIA 2WD", "SEQUOIA 4WD"]})
    assert model_family(df2).nunique() == 1


def test_epa_sentinels_present_in_source(epa):
    """El extracto contiene marcadores no físicos que el pipeline debe filtrar."""
    assert (epa["RND_ADJ_FE"] >= FE_SENTINEL_THRESHOLD).sum() > 0


def test_vehicle_signature_groups_cycles(epa):
    sig = vehicle_signature(epa)
    assert sig.nunique() < len(epa)  # cada vehículo aparece en varios ciclos
