"""Regression checks against frozen pre-Numba growth and solver outputs."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from probabilistic_piping import PipingSettings, pipe_growth
from probabilistic_piping.probabilistic_timedep import _z_pipelengte
from test_timedep import (
    DS_10,
    MC_2000,
    _hydrograph,
    _input_df,
    _model_df,
    _timedep,
)

EXPECTED = Path(__file__).parent / "data" / "timedep_expected"


@pytest.fixture(scope="session")
def growth_expected():
    with np.load(EXPECTED / "growth.npz", allow_pickle=False) as expected:
        yield expected


@pytest.fixture(scope="session")
def solver_expected():
    return json.loads((EXPECTED / "solvers.json").read_text(encoding="utf-8"))[
        "results"
    ]


def _settings(n=400, model="Pol2022", detect=False, mixed=False):
    """Synthetic samples with both erosion outcomes and random intervention times."""
    rng = np.random.default_rng(14)
    constants = {
        "L": 50.0,
        "D": 20.0,
        "D_cover": 2.0,
        "h_exit": 0.0,
        "r_exit": 0.8,
        "gamma_sat": 20.0,
        "gamma_water": 9.81,
        "k": 2e-5,
        "d_70": 2.5e-4,
        "d_70m": 2.08e-4,
        "eta": 0.25,
        "theta": 37.0,
        "g": 9.81,
        "v": 1.33e-6,
        "gamma_sp": 16.5,
        "m_p": 1.0,
        "m_u": 1.0,
        "i_ch": 0.6,
        "r_c": 0.3,
        "krit_verval_factor": 1.0,
        "n_aq": 0.4,
        "dldt_kezdi_mv": 2.0,
        "phi_gem": 3.0,
        "h_gem": 2.0,
        "L_vl": 100.0,
        "L_al": 50.0,
        "D_vl": 2.0,
        "D_al": 2.0,
        "k_vl": 1e-6,
        "k_al": 1e-6,
        "L_dijkbasis": 40.0,
    }
    columns = {name: np.full(n, value) for name, value in constants.items()}
    for name in ["L", "D_cover", "k", "m_p", "m_u", "i_ch"]:
        columns[name] *= rng.uniform(0.8, 1.2, n)
    columns.update(
        dldt_Ce=rng.lognormal(np.log(0.15), 0.7, n),
        dldt_kezdi_C=rng.lognormal(np.log(4.0), 0.7, n),
        dldt_constant=rng.lognormal(np.log(6e-4), 0.7, n),
        l_ini=rng.choice([0.0, 0.0, 0.3, 0.6], n) if mixed else np.zeros(n),
        intervention_hours=rng.uniform(0, 40, n),
        dldt_modeltype=model,
        detect_true=detect,
    )
    if n > 1:
        columns["r_exit"][::5] = 0.02
        columns["l_ini"][::5] = 0.0
    settings = PipingSettings()
    settings.set_params_fromdict(columns)
    settings.set_waterstand_tijdreeks(*_hydrograph(14.0))
    return settings


def _assert_growth_equal(actual, expected, case):
    prefix = f"{case}__"
    reference_fields = {
        key.removeprefix(prefix) for key in expected.files if key.startswith(prefix)
    }
    assert set(actual._fields) == reference_fields
    for name, value in actual._asdict().items():
        reference = expected[f"{prefix}{name}"]
        if name == "dldt_avg":
            np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)
        else:
            np.testing.assert_array_equal(value, reference)


def _assert_solver_equal(actual, expected):
    actual = asdict(actual)
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        value = actual[name]
        if isinstance(reference, (float, dict)):
            # Iteratieve solvers versterken afrondingsverschillen tussen builds.
            # Status, meldingen en aantallen blijven exact gelijk.
            assert value == pytest.approx(reference, rel=1e-9, abs=1e-12), name
        else:
            assert value == reference, name


@pytest.mark.parametrize("model", ["constant", "Pol2019", "Pol2022"])
@pytest.mark.parametrize("detect", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_growth_matches_frozen_outputs(growth_expected, model, detect, mixed):
    settings = _settings(model=model, detect=detect, mixed=mixed)
    case = f"batch_{model}_{int(detect)}_{int(mixed)}"
    _assert_growth_equal(pipe_growth(settings), growth_expected, case)
    if not detect:
        lengths = growth_expected[f"{case}__l_end"]
        assert (lengths > settings.L).any(), "Include failing samples"
        assert (lengths < settings.L).any(), "Include safe samples"


@pytest.mark.parametrize("model", ["constant", "Pol2019", "Pol2022"])
@pytest.mark.parametrize("detect", [False, True])
def test_scalar_growth_matches_frozen_outputs(growth_expected, model, detect):
    batch = _settings(n=1, model=model, detect=detect, mixed=True)
    settings = PipingSettings()
    settings.set_params_fromdict(
        {name: float(value[0]) for name, value in batch.get_settings_as_dict().items()}
    )
    settings.set_param("dldt_modeltype", model)
    settings.set_param("detect_true", detect)
    settings.set_waterstand_tijdreeks(batch.timegrid[:, 0], batch.h_timeseries[:, 0])
    _assert_growth_equal(
        pipe_growth(settings), growth_expected, f"scalar_{model}_{int(detect)}"
    )
    assert isinstance(_z_pipelengte(settings), float)


@pytest.mark.parametrize("model", ["constant", "Pol2019", "Pol2022"])
@pytest.mark.parametrize(
    "method", ["responsfactor", "responsfactor_phi_gem", "TRWD_4A"]
)
def test_nonuniform_series_matches_frozen_outputs(growth_expected, model, method):
    settings = _settings(n=32, model=model, detect=True, mixed=True)
    settings.set_param("methode_stijghoogte", method)
    time = np.array([0.0, 0.2, 1.5, 12.0, 27.3, 30.0, 36.7, 50.1, 60.0])
    settings.set_waterstand_tijdreeks(
        time, np.interp(time, [0, 30, 60], [0.0, 14.0, 0.0])
    )
    _assert_growth_equal(
        pipe_growth(settings), growth_expected, f"nonuniform_{model}_{method}"
    )


@pytest.mark.parametrize(
    "case", ["no-erosion", "equal-L", "initial-uplift", "dry-gap", "final-intervention"]
)
def test_growth_boundaries_match_frozen_outputs(growth_expected, case):
    settings = _settings(n=32, model="Pol2019", detect=True)
    if case == "no-erosion":
        settings.set_param("r_exit", np.zeros(32))
    elif case == "equal-L":
        settings.set_param("l_ini", np.ones(32))
    elif case == "initial-uplift":
        time = settings.timegrid[:, 0]
        settings.set_waterstand_tijdreeks(
            time, np.interp(time, [0, 10, 20, 40, 60], [14.0, 0.0, 0.0, 14.0, 0.0])
        )
    elif case == "dry-gap":
        time = np.arange(121.0)
        settings.set_waterstand_tijdreeks(
            time, np.interp(time, [0, 20, 40, 60, 90, 120], [0, 14, 0, 0, 14, 0])
        )
    else:
        settings.set_param(
            "intervention_hours", growth_expected["final_intervention_hours"]
        )
    actual = pipe_growth(settings)
    _assert_growth_equal(actual, growth_expected, f"boundary_{case}")
    if case == "final-intervention":
        assert np.isnan(actual.t_intervention).all()
    elif case == "equal-L":
        assert np.isnan(actual.t_fail).all()


def test_growth_preserves_input_settings():
    settings = _settings(n=32, detect=True, mixed=True)
    before = {
        name: value.copy() for name, value in settings.get_settings_as_dict().items()
    }
    before.update(
        timegrid=settings.timegrid.copy(), h_timeseries=settings.h_timeseries.copy()
    )
    pipe_growth(settings)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(settings, name), value)


@pytest.mark.parametrize("n", [1, 32])
@pytest.mark.parametrize("erosion", [False, True])
def test_pipe_growth_has_one_result_shape(n, erosion):
    settings = _settings(n=n)
    if not erosion:
        settings.set_param("r_exit", np.zeros(n))
    result = pipe_growth(settings)
    assert result.l_time.shape == (len(settings.timegrid), n)
    for name in result._fields:
        if name != "l_time":
            assert getattr(result, name).shape == (n,)
    if not erosion:
        expected = np.broadcast_to(settings.l_ini * settings.L, result.l_time.shape)
        np.testing.assert_array_equal(result.l_time, expected)


@pytest.mark.parametrize(
    "model, technique",
    [
        ("Pol2022", {"Rekentechniek": "FORM AbdoRackwitz", "MaximumCallsNumber": 2000}),
        ("Pol2022", {"Rekentechniek": "FORM Cobyla", "MaximumCallsNumber": 2000}),
        ("Pol2022", MC_2000),
        ("Pol2019", MC_2000),
        ("constant", MC_2000),
        ("Pol2022", DS_10),
        ("Pol2019", DS_10),
        ("constant", DS_10),
    ],
    ids=[
        "FORM-AbdoRackwitz",
        "FORM-Cobyla",
        "MC-Pol2022",
        "MC-Pol2019",
        "MC-constant",
        "DS-Pol2022",
        "DS-Pol2019",
        "DS-constant",
    ],
)
def test_full_solver_matches_frozen_outputs(solver_expected, model, technique):
    df = (
        _input_df(("Lognormaal", 0.1, 0.05)) if model == "Pol2022" else _model_df(model)
    )
    time, waterlevel = _hydrograph(14.0)
    actual = _timedep(df, time, waterlevel, technique)
    _assert_solver_equal(
        actual, solver_expected[f"{model}_{technique['Rekentechniek']}"]
    )


def test_simulation_with_stochastic_intervention_matches_frozen_outputs(
    solver_expected,
):
    df = _model_df(
        "Pol2022",
        factor=4.0,
        detect_true=("geen stochast", 1.0),
        intervention_hours=("Lognormaal", 30.0, 5.0),
    )
    time, waterlevel = _hydrograph(14.0)
    actual = _timedep(df, time, waterlevel, MC_2000)
    _assert_solver_equal(actual, solver_expected["stochastic_intervention"])
