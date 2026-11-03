import math
from pathlib import Path

import numpy as np
import openturns as ot
import pandas as pd
import pytest

from probabilistic_piping import (
    PipingSettings,
    ProbInput,
    ProbPipingFixedWaterlevel,
    ProbPipingFixedWaterlevelSimple,
)


def test_full_semiprob():
    data_path = Path(__file__).parent / "data" / "full_test.xlsx"
    df_input = pd.read_excel(data_path, sheet_name="input", index_col=0, header=0)

    # Create probabilistic piping object and calculate semi-probabilistic probability
    prob = ProbPipingFixedWaterlevel(progress=False)
    inp_data = ProbInput.from_dataframe(df_input)
    result = prob.fixed_waterlevel_semiprob(inp_data)

    assert np.isclose(0.414071792720178, result.beta_u, rtol=1e-08, atol=1e-08)
    assert np.isclose(0.184755478797373, result.beta_h, rtol=1e-08, atol=1e-08)
    assert np.isclose(7.77022397615446, result.beta_p, rtol=1e-08, atol=1e-08)
    assert np.isclose(3.94129173741931e-15, result.pf_combi, rtol=1e-08, atol=1e-08)


def test_full_fragilitycurve_simple():
    data_path = Path(__file__).parent / "data" / "full_test.xlsx"
    with pd.ExcelFile(data_path) as xlsx:
        df_input = pd.read_excel(xlsx, sheet_name="input", index_col=0, header=0)
        df_exp_results = pd.read_excel(
            xlsx, sheet_name="output_simple", index_col=None, header=0
        )

    # Create probabilistic piping object and calculate fragility curve
    prob = ProbPipingFixedWaterlevelSimple(progress=False)
    inp_form = ProbInput.from_dataframe(df_input)
    _, pu, ph, pp, pc = prob.fixed_waterlevel_fragilitycurve(inp_form)

    # Create results dataframe similar to the expected results
    df_results = {
        "waterstand": [],
        "prob_uplift": [],
        "prob_heave": [],
        "prob_sellmeijer": [],
        "mechanisme": [],
        "prob_cond": [],
    }
    for ru, rh, rp, rc in zip(pu.results, ph.results, pp.results, pc.results):
        df_results["waterstand"].append(ru.h)
        df_results["prob_uplift"].append(ru.prob_cond)
        df_results["prob_heave"].append(rh.prob_cond)
        df_results["prob_sellmeijer"].append(rp.prob_cond)
        df_results["mechanisme"].append(rc.mechanism)
        df_results["prob_cond"].append(rc.prob_cond)
    df_results = pd.DataFrame(df_results)

    # Compare the results column by column
    for name in df_results.columns:
        if name == "mechanisme":
            col_equal = df_results[name].to_numpy() == df_exp_results[name].to_numpy()
            assert col_equal.all()
        else:
            assert np.allclose(
                df_results[name].to_numpy(),
                df_exp_results[name].to_numpy(),
                rtol=1e-8,
                atol=1e-8,
            )


def test_full_fragilitycurve():
    data_path = Path(__file__).parent / "data" / "full_test.xlsx"
    with pd.ExcelFile(data_path) as xlsx:
        df_input = pd.read_excel(xlsx, sheet_name="input", index_col=0, header=0)
        df_exp_results = pd.read_excel(
            xlsx, sheet_name="output", index_col=None, header=0
        )

    # Create probabilistic piping object and calculate fragility curve
    prob = ProbPipingFixedWaterlevel(progress=False)
    inp_form = ProbInput.from_dataframe(df_input)
    _, pc = prob.fixed_waterlevel_fragilitycurve(inp_form)

    # Create results dataframe similar to the expected results
    df_results = {
        "waterstand": [],
        "prob_cond": [],
    }
    for rc in pc.results:
        df_results["waterstand"].append(rc.h)
        df_results["prob_cond"].append(rc.prob_cond)
    df_results = pd.DataFrame(df_results)

    # Compare the results column by column
    for name in df_results.columns:
        assert np.allclose(
            df_results[name].to_numpy(),
            df_exp_results[name].to_numpy(),
            rtol=1e-8,
            atol=1e-8,
        )


# Synthetic layer schematisation for the stijghoogte methods
LAYERS = {
    "k": 1e-4,
    "D": 5.0,
    "D_vl": 2.0,
    "D_al": 3.0,
    "k_vl": 1e-6,
    "k_al": 2e-6,
    "L_vl": 50.0,
    "L_al": 100.0,
    "L_dijkbasis": 40.0,
}
LAYERS_NONZERO = ["k", "D", "D_vl", "D_al", "k_vl", "k_al"]


def _input_df(methode="responsfactor", params=None):
    """Base input from the test data, with a stijghoogte method and extra params."""
    data_path = Path(__file__).parent / "data" / "full_test.xlsx"
    df = pd.read_excel(data_path, sheet_name="input", index_col=0, header=0)
    params = {"methode_stijghoogte": methode, **(params or {})}
    for name, val in params.items():
        df.loc[name, "Kansverdeling"] = "geen stochast"
        df.loc[name, "Waarde"] = val
    return df


def _semiprob(df):
    prob = ProbPipingFixedWaterlevel(progress=False)
    return prob.fixed_waterlevel_semiprob(ProbInput.from_dataframe(df))


def _failureprobability(df, z_type="heave"):
    prob = ProbPipingFixedWaterlevel(progress=False)
    _, result = prob.fixed_waterlevel_failureprobability(
        ProbInput.from_dataframe(df), z_type=z_type
    )
    return result.prob_cond


def _trwd_r_exit(layers):
    """Hand calculation of TRWD model 4A."""
    lam_vl = math.sqrt(layers["k"] * layers["D"] * layers["D_vl"] / layers["k_vl"])
    lam_al = math.sqrt(layers["k"] * layers["D"] * layers["D_al"] / layers["k_al"])
    a_vl = lam_vl * math.tanh(layers["L_vl"] / lam_vl)
    a_al = lam_al * math.tanh(layers["L_al"] / lam_al)
    return a_al / (a_vl + layers["L_dijkbasis"] + a_al)


def test_stijghoogte_phi_gem_semiprob_value():
    df = _input_df("responsfactor_phi_gem", {"phi_gem": 2.0, "h_gem": 3.0})
    result = _semiprob(df)

    h, r_exit = 5.0, df.loc["r_exit", "Waarde"]
    assert np.isclose(result["phi_exit"], 2.0 + r_exit * (h - 3.0), rtol=1e-12)


def test_stijghoogte_TRWD_4A_semiprob_value():
    df = _input_df("TRWD_4A", LAYERS)
    result = _semiprob(df)

    h_exit = df.loc["h_exit", "Waarde"]
    r_exit = _trwd_r_exit(LAYERS)
    assert np.isclose(result["phi_exit"], h_exit + r_exit * (5.0 - h_exit))


@pytest.mark.parametrize("name", LAYERS_NONZERO)
@pytest.mark.parametrize("bad", [0.0, np.nan])
def test_stijghoogte_TRWD_4A_rejects_zero_and_nan(name, bad):
    df = _input_df("TRWD_4A", {**LAYERS, name: bad})
    with pytest.raises(ValueError, match=name):
        _semiprob(df)
    with pytest.raises(ValueError, match=name):
        _failureprobability(df)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", ["L_vl", "L_al", "L_dijkbasis"])
def test_stijghoogte_TRWD_4A_length_zero_allowed_nan_rejected(name):
    result = _semiprob(_input_df("TRWD_4A", {**LAYERS, name: 0.0}))
    assert np.isfinite(result["phi_exit"])
    with pytest.raises(ValueError, match=name):
        _semiprob(_input_df("TRWD_4A", {**LAYERS, name: np.nan}))


def test_stijghoogte_phi_gem_zero_allowed():
    df = _input_df("responsfactor_phi_gem", {"phi_gem": 0.0, "h_gem": 0.0})
    result = _semiprob(df)
    assert np.isclose(result["phi_exit"], df.loc["r_exit", "Waarde"] * 5.0)

    df = _input_df("responsfactor_phi_gem", {"phi_gem": 0.0, "h_gem": 5.0})
    pf = _failureprobability(df)
    assert np.isfinite(pf)


@pytest.mark.parametrize("name", ["phi_gem", "h_gem"])
def test_stijghoogte_phi_gem_rejects_nan(name):
    params = {"phi_gem": 0.5, "h_gem": 5.0, name: np.nan}
    df = _input_df("responsfactor_phi_gem", params)
    with pytest.raises(ValueError, match=name):
        _semiprob(df)
    with pytest.raises(ValueError, match=name):
        _failureprobability(df)


def test_stijghoogte_unused_rows_allowed():
    # Default method ignores phi_gem and layer rows, even if zero or NaN
    base = _failureprobability(_input_df())
    layers0 = {k: 0.0 for k in LAYERS if k not in ["k", "D"]}
    bad_params = {**layers0, "phi_gem": np.nan, "h_gem": np.nan}
    assert np.isclose(_failureprobability(_input_df(params=bad_params)), base)
    assert np.isfinite(_semiprob(_input_df(params=bad_params))["phi_exit"])

    # phi_gem method ignores the layer rows
    params = {**layers0, "phi_gem": 0.5, "h_gem": 5.0}
    df = _input_df("responsfactor_phi_gem", params)
    assert np.isfinite(_failureprobability(df))
    assert np.isfinite(_semiprob(df)["phi_exit"])

    # TRWD_4A ignores phi_gem and h_gem
    df = _input_df("TRWD_4A", {**LAYERS, "phi_gem": np.nan, "h_gem": np.nan})
    assert np.isfinite(_failureprobability(df))
    assert np.isfinite(_semiprob(df)["phi_exit"])


def test_stijghoogte_TRWD_4A_all_lengths_zero_rejected():
    df = _input_df("TRWD_4A", {**LAYERS, "L_vl": 0.0, "L_al": 0.0, "L_dijkbasis": 0.0})
    with pytest.raises(ValueError, match="alle drie nul"):
        _semiprob(df)
    with pytest.raises(ValueError, match="alle drie nul"):
        _failureprobability(df)


def test_stijghoogte_TRWD_4A_simple():
    prob = ProbPipingFixedWaterlevelSimple(progress=False)
    df = _input_df("TRWD_4A", LAYERS)
    _, ru, rh, _, _ = prob.fixed_waterlevel_failureprobability(
        ProbInput.from_dataframe(df)
    )
    assert np.isclose(ru.prob_cond, _failureprobability(df, "uplift"))
    assert np.isclose(rh.prob_cond, _failureprobability(df, "heave"))

    df = _input_df("TRWD_4A", {**LAYERS, "D_vl": 0.0})
    with pytest.raises(ValueError, match="D_vl"):
        prob.fixed_waterlevel_failureprobability(ProbInput.from_dataframe(df))


def _add_stochast(df, name, mean, rel_spread=0.1):
    """Add a stochast row, copied from the D_cover row."""
    df.loc[name] = df.loc["D_cover"]
    df.loc[name, ["Mean", "StDev", "Spreiding"]] = [
        mean,
        rel_spread * mean,
        rel_spread * mean,
    ]
    df.loc[name, "Spreidingstype"] = "Standaardafwijking"


def _pf_cobyla(df, z_type):
    inp = ProbInput.from_dataframe(df)
    inp.calc_options["Rekentechniek"] = "FORM Cobyla"
    prob = ProbPipingFixedWaterlevel(progress=False)
    _, result = prob.fixed_waterlevel_failureprobability(inp, z_type=z_type)
    return result


@pytest.mark.parametrize("z_type", ["uplift", "heave", "combi"])
def test_stijghoogte_methods_equal_responsfactor(z_type):
    # Same fixed h_exit, k, D and r_exit give the same failure probability
    fixed = {"h_exit": 0.1, "k": LAYERS["k"], "D": LAYERS["D"]}
    r_trwd = _trwd_r_exit(LAYERS)
    ref = _failureprobability(_input_df(params={**fixed, "r_exit": r_trwd}), z_type)

    phi_gem = {"phi_gem": 0.1, "h_gem": 0.1, "r_exit": r_trwd}
    df = _input_df("responsfactor_phi_gem", {**fixed, **phi_gem})
    assert np.isclose(_failureprobability(df, z_type), ref, rtol=1e-6, atol=0)

    df = _input_df("TRWD_4A", {**LAYERS, "h_exit": 0.1, "r_exit": r_trwd})
    assert np.isclose(_failureprobability(df, z_type), ref, rtol=1e-6, atol=0)


@pytest.mark.parametrize("z_type", ["uplift", "heave", "combi"])
@pytest.mark.parametrize("name", list(LAYERS))
def test_stijghoogte_TRWD_4A_inputs_are_sampled(name, z_type):
    df = _input_df("TRWD_4A", LAYERS)
    fixed = _pf_cobyla(df, z_type)
    assert name not in fixed.physical_design

    _add_stochast(df, name, LAYERS[name], rel_spread=1e-3)
    result = _pf_cobyla(df, z_type)
    assert name in result.physical_design
    # combi has a very low probability, FORM does not converge there
    if z_type != "combi":
        assert np.isclose(result.prob_cond, fixed.prob_cond, rtol=1e-3, atol=0)


@pytest.mark.parametrize("z_type", ["uplift", "heave", "combi"])
def test_stijghoogte_TRWD_4A_stochasts_do_not_change_responsfactor(z_type):
    base = _input_df()
    df = _input_df()
    for name, val in LAYERS.items():
        if name not in ["k", "D"]:
            _add_stochast(df, name, val)
    assert _failureprobability(df, z_type) == _failureprobability(base, z_type)


def _simple_pf(df):
    prob = ProbPipingFixedWaterlevelSimple(progress=False)
    _, ru, rh, _, _ = prob.fixed_waterlevel_failureprobability(
        ProbInput.from_dataframe(df)
    )
    return ru.prob_cond, rh.prob_cond


@pytest.mark.parametrize("name", ["cv_al", "cv_vl", "no_such_setting"])
@pytest.mark.parametrize("as_stochast", [False, True])
def test_rows_that_are_not_settings_are_ignored(name, as_stochast):
    base = _input_df()
    df = _input_df()
    if as_stochast:
        _add_stochast(df, name, 1e-6)
    else:
        df.loc[name, "Kansverdeling"] = "geen stochast"
        df.loc[name, "Waarde"] = 1e-6

    pd.testing.assert_series_equal(_semiprob(df), _semiprob(base), check_exact=True)
    for z_type in ["uplift", "heave", "combi"]:
        assert _failureprobability(df, z_type) == _failureprobability(base, z_type)
    assert _simple_pf(df) == _simple_pf(base)


def _stochast_range(df, name):
    dist = ProbInput.from_dataframe(df).stochasts[name]
    rng = dist.getRange()
    return rng.getLowerBound()[0], rng.getUpperBound()[0]


def _design_r_exit(df):
    prob = ProbPipingFixedWaterlevel(progress=False)
    _, result = prob.fixed_waterlevel_failureprobability(
        ProbInput.from_dataframe(df), z_type="heave"
    )
    return result.physical_design["r_exit"]


def _truncated_r_exit(left=np.nan, right=np.nan):
    df = _input_df()
    df.loc["r_exit", ["Afknot_links", "Afknot_rechts"]] = [left, right]
    return df


def test_two_sided_truncation_from_input_table():
    mean, sd = 0.8, 0.08
    left, right = mean - sd, mean + 1.5 * sd
    assert _design_r_exit(_truncated_r_exit()) < left

    df = _truncated_r_exit(left, right)
    lower, upper = _stochast_range(df, "r_exit")
    assert np.isclose(lower, left)
    assert np.isclose(upper, right)
    assert left <= _design_r_exit(df) <= right


def test_no_and_one_sided_truncation_ranges():
    lower, upper = _stochast_range(_truncated_r_exit(), "r_exit")
    assert lower < 0.8 - 5 * 0.08
    assert upper > 0.8 + 5 * 0.08

    lower, upper = _stochast_range(_truncated_r_exit(left=0.7), "r_exit")
    assert np.isclose(lower, 0.7)
    assert upper > 0.8 + 5 * 0.08

    lower, upper = _stochast_range(_truncated_r_exit(right=0.9), "r_exit")
    assert lower < 0.8 - 5 * 0.08
    assert np.isclose(upper, 0.9)


def test_truncation_left_above_right_raises():
    with pytest.raises(TypeError):
        ProbInput.from_dataframe(_truncated_r_exit(0.9, 0.7))


def test_cv_not_in_piping_settings():
    from probabilistic_piping import PipingSettings

    settings = PipingSettings(cv_al=1e-6)
    assert not hasattr(settings, "cv_al")
    assert not hasattr(settings, "cv_vl")
    assert "cv_vl" not in PipingSettings.__doc__
    assert "cv_al" not in PipingSettings.__doc__


# Solver failures
FORM_AR = "FORM AbdoRackwitz"
FORM_COBYLA = "FORM Cobyla"
MONTE_CARLO = "Monte Carlo"
DIRECTIONAL = "DS MediumSafe OrthogonalDirection"
SIM_OPTIONS = {
    MONTE_CARLO: {"BlockSize": 100, "MaximumOuterSampling": 2},
    DIRECTIONAL: {"BlockSize": 10, "MaximumOuterSampling": 2},
}


def _prob_input(technique, **options):
    inp = ProbInput.from_dataframe(_input_df())
    inp.calc_options.update({"Rekentechniek": technique, **options})
    return inp


def _combi(technique, **options):
    prob = ProbPipingFixedWaterlevel(progress=False)
    return prob.fixed_waterlevel_failureprobability(
        _prob_input(technique, **options), z_type="combi"
    )[1]


def _custom_z(z_func, technique, **options):
    return _custom_z_input(z_func, _prob_input(technique, **options))


def _custom_z_input(z_func, inp):
    settings = PipingSettings()
    settings.set_params_fromdict({**inp.params, "h": 5.0})
    prob = ProbPipingFixedWaterlevel(progress=False)
    return prob.prob_calculation(5.0, "heave", z_func, settings, inp)[1]


def _constant_z(value):
    return lambda s: (
        value * np.ones(np.shape(s.h_exit)) if np.ndim(s.h_exit) else float(value)
    )


def _failing_z(s):
    raise ValueError("fout in z")


@pytest.mark.parametrize("technique", [FORM_AR, FORM_COBYLA])
def test_solver_failure_raises_the_real_error(technique):
    with pytest.raises(RuntimeError) as error:
        _combi(technique, MaximumCallsNumber=3)
    assert "NotYetImplemented" not in str(error.value)


@pytest.mark.parametrize("start", ["mean", "slice"])
def test_zero_gradient_raises_the_real_error(start):
    with pytest.raises(RuntimeError, match="gradient"):
        _custom_z(_constant_z(1.0), FORM_AR, **{"FORM start": start})


def test_residual_recovery_keeps_values_and_adds_message():
    result = _combi(FORM_AR, MaximumIterationNumber=1)
    assert 0 < result.prob_cond < 1
    assert not result.converged
    assert "not on the limit state" in result.message
    assert result.physical_design is not None
    assert result.importancefactors is not None


@pytest.mark.parametrize("technique", [FORM_AR, FORM_COBYLA, MONTE_CARLO, DIRECTIONAL])
def test_error_in_z_raises_original_error(technique):
    with pytest.raises(ValueError, match="fout in z"):
        _custom_z(_failing_z, technique, **SIM_OPTIONS.get(technique, {}))


def test_error_in_z_raises_original_error_from_slice_start():
    with pytest.raises(ValueError, match="fout in z"):
        _custom_z(_failing_z, FORM_AR, **{"FORM start": "slice"})


@pytest.mark.parametrize("technique", [FORM_AR, FORM_COBYLA, MONTE_CARLO])
def test_unknown_stijghoogte_method_raises_value_error(technique):
    inp = ProbInput.from_dataframe(_input_df("onbekende_methode"))
    inp.calc_options.update({"Rekentechniek": technique})
    inp.calc_options.update(SIM_OPTIONS.get(technique, {}))
    prob = ProbPipingFixedWaterlevel(progress=False)
    with pytest.raises(ValueError):
        prob.fixed_waterlevel_failureprobability(inp, z_type="heave")


def test_design_point_on_the_mean_has_no_importancefactors():
    result = _custom_z(lambda s: s.h_exit + 1.5, FORM_AR)
    assert np.isclose(result.prob_cond, 0.5)
    assert result.importancefactors is None
    assert result.physical_design is not None


@pytest.mark.parametrize("technique", [MONTE_CARLO, DIRECTIONAL])
def test_simulation_without_failures(technique):
    result = _custom_z(_constant_z(1.0), technique, **SIM_OPTIONS[technique])
    assert result.prob_cond == 0
    assert not result.converged
    assert result.message
    assert result.z_val is None
    assert result.physical_design is None
    assert result.standard_design is None
    assert result.importancefactors is None


@pytest.mark.parametrize("technique", [MONTE_CARLO, DIRECTIONAL])
def test_simulation_with_only_failures(technique):
    result = _custom_z(_constant_z(-1.0), technique, **SIM_OPTIONS[technique])
    assert result.prob_cond == 1
    assert result.message
    assert result.z_val is None
    assert result.physical_design is None
    assert result.standard_design is None
    assert result.importancefactors is None


@pytest.mark.parametrize("technique", [FORM_AR, FORM_COBYLA])
def test_normal_run_has_no_message(technique):
    assert _combi(technique).message is None


@pytest.mark.parametrize("technique", [MONTE_CARLO, DIRECTIONAL])
def test_normal_simulation_has_no_message(technique):
    ot.RandomGenerator.SetSeed(1)
    result = _custom_z(lambda s: s.h_exit + 1.5, technique, **SIM_OPTIONS[technique])
    assert 0 < result.prob_cond < 1
    assert result.message is None


def test_solver_failure_stops_a_fragilitycurve():
    prob = ProbPipingFixedWaterlevel(progress=False)
    inp = _prob_input(FORM_AR, MaximumCallsNumber=12)
    with pytest.raises(RuntimeError) as error:
        prob.fixed_waterlevel_fragilitycurve(inp, hlist=[0.0, 2.0, 4.0, 6.0])
    assert "NotYetImplemented" not in str(error.value)


# Uitlijning simulatie met FORM
MC_OPTIONS = {"BlockSize": 1000, "MaximumOuterSampling": 10}
DS_OPTIONS = {"BlockSize": 100, "MaximumOuterSampling": 5}


def _at_waterlevel(z_type, h, technique, **options):
    inp = _prob_input(technique, MaximumCoefficientOfVariation=0, **options)
    ot.RandomGenerator.SetSeed(1)
    prob = ProbPipingFixedWaterlevel(progress=False)
    return prob.fixed_waterlevel_failureprobability(inp, z_type=z_type, h=h)[1]


def _assert_matches_form(z_type, h, technique, options, max_distance):
    form = _at_waterlevel(z_type, h, FORM_AR)
    result = _at_waterlevel(z_type, h, technique, **options)
    assert 0 < result.prob_cond < 1
    distance = math.dist(
        [result.standard_design[k] for k in form.standard_design],
        list(form.standard_design.values()),
    )
    assert distance < max_distance
    assert abs(result.z_val) < 0.1
    strong = [k for k, a in form.importancefactors.items() if abs(a) >= 0.1]
    assert strong
    for k in strong:
        assert np.sign(result.importancefactors[k]) == np.sign(
            form.importancefactors[k]
        )


# heave bij h=1.5 heeft pf 0.79, waar een gewone tekenwissel fout zou zijn
@pytest.mark.parametrize("z_type, h", [("heave", 0.5), ("uplift", 1.0), ("heave", 1.5)])
def test_monte_carlo_matches_form(z_type, h):
    _assert_matches_form(z_type, h, MONTE_CARLO, MC_OPTIONS, 0.25)


def test_directional_sampling_matches_form():
    _assert_matches_form("heave", 0.5, DIRECTIONAL, DS_OPTIONS, 0.5)


def test_simulation_design_point_distance_is_beta():
    # De kans komt uit het simulatieresultaat en niet uit de richting
    result = _at_waterlevel("heave", 0.5, MONTE_CARLO, **MC_OPTIONS)
    beta = ot.Normal().computeInverseSurvivalFunction(result.prob_cond)[0]
    assert math.isclose(beta, math.hypot(*result.standard_design.values()))


def test_simulation_on_the_mean_has_no_importancefactors():
    # Precies de helft van de samples faalt, dus pf is 0.5 en beta 0
    calls = []

    def half_failing(s):
        if np.ndim(s.h_exit):
            return np.where(np.arange(len(s.h_exit)) % 2 == 0, -1.0, 1.0)
        calls.append(1)
        return -1.0 if len(calls) % 2 else 1.0

    ot.RandomGenerator.SetSeed(1)
    result = _custom_z(half_failing, MONTE_CARLO, **SIM_OPTIONS[MONTE_CARLO])
    assert result.prob_cond == 0.5
    assert result.importancefactors is None
    assert result.physical_design is not None
