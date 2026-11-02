import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from probabilistic_piping import (
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
