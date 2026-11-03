import subprocess
import sys
from pathlib import Path

import numpy as np
import openturns as ot
import pandas as pd
import pytest

from probabilistic_piping import (
    PipingEquations,
    PipingSettings,
    ProbInput,
    ProbPipingFixedWaterlevel,
    ProbPipingTimeDepWaterlevel,
    RelevantStochasts,
)

# Instellingen zonder stochast: Pol2022, geen beginpipe en geen interventie
GROWTH_FIXED = {
    "dldt_modeltype": ("geen stochast", "Pol2022"),
    "l_ini": ("geen stochast", 0.0),
    "detect_true": ("geen stochast", 0.0),
    "intervention_hours": ("geen stochast", 0.0),
}
MC_2000 = {
    "Rekentechniek": "Monte Carlo",
    "BlockSize": 1000,
    "MaximumOuterSampling": 2,
    "MaximumCoefficientOfVariation": 0.0,
}


def _input_df(dldt_Ce):
    """Base input from the test data, with the growth parameters added."""
    data_path = Path(__file__).parent / "data" / "full_test.xlsx"
    df = pd.read_excel(data_path, sheet_name="input", index_col=0, header=0)
    rows = {**GROWTH_FIXED, "dldt_Ce": dldt_Ce}
    for name, (kind, val, *spread) in rows.items():
        df.loc[name] = np.nan
        df.loc[name, ["Beschrijving", "Waarde", "Kansverdeling"]] = [name, val, kind]
        if spread:
            df.loc[name, ["Mean", "StDev", "Spreiding"]] = [val, spread[0], spread[0]]
            df.loc[name, ["Spreidingstype", "Verschuiving"]] = [
                "Standaardafwijking",
                0.0,
            ]
    return df


def _hydrograph(peak, plateau=0):
    """Triangle from 0 to the peak over 30 h and back, with an optional plateau."""
    t = np.arange(0, 60 + plateau + 1, 1.0)
    h = np.interp(t, [0, 30, 30 + plateau, 60 + plateau], [0.0, peak, peak, 0.0])
    return t, h


def _timedep(df, t, h, technique, seed=1):
    inp = ProbInput.from_dataframe(df)
    if isinstance(technique, dict):
        inp.calc_options.update(technique)
    else:
        inp.calc_options["Rekentechniek"] = technique
    ot.RandomGenerator.SetSeed(seed)
    prob = ProbPipingTimeDepWaterlevel()
    return prob.timedep_waterlevel_failureprobability(inp, t, h)[1]


# Bijna directe pipegroei geeft hetzelfde als de combi op de piekwaterstand
@pytest.mark.parametrize("peak, expected", [(8.0, 0.1135), (10.0, 0.6560)])
def test_instant_growth_equals_combi(peak, expected):
    df = _input_df(("geen stochast", 1e6))
    inp = ProbInput.from_dataframe(df)
    inp.calc_options.update(MC_2000)
    ot.RandomGenerator.SetSeed(1)
    prob = ProbPipingFixedWaterlevel(progress=False)
    _, res_combi = prob.fixed_waterlevel_failureprobability(inp, z_type="combi", h=peak)

    res = _timedep(df, *_hydrograph(peak), MC_2000)

    assert res.prob_cond == res_combi.prob_cond
    assert np.isclose(res.prob_cond, expected, atol=5e-5)
    # De waterstand van het resultaat is de piek van de reeks
    assert res.h == peak


# Regressie met Pol2022 en een stochastische groeicoefficient
def test_pol2022_form_abdorackwitz():
    df = _input_df(("Lognormaal", 0.1, 0.05))
    t, h = _hydrograph(10.0, plateau=3)
    res = _timedep(df, t, h, "FORM AbdoRackwitz")

    assert res.converged
    assert np.isclose(res.prob_cond, 4.8116e-05, rtol=1e-3)
    assert res.h == h.max()


def test_pol2022_form_cobyla():
    df = _input_df(("Lognormaal", 0.1, 0.05))
    t, h = _hydrograph(10.0, plateau=3)
    res = _timedep(df, t, h, "FORM Cobyla")

    assert 0 < res.prob_cond < 1
    assert not res.converged
    assert res.message


# Een enkel sample met scalaire instellingen geeft dezelfde Z als de batch
def test_z_pipelengte_scalar_equals_batch():
    df = _input_df(("Lognormaal", 0.1, 0.05))
    t, h = _hydrograph(10.0, plateau=3)
    inp = ProbInput.from_dataframe(df)
    names = [k for k in RelevantStochasts().timedep_piping if k in inp.stochasts]
    dist = ot.JointDistribution([inp.stochasts[k] for k in names])
    ot.RandomGenerator.SetSeed(3)
    X = np.array(dist.getSample(40))
    # Schaal de groeicoefficient, zodat een deel van de samples faalt
    X[:, names.index("dldt_Ce")] *= 30

    def new_settings():
        settings = PipingSettings()
        settings.set_params_fromdict(inp.params)
        settings.set_waterstand_tijdreeks(t, h)
        return settings

    settings = new_settings()
    settings.set_params_fromdict({k: X[:, j] for j, k in enumerate(names)})
    z_batch = PipingEquations.z_pipelengte(settings)
    assert (z_batch < 0).any()

    for row, z_expected in zip(X, z_batch):
        settings = new_settings()
        settings.set_params_fromdict({k: float(v) for k, v in zip(names, row)})
        z = PipingEquations.z_pipelengte(settings)
        assert isinstance(z, float)
        assert np.isclose(z, z_expected, rtol=1e-12, atol=1e-12)


# De reeks blijft behouden bij kopieren en deelinstellingen, zonder gedeeld geheugen
def test_series_survive_copy_and_partial():
    t, h = _hydrograph(10.0, plateau=3)
    settings = PipingSettings()
    settings.set_params_fromdict(
        ProbInput.from_dataframe(_input_df(("geen stochast", 0.1))).params
    )
    settings.set_waterstand_tijdreeks(t, h)
    settings.set_params_fromdict({"k": np.linspace(1e-4, 2e-4, 5)})

    for new in [settings.copy(), settings.get_partial_settings(2)]:
        assert new.timegrid.shape == (len(t), 1)
        assert new.h_timeseries.shape == (len(t), 1)
        assert np.array_equal(new.timegrid, settings.timegrid)
        assert np.array_equal(new.h_timeseries, settings.h_timeseries)
        assert not np.shares_memory(new.timegrid, settings.timegrid)
        assert not np.shares_memory(new.h_timeseries, settings.h_timeseries)
    assert len(settings.copy().k) == 5
    assert len(settings.get_partial_settings(2).k) == 1


# De reeks wordt op dt geinterpoleerd
def test_set_waterstand_tijdreeks_dt():
    settings = PipingSettings()
    settings.set_waterstand_tijdreeks([0.0, 2.0, 4.0], [0.0, 2.0, 0.0], dt=1.0)

    assert np.allclose(settings.timegrid[:, 0], [0, 1, 2, 3, 4])
    assert np.allclose(settings.h_timeseries[:, 0], [0, 1, 2, 1, 0])
    with pytest.raises(ValueError, match="niet even lang"):
        settings.set_waterstand_tijdreeks([0.0, 1.0], [0.0])


# Het resultaat van de pipelengte-ontwikkeling heeft namen en blijft uitpakbaar
def test_pipelengte_ontwikkeling_result_fields():
    t, h = _hydrograph(10.0, plateau=3)
    settings = PipingSettings()
    settings.set_params_fromdict(
        ProbInput.from_dataframe(_input_df(("geen stochast", 0.1))).params
    )
    settings.set_waterstand_tijdreeks(t, h)
    result = PipingEquations.pipelengte_ontwikkeling(settings)

    assert result._fields == (
        "l_end",
        "l_time",
        "dldt_avg",
        "dldt_end",
        "t_uplift",
        "t_heave",
        "t_sell",
        "t_first_erosion",
        "t_intervention",
        "t_fail",
    )
    l_end, l_time, *_, t_fail = result
    assert l_end is result.l_end
    assert t_fail is result.t_fail
    assert l_time[0].shape == (len(t), 1)


# Barends is nog niet beschikbaar
def test_mogelijke_erosie_barends_raises():
    settings = PipingSettings(methode_stijghoogte="superpositie_barends")
    with pytest.raises(ValueError, match="Onjuiste methode stijghoogte"):
        PipingEquations.mogelijke_erosie(np.zeros((3, 1)), np.zeros(1), settings)


# De invoer voor de stijghoogte wordt gecontroleerd voor de berekening
def test_stijghoogte_input_is_checked():
    df = _input_df(("geen stochast", 1e6))
    df.loc["methode_stijghoogte", ["Beschrijving", "Waarde", "Kansverdeling"]] = [
        "methode_stijghoogte",
        "TRWD_4A",
        "geen stochast",
    ]
    t, h = _hydrograph(10.0)
    with pytest.raises(ValueError, match="D_vl"):
        _timedep(df, t, h, "FORM AbdoRackwitz")


# Het importeren van het pakket importeert matplotlib niet
def test_import_without_matplotlib():
    code = "import sys, probabilistic_piping; print('matplotlib' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "False"


# De stochastenlijst is combi plus de groei
def test_timedep_piping_list():
    rel = RelevantStochasts()

    assert set(rel.combi) <= set(rel.timedep_piping)
    assert {
        "n_aq",
        "l_ini",
        "dldt_constant",
        "dldt_Ce",
        "dldt_kezdi_C",
        "dldt_kezdi_mv",
        "detect_true",
        "intervention_hours",
    } <= set(rel.timedep_piping)
    assert "cv_vl" not in rel.timedep_piping
    assert "cv_al" not in rel.timedep_piping
