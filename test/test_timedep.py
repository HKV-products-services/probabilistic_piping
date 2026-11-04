import functools
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

# Groeisnelheden per model waarbij Monte Carlo falen vindt (lognormaal)
SPEED_ROW = {
    "constant": "dldt_constant",
    "Pol2019": "dldt_kezdi_C",
    "Pol2022": "dldt_Ce",
}
MODEL_SPEED = {
    "constant": ("Lognormaal", 8e-4, 4e-4),
    "Pol2019": ("Lognormaal", 4.0, 2.0),
    "Pol2022": ("Lognormaal", 0.4, 0.2),
}
MODEL_FIXED = {
    "constant": {},
    "Pol2019": {
        "dldt_kezdi_mv": ("geen stochast", 2.0),
        "n_aq": ("geen stochast", 0.4),
    },
    "Pol2022": {},
}
DS_10 = {
    "Rekentechniek": "DS MediumSafe OrthogonalDirection",
    "BlockSize": 10,
    "MaximumOuterSampling": 1,
    "MaximumCoefficientOfVariation": 0.0,
}


def _df_with(rows, drop=()):
    """Base input from the test data, with the given rows added and others dropped."""
    data_path = Path(__file__).parent / "data" / "full_test.xlsx"
    df = pd.read_excel(data_path, sheet_name="input", index_col=0, header=0)
    for name, (kind, val, *spread) in rows.items():
        df.loc[name] = np.nan
        df.loc[name, ["Beschrijving", "Waarde", "Kansverdeling"]] = [name, val, kind]
        if spread:
            df.loc[name, ["Mean", "StDev", "Spreiding"]] = [val, spread[0], spread[0]]
            df.loc[name, ["Spreidingstype", "Verschuiving"]] = [
                "Standaardafwijking",
                0.0,
            ]
    return df.drop(index=list(drop))


def _input_df(dldt_Ce):
    """Base input from the test data, with the growth parameters added."""
    return _df_with({**GROWTH_FIXED, "dldt_Ce": dldt_Ce})


def _model_df(model, factor=1.0, **changes):
    """Input for a growth model at `factor` times the speeds of the ticket.

    A value in `changes` replaces that row, None removes it.
    """
    rows = {k: v for k, v in GROWTH_FIXED.items() if k != "dldt_modeltype"}
    rows["dldt_modeltype"] = ("geen stochast", model)
    kind, mean, sd = MODEL_SPEED[model]
    rows[SPEED_ROW[model]] = (kind, mean * factor, sd * factor)
    rows.update(MODEL_FIXED[model])
    rows.update(changes)
    return _df_with({k: v for k, v in rows.items() if v is not None})


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


# De reeks stopt op het laatste tijdstip, ook als dt niet past
@pytest.mark.parametrize("dt", [7.0, 0.7])
def test_set_waterstand_tijdreeks_dt_not_dividing(dt):
    settings = PipingSettings()
    settings.set_waterstand_tijdreeks(*_hydrograph(10.0), dt=dt)
    t, h = settings.timegrid[:, 0], settings.h_timeseries[:, 0]

    assert t[-1] == 60.0
    assert h[-1] == 0.0
    assert (h >= 0).all()
    assert (np.diff(t) > 0).all()
    assert np.allclose(np.diff(t)[:-1], dt)
    assert 0 < np.diff(t)[-1] < dt
    if dt == 7.0:
        assert np.isclose(np.diff(t)[-1], 4.0)


# Past dt precies, dan komt er geen extra stap bij
@pytest.mark.parametrize("dt, n", [(0.1, 601), (0.25, 241), (0.3, 201)])
def test_set_waterstand_tijdreeks_dt_dividing(dt, n):
    settings = PipingSettings()
    settings.set_waterstand_tijdreeks(*_hydrograph(10.0), dt=dt)

    assert len(settings.timegrid) == n
    assert settings.timegrid[-1, 0] == 60.0
    assert np.allclose(np.diff(settings.timegrid[:, 0]), dt)


# Een afrondingsfout geeft geen tijdstip voorbij het einde en geen dubbel tijdstip
@pytest.mark.parametrize(
    "t_end, dt, n",
    [
        (0.3, 0.1, 4),  # 3 * 0.1 is in floating point net groter dan 0.3
        (2.1, 0.3, 8),  # np.arange(0, 2.1, 0.3) eindigt al op 2.1
    ],
)
def test_set_waterstand_tijdreeks_dt_rounding(t_end, dt, n):
    settings = PipingSettings()
    settings.set_waterstand_tijdreeks([0.0, t_end], [0.0, 3.0], dt=dt)

    assert len(settings.timegrid) == n
    assert settings.timegrid[-1, 0] == t_end
    assert np.allclose(np.diff(settings.timegrid[:, 0]), dt)
    assert settings.h_timeseries[-1, 0] == 3.0


# Zonder dt blijft de reeks zoals opgegeven
def test_set_waterstand_tijdreeks_without_dt():
    t, h = _hydrograph(10.0)
    settings = PipingSettings()
    settings.set_waterstand_tijdreeks(t, h)

    assert settings.timegrid.shape == (len(t), 1)
    assert np.array_equal(settings.timegrid[:, 0], t)
    assert np.array_equal(settings.h_timeseries[:, 0], h)


# Een ongeldige reeks of dt geeft een foutmelding
@pytest.mark.parametrize(
    "t, h, dt",
    [
        ([0.0], [1.0], None),
        ([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], None),
        ([0.0, 2.0, 1.0], [0.0, 1.0, 0.0], 0.5),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 0.0),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], -1.0),
    ],
)
def test_set_waterstand_tijdreeks_invalid(t, h, dt):
    with pytest.raises(ValueError):
        PipingSettings().set_waterstand_tijdreeks(t, h, dt=dt)


# Een berekening met dt geeft de piek als waterstand
def test_timedep_with_dt_uses_peak():
    inp = ProbInput.from_dataframe(_input_df(("geen stochast", 1e6)))
    inp.calc_options.update(MC_2000)
    ot.RandomGenerator.SetSeed(1)
    prob = ProbPipingTimeDepWaterlevel()
    res = prob.timedep_waterlevel_failureprobability(inp, *_hydrograph(10.0), dt=0.5)[1]

    assert res.h == 10.0


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


# Barends is nog niet beschikbaar, ook niet in een timedep berekening
def test_mogelijke_erosie_barends_raises():
    with pytest.raises(NotImplementedError, match="superpositie_barends"):
        PipingSettings(methode_stijghoogte="superpositie_barends")

    df = _input_df(("geen stochast", 1e6))
    df.loc["methode_stijghoogte", ["Beschrijving", "Waarde", "Kansverdeling"]] = [
        "methode_stijghoogte",
        "superpositie_barends",
        "geen stochast",
    ]
    t, h = _hydrograph(10.0)
    with pytest.raises(NotImplementedError, match="superpositie_barends"):
        _timedep(df, t, h, "FORM AbdoRackwitz")


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


# Elk groeimodel rekent met Monte Carlo en met directional sampling
@pytest.mark.parametrize("model", ["constant", "Pol2019", "Pol2022"])
@pytest.mark.parametrize("technique", [MC_2000, DS_10], ids=["MC", "DS"])
def test_growth_models_and_solvers(model, technique):
    t, h = _hydrograph(10.0, plateau=3)
    res = _timedep(_model_df(model), t, h, technique)

    assert 0 < res.prob_cond < 1


# De kans neemt niet af als de groeisnelheid toeneemt
@pytest.mark.parametrize("model", ["constant", "Pol2019", "Pol2022"])
def test_probability_increases_with_growth_speed(model):
    t, h = _hydrograph(10.0, plateau=3)
    mc = {**MC_2000, "MaximumOuterSampling": 1}
    probs = [
        _timedep(_model_df(model, factor), t, h, mc).prob_cond
        for factor in (0.25, 1, 4)
    ]

    assert np.all(np.diff(probs) >= 0)
    assert probs[-1] > probs[0]


# Met een stochastische beginlengte geeft de batch dezelfde Z als losse samples
def test_z_pipelengte_stochastic_l_ini_scalar_equals_batch():
    df = _model_df("Pol2022")
    t, h = _hydrograph(10.0, plateau=3)
    inp = ProbInput.from_dataframe(df)
    names = [k for k in RelevantStochasts().timedep_piping if k in inp.stochasts]
    ot.RandomGenerator.SetSeed(3)
    X = np.array(ot.JointDistribution([inp.stochasts[k] for k in names]).getSample(200))
    # Schaal de groeicoefficient, zodat de helft van de samples faalt
    X[:, names.index("dldt_Ce")] *= 15
    l_ini = np.random.default_rng(0).uniform(0, 0.4, len(X))

    def new_settings(values):
        settings = PipingSettings()
        settings.set_params_fromdict(inp.params)
        settings.set_waterstand_tijdreeks(t, h)
        settings.set_params_fromdict(values)
        return settings

    columns = {k: X[:, j] for j, k in enumerate(names)}
    z_batch = PipingEquations.z_pipelengte(new_settings({**columns, "l_ini": l_ini}))
    assert 0.2 < (z_batch < 0).mean() < 0.8

    for i, z_expected in enumerate(z_batch):
        values = {k: float(v[i]) for k, v in columns.items()}
        z = PipingEquations.z_pipelengte(new_settings({**values, "l_ini": l_ini[i]}))
        assert np.isclose(z, z_expected, rtol=1e-12, atol=1e-12)


# Ontbrekende of ongeldige groei-invoer geeft een fout voor de berekening
@pytest.mark.parametrize(
    "technique", [MC_2000, "FORM AbdoRackwitz"], ids=["MC", "FORM"]
)
@pytest.mark.parametrize(
    "model, changes, row",
    [
        ("Pol2022", {"dldt_modeltype": None}, "^'dldt_modeltype' is None"),
        ("Pol2022", {"dldt_modeltype": ("geen stochast", "foo")}, "^'dldt_modeltype'"),
        ("Pol2022", {"l_ini": None}, "l_ini"),
        ("Pol2022", {"dldt_Ce": None}, "dldt_Ce"),
        ("constant", {"dldt_constant": None}, "dldt_constant"),
        ("Pol2019", {"n_aq": None}, "n_aq"),
        ("Pol2019", {"n_aq": ("geen stochast", 0.0)}, "n_aq"),
    ],
)
def test_growth_input_is_checked(technique, model, changes, row):
    t, h = _hydrograph(10.0, plateau=3)
    with pytest.raises(ValueError, match=row):
        _timedep(_model_df(model, **changes), t, h, technique)


# Een groeivariabele die een stochast is wordt niet gecontroleerd
def test_growth_input_check_skips_stochasts():
    settings = PipingSettings(dldt_modeltype="Pol2022", l_ini=0.0)

    with pytest.raises(ValueError, match="dldt_Ce"):
        PipingEquations.check_pipegroei(settings)
    PipingEquations.check_pipegroei(settings, {"dldt_Ce"})


# Zonder falen is de kans nul, met de melding van een simulatie zonder falen
def test_zero_failures():
    t, h = _hydrograph(4.0, plateau=3)
    res = _timedep(_model_df("Pol2022"), t, h, MC_2000)

    assert res.prob_cond == 0
    assert "Geen falende samples" in res.message


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
        "intervention_hours",
    } <= set(rel.timedep_piping)
    assert "detect_true" not in rel.timedep_piping
    assert "cv_vl" not in rel.timedep_piping
    assert "cv_al" not in rel.timedep_piping


# Z van de continuiteitscase, bij elke interventietijd van 0 tot 40 uur
@functools.cache
def _z_intervention(detect=True):
    inp = ProbInput.from_dataframe(
        _df_with({**GROWTH_FIXED, "dldt_Ce": ("Lognormaal", 1.0, 0.5)})
    )
    ih = np.round(np.arange(0, 40.0001, 0.01), 2)
    settings = PipingSettings()
    settings.set_params_fromdict(inp.params)
    settings.set_waterstand_tijdreeks(*_hydrograph(10.0, plateau=3))
    names = [k for k in RelevantStochasts().timedep_piping if k in inp.stochasts]
    means = {k: np.full(len(ih), inp.stochasts[k].getMean()[0]) for k in names}
    settings.set_params_fromdict({**means, "intervention_hours": ih})
    PipingEquations.check_pipegroei(settings, set(inp.stochasts))
    settings.set_param("detect_true", detect, verify_integrity=False)
    growth = PipingEquations.pipelengte_ontwikkeling(settings)
    return ih, 1 - growth.l_end / settings.L, growth


# Z is continu en niet stijgend in de interventietijd
def test_intervention_z_continuous_and_monotone():
    ih, z, _ = _z_intervention()

    assert np.abs(np.diff(z)).max() < 0.001
    assert (np.diff(z) <= 1e-12).all()


# Op hele uren is Z gelijk aan de waarden zonder deelstap
def test_intervention_z_on_time_steps():
    ih, z, _ = _z_intervention()
    hours = [0, 20, 24, 25, 26, 27, 30, 40]
    expected = [0.92290, 0.92290, 0.88662, 0.86093, 0.83429, 0.80667, 0.78809, 0.78809]

    assert np.allclose(z[np.searchsorted(ih, hours)], expected, rtol=0, atol=6e-6)


# Zonder detectie speelt de interventietijd geen rol
def test_intervention_detection_off():
    _, z, growth = _z_intervention(detect=False)

    assert np.allclose(z, 0.78809, rtol=0, atol=6e-6)
    assert np.isnan(growth.t_intervention).all()


# De interventietijd is het begin van het uitspoelen plus de uren
def test_intervention_time_is_exact():
    ih, _, growth = _z_intervention()

    assert np.allclose(growth.t_first_erosion, 6.0)
    for hours, expected in [(20.0, 26.0), (20.5, 26.5)]:
        assert growth.t_intervention[np.searchsorted(ih, hours)] == pytest.approx(
            expected
        )


# Een interventie nadat heave voorbij is, geeft geen extra groei in de deelstap
def test_intervention_after_heave_adds_no_growth():
    inp = ProbInput.from_dataframe(
        _df_with(
            {
                **GROWTH_FIXED,
                "dldt_modeltype": ("geen stochast", "constant"),
                "dldt_constant": ("geen stochast", 2e-4),
            }
        )
    )
    # Na de golf 25 uur droog, zodat heave ruim voor het einde stopt
    t = np.arange(0, 88.5, 1.0)
    h = np.interp(t, [0, 30, 33, 63, 88], [0.0, 10.0, 10.0, 0.0, 0.0])
    ih = np.arange(65.0, 80.0, 0.25)
    names = [k for k in RelevantStochasts().timedep_piping if k in inp.stochasts]
    means = {k: np.full(len(ih), inp.stochasts[k].getMean()[0]) for k in names}

    l_end = {}
    for detect in [True, False]:
        settings = PipingSettings()
        settings.set_params_fromdict(inp.params)
        settings.set_waterstand_tijdreeks(t, h)
        settings.set_params_fromdict({**means, "intervention_hours": ih})
        settings.set_param("detect_true", detect, verify_integrity=False)
        l_end[detect] = PipingEquations.pipelengte_ontwikkeling(settings).l_end

    assert np.array_equal(l_end[True], l_end[False])


# Interventie als vaste waarde en als stochast geeft een kans
@pytest.mark.parametrize(
    "hours", [("geen stochast", 30.0), ("Lognormaal", 30.0, 5.0)], ids=["fixed", "ln"]
)
def test_intervention_fixed_and_stochastic(hours):
    df = _model_df(
        "Pol2022",
        factor=4.0,
        detect_true=("geen stochast", 1.0),
        intervention_hours=hours,
    )
    t, h = _hydrograph(10.0, plateau=3)
    res = _timedep(df, t, h, MC_2000)

    assert 0 < res.prob_cond < 1


# detect_true accepteert alleen een boolean waarde
@pytest.mark.parametrize("value", [0, 1, True, False, 0.0, 1.0, np.float64(1.0)])
def test_detect_true_accepted(value):
    settings = PipingSettings(
        dldt_modeltype="Pol2022", l_ini=0.0, dldt_Ce=1.0, intervention_hours=5.0
    )
    settings.set_param("detect_true", value, verify_integrity=False)
    PipingEquations.check_pipegroei(settings)

    assert settings.detect_true is bool(value)


@pytest.mark.parametrize("value", [2, 0.5, "ja", None, np.nan, [1, 0]])
def test_detect_true_invalid(value):
    settings = PipingSettings(
        dldt_modeltype="Pol2022", l_ini=0.0, dldt_Ce=1.0, intervention_hours=5.0
    )
    settings.set_param("detect_true", value, verify_integrity=False)

    with pytest.raises(ValueError, match="detect_true"):
        PipingEquations.check_pipegroei(settings)


# detect_true als stochast, of detectie zonder interventietijd, geeft een fout
def test_detect_true_stochast_and_missing_hours_raise():
    settings = PipingSettings(dldt_modeltype="Pol2022", l_ini=0.0, dldt_Ce=1.0)

    with pytest.raises(ValueError, match="detect_true"):
        PipingEquations.check_pipegroei(settings, {"detect_true"})
    PipingEquations.check_pipegroei(settings)
    settings.set_param("detect_true", True, verify_integrity=False)
    with pytest.raises(ValueError, match="intervention_hours"):
        PipingEquations.check_pipegroei(settings)
    PipingEquations.check_pipegroei(settings, {"intervention_hours"})


@pytest.mark.parametrize(
    "changes, row",
    [
        ({"detect_true": ("Lognormaal", 1.0, 0.5)}, "detect_true"),
        ({"detect_true": ("geen stochast", 2.0)}, "detect_true"),
        (
            {"detect_true": ("geen stochast", 1.0), "intervention_hours": None},
            "intervention_hours",
        ),
    ],
)
def test_detect_true_invalid_input_rows(changes, row):
    t, h = _hydrograph(10.0, plateau=3)
    with pytest.raises(ValueError, match=row):
        _timedep(_model_df("Pol2022", **changes), t, h, MC_2000)
