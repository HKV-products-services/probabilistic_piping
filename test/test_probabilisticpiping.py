from pathlib import Path

import numpy as np
import pandas as pd

from probabilistic_piping import (
    ProbInput,
    ProbPipingFixedWaterlevel,
    ProbPipingFixedWaterlevelSimple,
)


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
