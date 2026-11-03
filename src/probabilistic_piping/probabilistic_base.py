from contextlib import contextmanager
from dataclasses import field
from typing import Callable

import numpy as np
import openturns as ot
import tqdm.auto as tqdm
from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass

from probabilistic_piping.piping_equations import PipingEquations
from probabilistic_piping.piping_settings import PipingSettings
from probabilistic_piping.probabilistic_io import ProbInput, ProbResult


@dataclass
class RelevantStochasts:
    """
    Dataclass to hold relevant stochastic variables for different types of analyses.

    Attributes
    ----------
    algemeen : list of str
        General stochastic variables.
    uplift : list of str
        Stochastic variables relevant for uplift analysis.
    heave : list of str
        Stochastic variables relevant for heave analysis.
    sellmeijer : list of str
        Stochastic variables relevant for Sellmeijer analysis.
    combi : list of str
        Combined stochastic variables for combined analyses.
    TRWD_4A : list of str
        Stochastic variables that are also used by uplift, heave and combi
        when methode_stijghoogte is "TRWD_4A".
    """

    algemeen: list[str] = field(
        default_factory=lambda: [
            "d_70m",
            "g",
            "gamma_water",
            "r_c",
            "v",
            "gamma_sp",
        ]
    )
    uplift: list[str] = field(
        default_factory=lambda: [
            "D_cover",
            "gamma_sat",
            "h_exit",
            "m_u",
            "r_exit",
            "phi_gem",
            "h_gem",
        ]
    )
    heave: list[str] = field(
        default_factory=lambda: [
            "D_cover",
            "h_exit",
            "i_ch",
            "r_exit",
            "phi_gem",
            "h_gem",
        ]
    )
    sellmeijer: list[str] = field(
        default_factory=lambda: [
            "D",
            "D_cover",
            "d_70",
            "eta",
            "h_exit",
            "k",
            "L",
            "m_p",
            "theta",
        ]
    )
    combi: list[str] = field(
        default_factory=lambda: [
            "D_cover",
            "gamma_sat",
            "h_exit",
            "m_u",
            "r_exit",
            "phi_gem",
            "h_gem",
            "i_ch",
            "D",
            "d_70",
            "eta",
            "k",
            "L",
            "m_p",
            "theta",
        ]
    )
    TRWD_4A: list[str] = field(
        default_factory=lambda: [
            "k",
            "D",
            "L_dijkbasis",
            "L_vl",
            "L_al",
            "D_vl",
            "D_al",
            "k_vl",
            "k_al",
        ]
    )


class ProbPipingBase(BaseModel):
    """
    Base class for probabilistic piping calculations.

    Attributes
    ----------
    model_config :ConfigDict
        Configuration for the pydantic model.
    progress : bool
        Flag to indicate if progress should be shown.
    debug : bool
        Flag to indicate if debug information should be printed.
    rel_stochasts : RelevantStochasts
        Relevant stochastic variables for different types of analyses.
    piping_eq : PipingEquations
        Piping equations to use for the calculations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    progress: bool = False
    debug: bool = False
    rel_stochasts: RelevantStochasts = field(default_factory=RelevantStochasts)
    piping_eq: PipingEquations = field(default_factory=PipingEquations)

    def prob_calculation(
        self,
        h: float,
        z_type: str,
        z_func: Callable,
        settings: PipingSettings,
        prob_input: ProbInput,
        copula: ot.Distribution | None = None,
        leave: bool = True,
    ) -> tuple[PipingSettings, ProbResult]:
        """
        Perform a probabilistic calculation.

        Parameters
        ----------
        h : float
            Water level.
        z_type : str
            Type of the limit state function.
        z_func : Callable
            Limit state function.
        settings : PipingSettings
            Piping settings.
        prob_input : ProbInput
            Probabilistic input settings.
        copula : ot.Distribution or None, optional
            Copula distribution, by default None.
        leave: bool, optional
            If True, remove the progress bar, by default True

        Returns
        -------
        tuple[PipingSettings, ProbResult]
            Updated settings and results.
        """
        # Bepaal de relevante stochasten subset
        subset_stochasten = {}
        for sname, stochast in prob_input.stochasts.items():
            if sname in self.rel_stochasts.algemeen:
                subset_stochasten[sname] = stochast
            elif sname in getattr(self.rel_stochasts, z_type):
                subset_stochasten[sname] = stochast
            elif (
                sname in self.rel_stochasts.TRWD_4A
                and z_type != "sellmeijer"
                and settings.methode_stijghoogte == "TRWD_4A"
            ):
                subset_stochasten[sname] = stochast

        description = list(subset_stochasten.keys())
        marginals = list(subset_stochasten.values())
        rekentechniek = prob_input.calc_options["Rekentechniek"]

        # OpenTURNS geeft een fout in de z-functie door als RuntimeError, net als
        # een mislukte solver. Bewaar daarom de originele fout, zodat die na een
        # mislukte solver aanroep opnieuw gegeven kan worden.
        z_errors = []

        def z_func_checked(zsettings: PipingSettings) -> np.ndarray | float:
            try:
                return z_func(zsettings)
            except Exception as e:
                z_errors.append(e)
                raise

        def raise_z_error() -> None:
            if z_errors:
                raise z_errors[0] from None

        @contextmanager
        def original_z_error():
            try:
                yield
            except Exception:
                raise_z_error()
                raise

        def z_func_sample(X: np.ndarray) -> list[float]:
            zsettings = settings.copy()
            zsettings.set_params_fromdict({k: v for k, v in zip(description, X)})
            zval = z_func_checked(zsettings)
            return [zval]

        def z_func_vecsample(X: np.ndarray) -> np.ndarray:
            X = np.array(X)
            if X.shape[0] == 1:
                zval = np.array(z_func_sample(X.flatten()))
            else:
                zsettings = settings.copy()
                zsettings.set_params_fromdict(
                    {k: X[:, i] for i, k in enumerate(description)}
                )
                zval = z_func_checked(zsettings)
            return zval[:, np.newaxis]

        if copula is None:
            copula = ot.IndependentCopula(len(marginals))
        otzfunc = ot.PythonFunction(
            len(marginals), 1, z_func_sample, func_sample=z_func_vecsample
        )
        otzfunc.setInputDescription(description)
        if rekentechniek == "Monte Carlo" or rekentechniek.startswith("DS "):
            otzfunc = ot.MemoizeFunction(otzfunc)
            otzfunc.clearHistory()
        distribution = ot.JointDistribution(marginals, copula)
        distribution.setDescription(description)

        # Falen is equivalent met Z < 0
        vect = ot.RandomVector(distribution)
        G = ot.CompositeRandomVector(otzfunc, vect)
        event = ot.ThresholdEvent(G, ot.Less(), 0.0)
        event.setName("Faalkans gegeven een waterstand")

        if rekentechniek.startswith("FORM"):
            startmethod = prob_input.calc_options["FORM start"]
            with original_z_error():
                startpoint = self.get_FORM_startpoint(
                    distribution, otzfunc, method=startmethod
                )
            _, algoname = rekentechniek.split(" ")
            optimAlgo = getattr(ot, algoname)()
            self.set_calc_options(optimAlgo, prob_input.calc_options, self.debug)
            optimAlgo.setStartingPoint(startpoint)
            algo = ot.FORM(optimAlgo, event)
            if algoname == "Cobyla":
                max_iter = prob_input.calc_options["MaximumIterationNumber"]
            else:
                max_iter = prob_input.calc_options["MaximumIterationNumber"]

            with tqdm.tqdm(
                desc=rekentechniek,
                disable=(not self.progress),
                leave=leave,
                total=max_iter,
            ) as pbar:
                set_pbar = lambda _: pbar.update(1)  # noqa: E731
                optimAlgo.setProgressCallback(set_pbar)
                # Op basis van de volgende pull request:
                #  - https://github.com/openturns/openturns/pull/2561
                # Wordt vanaf openturns 1.23 een error gegeneerd op:
                #  - residual <= 1.1 * limitStateTolerance
                # Vang deze error af en los dit later op in de resultaten.
                message = None
                try:
                    algo.run()
                    result = algo.getResult()
                except Exception as e:
                    # Een fout in de z-functie is geen solver fout
                    raise_z_error()
                    if self.debug:
                        print(f"Error: {e}")
                    message = str(e)
                    try:
                        result = ot.FORMResult(
                            algo.getAnalyticalResult().getStandardSpaceDesignPoint(),
                            algo.getAnalyticalResult().getLimitStateVariable(),
                            algo.getAnalyticalResult().getIsStandardPointOriginInFailureSpace(),
                        )
                        result.setOptimizationResult(
                            algo.getAnalyticalResult().getOptimizationResult()
                        )
                    except Exception:
                        # Niets om te herstellen, geef de fout van de solver door
                        raise e from None

            prob_result = ProbResult.from_FORM_result(
                h, optimAlgo, otzfunc, result, z_type
            )
            prob_result.message = message
        elif rekentechniek == "Monte Carlo":
            experiment = ot.MonteCarloExperiment()
            algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
            self.set_calc_options(algo, prob_input.calc_options, self.debug)

            with tqdm.tqdm(
                desc=rekentechniek,
                disable=(not self.progress),
                leave=leave,
                total=int(algo.getMaximumOuterSampling() * algo.getBlockSize()),
            ) as pbar:
                set_pbar = lambda _: pbar.update(algo.getBlockSize())  # noqa: E731
                algo.setProgressCallback(set_pbar)
                with original_z_error():
                    algo.run()
            result = algo.getResult()
            prob_result = ProbResult.from_sim_result(
                h, algo, otzfunc, distribution, result, z_type
            )
        elif rekentechniek.startswith("DS"):
            _, rootname, samplingname = rekentechniek.split(" ")
            rootStrategy = getattr(ot, rootname)()
            samplingStrategy = getattr(ot, samplingname)()
            algo = ot.DirectionalSampling(event, rootStrategy, samplingStrategy)
            self.set_calc_options(algo, prob_input.calc_options, self.debug)
            with tqdm.tqdm(
                desc=rekentechniek,
                disable=(not self.progress),
                leave=leave,
                total=int(algo.getMaximumOuterSampling() * algo.getBlockSize()),
            ) as pbar:
                set_pbar = lambda _: pbar.update(algo.getBlockSize())  # noqa: E731
                algo.setProgressCallback(set_pbar)
                with original_z_error():
                    algo.run()
            result = algo.getResult()
            prob_result = ProbResult.from_sim_result(
                h, algo, otzfunc, distribution, result, z_type
            )
        else:
            raise ValueError(f"Onbekende rekentechniek '{rekentechniek}'")

        return settings, prob_result

    @staticmethod
    def set_calc_options(
        optimAlgo: ot.AbdoRackwitz
        | ot.Cobyla
        | ot.ProbabilitySimulationAlgorithm
        | ot.DirectionalSampling,
        calc_options: dict[str, float | int],
        debug: bool,
    ) -> None:
        """
        Set calculation options for the optimization algorithm.

        Parameters
        ----------
        optimAlgo : ot.AbdoRackwitz or ot.Cobyla or ot.ProbabilitySimulationAlgorithm or ot.DirectionalSampling
            The optimization algorithm instance to configure.
        calc_options : dict of str to float or int
            A dictionary containing the calculation options to set, where keys are option names and values are option values.
        debug : bool
            If True, print debug information about the options being set.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If MaximumCallsNumber or MaximumEvaluationNumber is not a whole number
            from 1 to 4294967295 and the algorithm has a limit on the number of calls.
            MaximumEvaluationNumber is the old name of MaximumCallsNumber (OpenTURNS < 1.24),
            MaximumCallsNumber wins if both are present.
        """
        if hasattr(optimAlgo, "setMaximumCallsNumber"):
            calc_options = dict(calc_options)
            limits = {}
            for name in ("MaximumEvaluationNumber", "MaximumCallsNumber"):
                val = calc_options.pop(name, None)
                if val is None:
                    continue
                if not 1 <= val <= 2**32 - 1 or val != int(val):
                    raise ValueError(
                        f"Onjuiste waarde voor {name}: {val}, "
                        "moet een geheel getal van 1 tot 4294967295 zijn"
                    )
                limits[name] = int(val)
            # MaximumCallsNumber (laatste) wint van de oude naam
            if limits:
                calc_options["MaximumCallsNumber"] = list(limits.values())[-1]
        for option_name, option_val in calc_options.items():
            attr_name = f"set{option_name}"
            if hasattr(optimAlgo, attr_name):
                getattr(optimAlgo, attr_name)(option_val)
                if debug:
                    print(f"set {option_name} to {option_val}")

    @staticmethod
    def get_FORM_startpoint(
        distribution: ot.JointDistribution,
        otzfunc: ot.PythonFunction,
        method="slice",
    ) -> list[float]:
        """
        Get the starting point for the FORM analysis.

        Parameters
        ----------
        distribution : ot.JointDistribution
            Distribution of the input variables.
        otzfunc : ot.PythonFunction
            OpenTURNS Python function representing the limit state function.
        method : str, optional
            Method to determine the starting point, by default "slice".

        Returns
        -------
        list[float]
            Starting point for the FORM analysis.
        """
        if method == "mean":
            initial_point = distribution.getMean()
        elif method == "slice":
            # Voor deze startmethode gaan we uit dat alle stochasten
            # sterkte-stochasten zijn, en dus dat we aan de rechterzijde
            # van de verdeling moeten zoeken.
            marginals = [
                distribution.getMarginal(i) for i in range(distribution.getDimension())
            ]
            initial_points, zvals = [], []
            for sp in [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                sp = [m.computeQuantile(sp)[0] for m in marginals]
                initial_points.append(sp)
                zvals.append(otzfunc(sp)[0])
            initial_point = initial_points[np.abs(zvals).argmin()]

        return initial_point
