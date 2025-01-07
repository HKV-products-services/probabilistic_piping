import unicodedata
from dataclasses import field
from typing import Self

import numpy as np
import openturns as ot
import pandas as pd
from pydantic import validate_call
from pydantic.dataclasses import dataclass


@dataclass(config={"arbitrary_types_allowed": True})
class ProbInput:
    """
    A class to represent probabilistic input data.

    Attributes
    ----------
    params : dict[str, float| int | str] | None
        Dictionary of deterministic parameters.
    stochasts : dict[str, object] | None
        Dictionary of stochastic distributions.
    charvals : dict[str, float] | None
        Dictionary of characteristic values.
    calc_options : dict[str, float | int | str] | None
        Dictionary of calculation options.
    hlist : list[float] | None
        List of water levels.
    """

    params: dict[str, float | int | str] | None = None
    stochasts: dict[str, object] | None = None
    charvals: dict[str, float] | None = None
    calc_options: dict[str, float | int | str] | None = None
    hlist: list[float] | None = None

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        Create a ProbInput instance from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the probabilistic piping input data.

        Returns
        -------
        ProbInput
            An instance of the ProbInput class.
        """
        params, dist_params, charvals, calc_options = {}, {}, {}, {}
        hlist = None
        for row in df.itertuples():
            var_type = row.Kansverdeling.lower().strip()
            var_name = row.Index
            if var_name == "h":
                # Voeg normwaterstand en lijst van waterstanden toe
                charvals[var_name] = row.Waarde
                hsteps = int((row.Max - row.Min) / row.Step) + 1
                hlist = np.linspace(row.Min, row.Max, hsteps)
            elif var_type == "rekeninstelling":
                # Sla waarde als rekeninstelling op
                if not isinstance(row.Waarde, str) and int(row.Waarde) == row.Waarde:
                    calc_options[var_name] = int(row.Waarde)
                else:
                    calc_options[var_name] = row.Waarde
            elif var_type == "geen stochast":
                # Geen stochast, deterministische waarde
                params[var_name] = row.Waarde
            else:
                # Stochast
                ProbInput.validate_stdev(
                    var_name, row.Mean, row.Spreiding, row.Spreidingstype, row.StDev
                )
                dist_params[var_name] = (
                    row.Kansverdeling,
                    row.Mean,
                    row.StDev,
                    row.Verschuiving,
                    row.Afknot_links,
                    row.Afknot_rechts,
                )
                charvals[var_name] = row.Waarde

        # Sla informatie over variabelen op.
        params = {k: v for k, v in params.items()}
        stochasts = {k: ProbInput.create_stochast(*v) for k, v in dist_params.items()}
        charvals = {k: v for k, v in charvals.items()}
        calc_options = {k: v for k, v in calc_options.items()}
        hlist = hlist.tolist()

        return cls(params, stochasts, charvals, calc_options, hlist)

    @staticmethod
    @validate_call
    def validate_stdev(
        stochast: str, mean: float, spread: float, spread_type: str, stdev: float
    ) -> None:
        """
        Validate the standard deviation of a stochastic variable.

        Parameters
        ----------
        stochast : str
            Name of the stochastic variable.
        mean : float
            Mean value of the stochastic variable.
        spread : float
            Spread value of the stochastic variable.
        spread_type : str
            Type of spread (e.g., "standaardafwijking" or "variatiecoefficient").
        stdev : float
            Standard deviation of the stochastic variable.

        Raises
        ------
        ValueError
            If the spread and standard deviation are inconsistent.
        """
        check_type = spread_type.lower().strip()
        check_type = (
            unicodedata.normalize("NFKD", check_type).encode("ascii", "ignore").decode()
        )

        # Determine check_stdev
        if check_type == "standaardafwijking":
            check_stdev = spread
        elif check_type == "variatiecoefficient":
            check_stdev = abs(mean) * spread
        else:
            raise ValueError(f"Onbekend spreidingstype {spread_type}")

        # Check stdev
        if not np.isclose(check_stdev, stdev, atol=0, rtol=1e-03):
            raise ValueError(
                f"De opgegeven spreiding en standaardafwijking van {stochast=} is inconsistent"
            )

    @staticmethod
    @validate_call
    def create_stochast(
        dist_type: str,
        mu: float,
        sigma: float,
        shift: float,
        afknot_l: float,
        afknot_r: float,
    ) -> ot.Distribution | ot.TruncatedDistribution:
        """
        Create a stochastic distribution.

        Parameters
        ----------
        dist_type : str
            Type of the distribution (e.g., "normaal", "lognormaal").
        mu : float
            Mean value of the distribution.
        sigma : float
            Standard deviation of the distribution.
        shift : float
            Shift value of the distribution.
        afknot_l : float
            Left truncation value.
        afknot_r : float
            Right truncation value.

        Returns
        -------
        ot.Distribution or ot.TruncatedDistribution
            Created distribution.

        Raises
        ------
        ValueError
            If the distribution type is unknown.
        """
        dist_type = dist_type.lower().strip()
        if dist_type == "normaal":
            dist = ot.Normal(mu, sigma)
        elif dist_type == "lognormaal":
            dist = ot.LogNormalMuSigma(mu, sigma, shift).getDistribution()
        elif dist_type == "gamma":
            dist = ot.GammaMuSigma(mu, sigma, shift).getDistribution()
        elif dist_type == "gumbel":
            dist = ot.GumbelMuSigma(mu, sigma).getDistribution()
        elif dist_type == "uniform":
            dist = ot.UniformMuSigma(mu, sigma).getDistribution()
        elif dist_type == "weibullmin":
            dist = ot.WeibullMinMuSigma(mu, sigma, shift).getDistribution()
        elif dist_type == "weibullmax":
            dist = ot.WeibullMaxMuSigma(mu, sigma, shift).getDistribution()
        elif dist_type == "arcsinus":
            dist = ot.ArcsineMuSigma(mu, sigma).getDistribution()
        else:
            raise ValueError(f"Onbekende verdeling {dist_type}")

        if not np.isnan(afknot_l) and not np.isnan(afknot_r):
            dist = ot.TruncatedDistribution(dist, afknot_l, afknot_l)
        elif not np.isnan(afknot_l):
            dist = ot.TruncatedDistribution(
                dist, afknot_l, ot.TruncatedDistribution.LOWER
            )
        elif not np.isnan(afknot_r):
            dist = ot.TruncatedDistribution(
                dist, afknot_r, ot.TruncatedDistribution.UPPER
            )

        return dist


@dataclass
class ProbResult:
    """
    A class to represent the result of a probabilistic analysis.

    Attributes
    h : float or None
        Water level.
    prob_cond : float
        Conditional probability of failure.
    converged : bool
        Indicates whether the analysis has converged.
    z_val : float or None
        Value of the limit state function.
    physical_design : dict of str to float or None
        Physical space design point.
    standard_design : dict of str to float or None
        Standard space design point.
    importancefactors : dict of str to float or None
        Importance factors.
    functionevals : int
        Number of function evaluations.
    mechanism : str or None
        Type of the limit state function.
    """

    h: float | None = None
    prob_cond: float = None
    converged: bool = False
    z_val: float | None = None
    physical_design: dict[str, float] | None = None
    standard_design: dict[str, float] | None = None
    importancefactors: dict[str, float] | None = None
    functionevals: int = 0
    mechanism: str | None = None

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_FORM_result(
        cls,
        h: float,
        optimAlgo: ot.AbdoRackwitz | ot.Cobyla,
        otzfunc: ot.Function,
        result: ot.FORMResult,
        z_type: str,
    ) -> Self:
        """
        Create a ProbResult from a FORM analysis.

        Parameters
        ----------
        h : float
            Water level.
        optimAlgo : ot.AbdoRackwitz or ot.Cobyla
            Optimization algorithm used in the FORM analysis.
        otzfunc : ot.Function
            OpenTURNS Python function representing the limit state function.
        result : ot.FORMResult
            Result of the FORM analysis.
        z_type : str
            Type of the limit state function.
        """
        description = list(otzfunc.getInputDescription())
        beta = result.getHasoferReliabilityIndex()
        up = {k: p for k, p in zip(description, result.getStandardSpaceDesignPoint())}
        pp = {k: p for k, p in zip(description, result.getPhysicalSpaceDesignPoint())}
        alphas = {k: -p / beta for k, p in up.items()}

        # Op basis van de volgende pull request:
        #  - https://github.com/openturns/openturns/pull/2561
        # Wordt vanaf openturns 1.23 een error gegeneerd op:
        #  - residual <= 1.1 * limitStateTolerance
        # Dit hebben we omzeild door een try/except toe te voegen. Maar dan
        # dan moeten we hier alsnog checken of het resultaat geconvergeerd is
        # naar het gewenste criterium. Criterium op dezelfde wijze als in
        # openturns geimplementeerd:
        #  - const Scalar residual = result_.getOptimizationResult().getConstraintError();
        #  - const Scalar limitStateTolerance = nearestPointAlgorithm_.getMaximumConstraintError();
        #  - residual <= 1.1 * limitStateTolerance
        #  - https://github.com/openturns/openturns/blob/cf34b286354ff871523b92020122a8645d6b2b34/lib/src/Uncertainty/Algorithm/Analytical/Analytical.cxx#L143
        cons_error = abs(result.getOptimizationResult().getConstraintError())
        converged = cons_error <= 1.1 * optimAlgo.getMaximumConstraintError()

        return cls(
            h,
            result.getEventProbability(),
            converged,
            otzfunc(result.getPhysicalSpaceDesignPoint())[0],
            pp,
            up,
            alphas,
            otzfunc.getEvaluationCallsNumber(),
            z_type,
        )

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_sim_result(
        cls,
        h: float,
        algo: ot.ProbabilitySimulationAlgorithm | ot.DirectionalSampling,
        otzfunc: ot.PythonFunction | ot.MemoizeFunction,
        distribution: ot.ComposedDistribution,
        result: ot.ProbabilitySimulationResult,
        z_type: str,
    ) -> Self:
        """
        Create a ProbResult from a simulation analysis.

        Parameters
        ----------
        h : float
            Water level.
        algo : ot.ProbabilitySimulationAlgorithm or ot.DirectionalSampling
            Simulation algorithm used in the analysis.
        otzfunc : ot.PythonFunction or ot.MemoizeFunction
            OpenTURNS Python function representing the limit state function.
        distribution : ot.ComposedDistribution
            Distribution of the input variables.
        result : ot.OptimizationResult
            Result of the simulation analysis.
        z_type : str
            Type of the limit state function.
        """
        description = list(otzfunc.getInputDescription())
        pf = result.getProbabilityEstimate()
        beta = ot.Normal().computeInverseSurvivalFunction(pf)[0]
        converged = (
            result.getCoefficientOfVariation()
            <= algo.getMaximumCoefficientOfVariation()
        )
        sens = ot.SimulationSensitivityAnalysis(result)
        mean_uvals = distribution.getIsoProbabilisticTransformation()(
            sens.computeMeanPointInEventDomain()
        )
        mean_uvals = np.array(mean_uvals)
        udist = np.linalg.norm(mean_uvals)
        alphas = mean_uvals / udist
        alphas = {k: a for k, a in zip(description, alphas)}
        up = [-a * beta for a in alphas.values()]
        pp = distribution.getInverseIsoProbabilisticTransformation()(up)
        up = {k: p for k, p in zip(description, up)}
        pp = {k: p for k, p in zip(description, pp)}

        return cls(
            h,
            pf,
            converged,
            otzfunc(list(pp.values()))[0],
            pp,
            up,
            alphas,
            otzfunc.getEvaluationCallsNumber(),
            z_type,
        )


@dataclass(config={"arbitrary_types_allowed": True})
class ProbResults:
    """
    A class to represent the results of probabilistic analysis.

    Attributes
    ----------
    results : list of ProbResult
        A list to store instances of ProbResult, initialized as an empty list.
    """

    results: list[ProbResult] = field(default_factory=list)
