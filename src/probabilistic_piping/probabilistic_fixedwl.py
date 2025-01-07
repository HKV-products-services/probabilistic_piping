import statistics as stat
from dataclasses import field

import numpy as np
import openturns as ot
import pandas as pd
import tqdm.auto as tqdm
from pydantic import validate_call, ConfigDict

from probabilistic_piping.piping_equations import PipingEquations
from probabilistic_piping.piping_settings import PipingSettings
from probabilistic_piping.probabilistic_base import ProbPipingBase, RelevantStochasts
from probabilistic_piping.probabilistic_io import ProbInput, ProbResult, ProbResults


class ProbPipingFixedWaterlevelBase(ProbPipingBase):
    """
    Base class for probabilistic piping calculations with a fixed water level.

    Attributes
    ----------
    model_config : ConfigDict
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

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fixed_waterlevel_semiprob(
        self,
        prob_input: ProbInput,
        settings: PipingSettings | None = None,
        h: float | None = None,
    ) -> pd.DataFrame:
        """
        Perform semi-probabilistic calculations for a constant water level.

        Parameters
        ----------
        prob_input : ProbInput
            Probabilistic input settings.
        settings : PipingSettings or None, optional
            Piping settings, by default None.
        h : float or None, optional
            Water level, by default None.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the calculations.
        """
        # Standaard normale verdeling
        norm = stat.NormalDist(mu=0, sigma=1)

        # Maak settings object aan.
        settings_dict = {}
        if settings is None:
            settings = PipingSettings()
            settings_dict = {**prob_input.params, **prob_input.charvals}

        # Update buitenwaterstand en eventuele default settings
        if h is not None:
            settings_dict["h"] = h
        settings.set_params_fromdict(settings_dict)

        # bereken veiligheidsfactoren voor de deelmechanismes
        Hc = self.piping_eq.H_c(settings)
        phi_exit = self.piping_eq.phi_exit(settings)
        delta_phi_cu = self.piping_eq.delta_phi_cu(settings)
        sf_h = self.piping_eq.sf_h(
            settings.i_ch, phi_exit, settings.h_exit, settings.D_cover
        )
        sf_u = self.piping_eq.sf_u(
            settings.m_u, delta_phi_cu, phi_exit, settings.h_exit
        )
        sf_p = self.piping_eq.sf_p(
            settings.m_p,
            Hc,
            settings.h,
            settings.h_exit,
            settings.r_c,
            settings.D_cover,
        )

        # Beta en faalkansen
        inv_t = np.array([norm.inv_cdf(1 / t) for t in np.atleast_1d(settings.t_norm)])
        beta_h = (np.log(sf_h / 0.37) + 0.30 * -inv_t) / 0.48
        beta_u = (np.log(sf_u / 0.48) + 0.27 * -inv_t) / 0.46
        beta_p = (np.log(sf_p / 1.04) + 0.43 * -inv_t) / 0.37
        pf_u = np.array([norm.cdf(-b) for b in np.atleast_1d(beta_u)])
        pf_h = np.array([norm.cdf(-b) for b in np.atleast_1d(beta_h)])
        pf_p = np.array([norm.cdf(-b) for b in np.atleast_1d(beta_p)])
        if np.ndim(settings.t_norm) == 0:
            beta_h = beta_h[0]
            beta_u = beta_u[0]
            beta_p = beta_p[0]
            pf_u = pf_u[0]
            pf_h = pf_h[0]
            pf_p = pf_p[0]

        # Bepaal gecombineerde faalkans
        pf_combi = np.vstack([pf_u, pf_h, pf_p]).min(axis=0)
        if np.ndim(pf_u) == 0:
            pf_combi = pf_combi[0]

        # Maak een pandas object aan met de waardes voor de relevante parameters
        result = {}
        for k in prob_input.charvals:
            result[k] = getattr(settings, k)
        for k in prob_input.params:
            result[k] = getattr(settings, k)
        result["Hc"] = Hc
        result["phi_exit"] = phi_exit
        result["delta_phi_cu"] = delta_phi_cu
        result["sf_h"] = sf_h
        result["sf_u"] = sf_u
        result["sf_p"] = sf_p
        result["beta_h"] = beta_h
        result["beta_u"] = beta_u
        result["beta_p"] = beta_p
        result["pf_u"] = pf_u
        result["pf_h"] = pf_h
        result["pf_p"] = pf_p
        result["pf_combi"] = pf_combi
        result = pd.Series(result)

        return result


class ProbPipingFixedWaterlevelSimple(ProbPipingFixedWaterlevelBase):
    """
    Class for simple probabilistic piping calculations with a fixed water level.

    Attributes
    ----------
    model_config : ConfigDict
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

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fixed_waterlevel_fragilitycurve(
        self,
        prob_input: ProbInput,
        hlist: list[float] | None = None,
        settings: PipingSettings | None = None,
        copula: ot.Distribution | None = None,
    ) -> ProbResults:
        """
        Calculate the fragility curve for a fixed water level.

        Parameters
        ----------
        prob_input : ProbInput
            Probabilistic input settings.
        hlist : list of float or None, optional
            List of water levels, by default None.
        settings : PipingSettings or None, optional
            Piping settings, by default None.
        copula : ot.Distribution or None, optional
            Copula distribution, by default None.

        Returns
        -------
        ProbResults
            Results of the fragility curve calculations.
        """
        results_u = ProbResults()
        results_h = ProbResults()
        results_p = ProbResults()
        results_c = ProbResults()

        if hlist is None:
            hlist = prob_input.hlist.copy()

        if settings is None:
            settings = PipingSettings()
            settings.set_params_fromdict({**prob_input.params})

        # loop over waterstanden en bepaal faalkansen
        for h in tqdm.tqdm(hlist, disable=(not self.progress)):
            _, ru, rh, rp, rc = self.fixed_waterlevel_failureprobability(
                prob_input=prob_input,
                h=h,
                settings=settings,
                copula=copula,
                leave=False,
            )
            results_u.results.append(ru)
            results_h.results.append(rh)
            results_p.results.append(rp)
            results_c.results.append(rc)

        # Set waterstand naar NaN voor het settings object, omdat de correcte
        # lijst met waterstanden in het results object zit
        settings.set_param("h", np.nan)

        return settings, results_u, results_h, results_p, results_c

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fixed_waterlevel_failureprobability(
        self,
        prob_input: ProbInput,
        h: float | None = None,
        settings: PipingSettings | None = None,
        copula: ot.Distribution | None = None,
        leave: bool = True,
    ):
        """
        Calculate the failure probability for a fixed water level.

        Parameters
        ----------
        prob_input : ProbInput
            Probabilistic input settings.
        h : float or None, optional
            Water level, by default None.
        settings : PipingSettings or None, optional
            Piping settings, by default None.
        copula : ot.Distribution or None, optional
            Copula distribution, by default None.
        leave: bool, optional
            If True, remove the progress bar, by default True

        Returns
        -------
        tuple
            Updated settings and results for uplift, heave, Sellmeijer, and combined mechanisms.
        """
        if h is None:
            h = prob_input.charvals["h"]

        # Maak settings object aan.
        settings_dict = {}
        if settings is None:
            settings = PipingSettings()
            settings_dict = {**prob_input.params}
        settings_dict["h"] = h
        settings.set_params_fromdict(settings_dict)

        _, result_u = self.prob_calculation(
            h,
            "uplift",
            self.piping_eq.Z_u,
            settings,
            prob_input,
            copula=copula,
            leave=leave,
        )
        _, result_h = self.prob_calculation(
            h,
            "heave",
            self.piping_eq.Z_h,
            settings,
            prob_input,
            copula=copula,
            leave=leave,
        )
        _, result_p = self.prob_calculation(
            h,
            "sellmeijer",
            self.piping_eq.Z_p,
            settings,
            prob_input,
            copula=copula,
            leave=leave,
        )

        # Bepaal gecombineerd resultaat op basis van minimale faalkans
        list_results = [result_u, result_h, result_p]
        idx_min = np.argmin([r.prob_cond for r in list_results])
        result_min = list_results[idx_min]

        return settings, result_u, result_h, result_p, result_min


class ProbPipingFixedWaterlevel(ProbPipingFixedWaterlevelBase):
    """
    Class for probabilistic piping calculations with a fixed water level.

    Attributes
    ----------
    model_config : ConfigDict
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

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fixed_waterlevel_fragilitycurve(
        self,
        prob_input: ProbInput,
        hlist: list[float] | None = None,
        settings: PipingSettings | None = None,
        z_type: str = "sellmeijer",
        copula: ot.Distribution | None = None,
    ) -> tuple[PipingSettings, ProbResults]:
        """
        Calculate the fragility curve for a fixed water level.

        Parameters
        ----------
        prob_input : ProbInput
            Probabilistic input settings.
        hlist : list of float or None, optional
            List of water levels, by default None.
        settings : PipingSettings or None, optional
            Piping settings, by default None.
        z_type : str, optional
            Type of the limit state function, by default "sellmeijer".
        copula : ot.Distribution or None, optional
            Copula distribution, by default None.

        Returns
        -------
        tuple
            Updated settings and results of the fragility curve calculations.
        """
        if hlist is None:
            hlist = prob_input.hlist.copy()

        if settings is None:
            settings = PipingSettings()
            settings.set_params_fromdict({**prob_input.params})

        # loop over waterstanden en bepaal faalkansen
        results = ProbResults()
        for h in tqdm.tqdm(hlist, disable=(not self.progress)):
            _, result = self.fixed_waterlevel_failureprobability(
                prob_input=prob_input,
                h=h,
                settings=settings,
                z_type=z_type,
                copula=copula,
                leave=False,
            )
            results.results.append(result)

        # Set waterstand naar NaN voor het settings object, omdat de correcte
        # lijst met waterstanden in het results object zit
        settings.set_param("h", np.nan)

        return settings, results

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fixed_waterlevel_failureprobability(
        self,
        prob_input: ProbInput,
        h: float | None = None,
        settings: PipingSettings | None = None,
        z_type: str = "sellmeijer",
        copula: ot.Distribution | None = None,
        leave: bool = True,
    ) -> tuple[PipingSettings, ProbResult]:
        """
        Calculate the failure probability for a fixed water level.

        Parameters
        ----------
        prob_input : ProbInput
            Probabilistic input settings.
        h : float or None, optional
            Water level, by default None.
        settings : PipingSettings or None, optional
            Piping settings, by default None.
        z_type : str, optional
            Type of the limit state function, by default "sellmeijer".
        copula : ot.Distribution or None, optional
            Copula distribution, by default None.
        leave: bool, optional
            If True, remove the progress bar, by default True

        Returns
        -------
        tuple
            Updated settings and results of the failure probability calculations.
        """
        if h is None:
            h = prob_input.charvals["h"]

        # Maak settings object aan.
        settings_dict = {}
        if settings is None:
            settings = PipingSettings()
            settings_dict = {**prob_input.params}
        settings_dict["h"] = h
        settings.set_params_fromdict(settings_dict)

        # z-functie
        if z_type == "sellmeijer":
            z_func = self.piping_eq.Z_p
        elif z_type == "heave":
            z_func = self.piping_eq.Z_h
        elif z_type == "uplift":
            z_func = self.piping_eq.Z_u
        elif z_type == "combi":
            z_func = self.piping_eq.Z_all
        else:
            raise ValueError(
                f"Argument z_type is niet 'sellmeijer', 'heave', 'uplift' of 'combi' ('{z_type}')"
            )

        # Bepaal faalkans
        settings, results = self.prob_calculation(
            h, z_type, z_func, settings, prob_input, copula=copula, leave=leave
        )

        return settings, results
