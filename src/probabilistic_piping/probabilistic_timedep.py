import numpy as np
import openturns as ot
from pydantic import validate_call

from probabilistic_piping.piping_equations import PipingEquations
from probabilistic_piping.piping_settings import PipingSettings
from probabilistic_piping.probabilistic_base import ProbPipingBase
from probabilistic_piping.probabilistic_io import ProbInput, ProbResult


class ProbPipingTimeDepWaterlevel(ProbPipingBase):
    """
    Class for probabilistic piping calculations with a timedep water level.
    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def timedep_waterlevel_failureprobability(
        self,
        prob_input: ProbInput,
        timegrid: np.ndarray,
        h_timeseries: np.ndarray,
        dt: float | None = None,
        settings: PipingSettings | None = None,
        copula: ot.Distribution | None = None,
        leave: bool = True,
    ) -> tuple[PipingSettings, ProbResult]:
        """
        Calculate the failure probability for a timedep water level.

        Parameters
        ----------
        prob_input : ProbInput
            Probabilistic input settings.
        timegrid : np.ndarray
            Times of the water level series [h].
        h_timeseries : np.ndarray
            Water level series [m+NAP], as long as timegrid.
        dt : float or None, optional
            Time step (in hours) for interpolation of the series, by default None.
        settings : PipingSettings or None, optional
            Piping settings, by default None.
        copula : ot.Distribution or None, optional
            Copula distribution, by default None.
        leave: bool, optional
            If True, remove the progress bar, by default True

        Returns
        -------
        tuple
            Updated settings and results of the failure probability calculations.
        """
        # Maak settings object aan.
        settings_dict = {}
        if settings is None:
            settings = PipingSettings()
            settings_dict = {**prob_input.params}
        settings.set_params_fromdict(settings_dict)
        self.piping_eq.check_stijghoogte(settings, set(prob_input.stochasts))

        # Zet de waterstandsreeks. De piek is de waterstand voor de resultaten.
        settings.set_waterstand_tijdreeks(timegrid, h_timeseries, dt=dt)
        h = float(np.max(settings.h_timeseries))

        # Bepaal faalkans
        settings, results = self.prob_calculation(
            h,
            "timedep_piping",
            PipingEquations.z_pipelengte,
            settings,
            prob_input,
            copula=copula,
            leave=leave,
        )

        return settings, results
