from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from probabilistic_piping.piping_settings import PipingSettings


class PipeGrowth(NamedTuple):
    """
    Resultaat van de pipelengteontwikkeling, met een waarde per sample.

    Attributes
    ----------
    l_end : np.ndarray
        Pipelengte aan het einde van de reeks [m].
    l_time : list of np.ndarray
        Pipelengte over de tijd per sample [m], vorm n x 1.
    dldt_avg : np.ndarray
        Gemiddelde groeisnelheid tijdens groei [m/s].
    dldt_end : np.ndarray
        Groeisnelheid aan het einde van de reeks [m/s].
    t_uplift : np.ndarray
        Tijdstip van opbarsten [uur].
    t_heave : np.ndarray
        Tijdstip van heave [uur].
    t_sell : np.ndarray
        Tijdstip waarop het kritieke verval wordt overschreden [uur].
    t_first_erosion : np.ndarray
        Tijdstip van het begin van het uitspoelen van zand [uur].
    t_intervention : np.ndarray
        Exacte tijdstip van interventie [uur]: het begin van het uitspoelen
        plus `intervention_hours`, niet de eerstvolgende tijdstap.
    t_fail : np.ndarray
        Tijdstip van falen [uur].
    """

    l_end: np.ndarray
    l_time: list[np.ndarray]
    dldt_avg: np.ndarray
    dldt_end: np.ndarray
    t_uplift: np.ndarray
    t_heave: np.ndarray
    t_sell: np.ndarray
    t_first_erosion: np.ndarray
    t_intervention: np.ndarray
    t_fail: np.ndarray


class PipingEquations:
    """
    A class containing various static and class methods for performing piping calculations.
    """

    @staticmethod
    def stijghoogte(
        h_exit: float | np.ndarray, r_exit: float | np.ndarray, h: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Generieke stijghoogte berekening.

        Parameters
        ----------
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].
        r_exit : float or np.ndarray
            Dempingsfactor bij uitgang [-].
        h : float or np.ndarray
            Buitenwaterstand [m+NAP].

        Returns
        -------
        float or np.ndarray
            Berekende stijghoogte.
        """
        return h_exit + r_exit * (h - h_exit)

    @staticmethod
    def verval(
        h: float | np.ndarray,
        h_exit: float | np.ndarray,
        r_c: float | np.ndarray,
        D_cover: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Berekening van het verval.

        Parameters
        ----------
        h : float or np.ndarray
            Buitenwaterstand [m+NAP].
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].
        r_c : float or np.ndarray
            Reductiefactor [-].
        D_cover : float or np.ndarray
            Dikte deklaag [m].

        Returns
        -------
        float or np.ndarray
            Berekend verval.
        """
        return h - h_exit - r_c * D_cover

    @staticmethod
    def delta_phi_cu(settings: PipingSettings) -> float | np.ndarray:
        """
        Het kritieke stijghoogteverschil voor opbarsten.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Kritieke stijghoogteverschil.
        """
        return (
            settings.D_cover
            * (settings.gamma_sat - settings.gamma_water)
            / settings.gamma_water
        )

    @classmethod
    def phi_exit(cls, settings: PipingSettings) -> float | np.ndarray:
        """
        Stijghoogte in de watervoerende laag bij het uittredepunt.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Stijghoogte bij het uittredepunt.
        """
        if settings.methode_stijghoogte == "responsfactor":
            phi_exit = cls.stijghoogte_responsfactor(settings.h, settings)
        elif settings.methode_stijghoogte == "responsfactor_phi_gem":
            phi_exit = cls.stijghoogte_responsfactor_phi_gem(settings.h, settings)
        elif settings.methode_stijghoogte == "TRWD_4A":
            phi_exit = cls.stijghoogte_TRWD_4A(settings.h, settings)
        else:
            raise ValueError(
                f"{settings.methode_stijghoogte=}' kan niet toegepast worden in stationaire pipingberekeningen"
            )

        return phi_exit

    @staticmethod
    def F_res(settings: PipingSettings) -> float | np.ndarray:
        """
        Factor resistance.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Factor resistance.
        """
        return (
            settings.eta
            * (settings.gamma_sp / settings.gamma_water)
            * np.tan(settings.theta * np.pi / 180)
        )

    @staticmethod
    def F_scale(settings: PipingSettings) -> float | np.ndarray:
        """
        Factor scale.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Factor scale.
        """
        kappa = settings.v / settings.g * settings.k
        result = (
            settings.d_70m
            / np.cbrt(kappa * settings.L)
            * (settings.d_70 / settings.d_70m) ** 0.4
        )
        return result

    @staticmethod
    def F_geo(settings: PipingSettings) -> float | np.ndarray:
        """
        Factor geo.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Factor geo.
        """
        D, L = np.atleast_1d(settings.D), np.atleast_1d(settings.L)
        sel = np.not_equal(D, L)
        result = np.full_like(D, 0.91, dtype=np.float64)
        result[sel] = 0.91 * (D[sel] / L[sel]) ** (
            0.28 / ((D[sel] / L[sel]) ** 2.8 - 1) + 0.04
        )
        if result.shape == (1,):
            result = result[0]
        return result

    @classmethod
    def H_c(cls, settings: PipingSettings) -> float | np.ndarray:
        """
        Kritieke verval.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Kritieke verval.
        """
        Fres = cls.F_res(settings)
        Fscale = cls.F_scale(settings)
        Fgeo = cls.F_geo(settings)
        return settings.krit_verval_factor * settings.L * Fres * Fscale * Fgeo

    @classmethod
    def _Z_p(
        cls,
        m_p: float | np.ndarray,
        H_c: float | np.ndarray,
        h: float | np.ndarray,
        h_exit: float | np.ndarray,
        r_c: float | np.ndarray,
        D_cover: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Berekening van de Z_p waarde.

        Parameters
        ----------
        m_p : float or np.ndarray
            Parameter m_p.
        H_c : float or np.ndarray
            Kritieke verval.
        h : float or np.ndarray
            Buitenwaterstand [m+NAP].
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].
        r_c : float or np.ndarray
            Reductiefactor [-].
        D_cover : float or np.ndarray
            Dikte deklaag [m].

        Returns
        -------
        float or np.ndarray
            Grenstoestandsfunctie sellmeijer.
        """
        return m_p * H_c - cls.verval(h, h_exit, r_c, D_cover)

    @classmethod
    def Z_p(cls, settings: PipingSettings) -> float | np.ndarray:
        """
        Grenstoestandsfunctie sellmeijer.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Grenstoestandsfunctie sellmeijer.
        """
        Hc = cls.H_c(settings)
        result = cls._Z_p(
            settings.m_p,
            Hc,
            settings.h,
            settings.h_exit,
            settings.r_c,
            settings.D_cover,
        )
        return result

    @classmethod
    def sf_p(
        cls,
        m_p: float | np.ndarray,
        H_c: float | np.ndarray,
        h: float | np.ndarray,
        h_exit: float | np.ndarray,
        r_c: float | np.ndarray,
        D_cover: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Veiligheidsfactor sellmeijer.

        Parameters
        ----------
        m_p : float or np.ndarray
            Parameter m_p.
        H_c : float or np.ndarray
            Kritieke verval.
        h : float or np.ndarray
            Buitenwaterstand [m+NAP].
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].
        r_c : float or np.ndarray
            Reductiefactor [-].
        D_cover : float or np.ndarray
            Dikte deklaag [m].

        Returns
        -------
        float or np.ndarray
            Veiligheidsfactor sellmeijer.
        """
        return (H_c * m_p) / cls.verval(h, h_exit, r_c, D_cover)

    @staticmethod
    def sf_u(
        m_u: float | np.ndarray,
        delta_phi_cu: float | np.ndarray,
        phi_exit: float | np.ndarray,
        h_exit: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Veiligheidsfactor uplift.

        Parameters
        ----------
        m_u : float or np.ndarray
            Parameter m_u.
        delta_phi_cu : float or np.ndarray
            Kritieke stijghoogteverschil.
        phi_exit : float or np.ndarray
            Stijghoogte bij het uittredepunt.
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].

        Returns
        -------
        float or np.ndarray
            Veiligheidsfactor uplift.
        """
        return (m_u * delta_phi_cu) / (phi_exit - h_exit)

    @staticmethod
    def sf_h(
        i_ch: float | np.ndarray,
        phi_exit: float | np.ndarray,
        h_exit: float | np.ndarray,
        D_cover: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Veiligheidsfactor heave.

        Parameters
        ----------
        i_ch : float or np.ndarray
            Kritieke gradient.
        phi_exit : float or np.ndarray
            Stijghoogte bij het uittredepunt.
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].
        D_cover : float or np.ndarray
            Dikte deklaag [m].

        Returns
        -------
        float or np.ndarray
            Veiligheidsfactor heave.
        """
        return i_ch / ((phi_exit - h_exit) / D_cover)

    @staticmethod
    def _Z_u(
        m_u: float | np.ndarray,
        delta_phi_cu: float | np.ndarray,
        phi_exit: float | np.ndarray,
        h_exit: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Grenstoestandsfunctie uplift.

        Parameters
        ----------
        m_u : float or np.ndarray
            Parameter m_u.
        delta_phi_cu : float or np.ndarray
            Kritieke stijghoogteverschil.
        phi_exit : float or np.ndarray
            Stijghoogte bij het uittredepunt.
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].

        Returns
        -------
        float or np.ndarray
            Grenstoestandsfunctie uplift.
        """
        return m_u * delta_phi_cu - (phi_exit - h_exit)

    @classmethod
    def Z_u(cls, settings: PipingSettings) -> float | np.ndarray:
        """
        Grenstoestandsfunctie uplift.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Grenstoestandsfunctie uplift.
        """
        result = cls._Z_u(
            settings.m_u,
            cls.delta_phi_cu(settings),
            cls.phi_exit(settings),
            settings.h_exit,
        )
        return result

    @staticmethod
    def _Z_h(
        i_ch: float | np.ndarray,
        phi_exit: float | np.ndarray,
        h_exit: float | np.ndarray,
        D_cover: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Grenstoestandsfunctie heave.

        Parameters
        ----------
        i_ch : float or np.ndarray
            Kritieke gradient.
        phi_exit : float or np.ndarray
            Stijghoogte bij het uittredepunt.
        h_exit : float or np.ndarray
            Kwelslootpeil bij uitgang [m+NAP].
        D_cover : float or np.ndarray
            Dikte deklaag [m].

        Returns
        -------
        float or np.ndarray
            Grenstoestandsfunctie heave.
        """
        return i_ch - (phi_exit - h_exit) / D_cover

    @classmethod
    def Z_h(cls, settings: PipingSettings) -> float | np.ndarray:
        """
        Grenstoestandsfunctie heave.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Grenstoestandsfunctie heave.
        """
        result = cls._Z_h(
            settings.i_ch, cls.phi_exit(settings), settings.h_exit, settings.D_cover
        )
        return result

    @staticmethod
    def _Z_all(
        Z_u: float | np.ndarray, Z_h: float | np.ndarray, Z_p: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Grenstoestandsfunctie gecombineerd voor uplift, heave en sellmeijer.

        Parameters
        ----------
        Z_u : float or np.ndarray
            Grenstoestandsfunctie uplift.
        Z_h : float or np.ndarray
            Grenstoestandsfunctie heave.
        Z_p : float or np.ndarray
            Grenstoestandsfunctie sellmeijer.

        Returns
        -------
        float or np.ndarray
            Overkoepelende grenstoestandsfunctie.
        """
        result = np.vstack([Z_u, Z_h, Z_p]).max(axis=0)
        if np.ndim(Z_u) == 0:
            result = result[0]
        return result

    @classmethod
    def Z_all(cls, settings: PipingSettings) -> float | np.ndarray:
        """
        Overkoepelende grenstoestandsfunctie piping voor alle deelfaalmechanismen.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Overkoepelende grenstoestandsfunctie.
        """
        return cls._Z_all(cls.Z_u(settings), cls.Z_h(settings), cls.Z_p(settings))

    @classmethod
    def stijghoogte_responsfactor(
        cls, waterstand: float | np.ndarray, settings: PipingSettings
    ) -> float | np.ndarray:
        """
        Stijghoogte in de watervoerende laag bij het uittredepunt.

        Parameters
        ----------
        waterstand : float or np.ndarray
            Buitenwaterstand [m+NAP].
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Stijghoogte bij het uittredepunt.
        """
        return cls.stijghoogte(settings.h_exit, settings.r_exit, waterstand)

    @classmethod
    def stijghoogte_responsfactor_phi_gem(
        cls, waterstand: float | np.ndarray, settings: PipingSettings
    ) -> float | np.ndarray:
        """
        Stijghoogte in de watervoerende laag bij het uittredepunt, uitgaande van
        een gemiddelde stijghoogte bij een gemiddelde buitenwaterstand.

        Parameters
        ----------
        waterstand : float or np.ndarray
            Buitenwaterstand [m+NAP].
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Stijghoogte bij het uittredepunt.
        """
        return settings.phi_gem + settings.r_exit * (waterstand - settings.h_gem)

    @classmethod
    def stijghoogte_TRWD_4A(
        cls, waterstand: float | np.ndarray, settings: PipingSettings
    ) -> float | np.ndarray:
        """
        Stijghoogte in de watervoerende laag bij het uittredepunt, waarbij de
        dempingsfactor volgt uit TRWD model 4A (voorland, dijkbasis en achterland).

        Parameters
        ----------
        waterstand : float or np.ndarray
            Buitenwaterstand [m+NAP].
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Stijghoogte bij het uittredepunt.
        """
        lambda_vl = np.sqrt(settings.k * settings.D * settings.D_vl / settings.k_vl)
        lambda_al = np.sqrt(settings.k * settings.D * settings.D_al / settings.k_al)
        tanh_vl = np.tanh(settings.L_vl / lambda_vl)
        tanh_al = np.tanh(settings.L_al / lambda_al)
        r_exit = (lambda_al * tanh_al) / (
            lambda_vl * tanh_vl + settings.L_dijkbasis + lambda_al * tanh_al
        )

        return cls.stijghoogte(settings.h_exit, r_exit, waterstand)

    @classmethod
    def check_stijghoogte(
        cls, settings: PipingSettings, stochasts: set[str] = frozenset()
    ) -> None:
        """
        Controleer of de invoer van de gekozen stijghoogte methode geldig is.

        Alleen de invoer die de gekozen methode nodig heeft wordt gecontroleerd.
        Variabelen in `stochasts` worden overgeslagen, omdat deze per sample
        worden ingevuld. Bij TRWD_4A mogen de lengtes niet allemaal nul zijn.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.
        stochasts : set of str, optional
            Namen van variabelen die als stochast worden doorgerekend, by default leeg.

        Raises
        ------
        ValueError
            Als een benodigde variabele NaN is, of nul terwijl dat niet is toegestaan.
        """
        if settings.methode_stijghoogte == "responsfactor_phi_gem":
            nonzero, zero_ok = [], ["phi_gem", "h_gem", "r_exit"]
        elif settings.methode_stijghoogte == "TRWD_4A":
            nonzero = ["k", "D", "D_vl", "D_al", "k_vl", "k_al"]
            zero_ok = ["L_vl", "L_al", "L_dijkbasis"]
        else:
            return

        for name in nonzero + zero_ok:
            if name in stochasts:
                continue
            val = np.asarray(getattr(settings, name), dtype=float)
            if np.isnan(val).any():
                raise ValueError(
                    f"'{name}' is NaN, maar is nodig voor {settings.methode_stijghoogte=}"
                )
            if name in nonzero and (val == 0).any():
                raise ValueError(
                    f"'{name}' is nul, maar mag niet nul zijn voor {settings.methode_stijghoogte=}"
                )

        lengths = ["L_vl", "L_al", "L_dijkbasis"]
        if settings.methode_stijghoogte == "TRWD_4A" and not stochasts.intersection(
            lengths
        ):
            if all((np.asarray(getattr(settings, n)) == 0).all() for n in lengths):
                raise ValueError(
                    "'L_vl', 'L_al' en 'L_dijkbasis' zijn alle drie nul voor "
                    f"{settings.methode_stijghoogte=}"
                )

    @classmethod
    def check_pipegroei(
        cls, settings: PipingSettings, stochasts: set[str] = frozenset()
    ) -> None:
        """
        Controleer of de invoer van de pipegroei geldig is.

        Alleen de invoer die het gekozen groeimodel nodig heeft wordt
        gecontroleerd. Variabelen in `stochasts` worden overgeslagen, omdat
        deze per sample worden ingevuld.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.
        stochasts : set of str, optional
            Namen van variabelen die als stochast worden doorgerekend, by default leeg.

        Raises
        ------
        ValueError
            Als het groeimodel ontbreekt of onbekend is, als een benodigde
            variabele NaN is, als `n_aq` nul is bij Pol2019, als `detect_true`
            geen boolean of een stochast is, of als detectie aan staat zonder
            `intervention_hours`.
        """
        required = {
            "constant": ["dldt_constant"],
            "Pol2019": ["dldt_kezdi_C", "dldt_kezdi_mv", "n_aq"],
            "Pol2022": ["dldt_Ce"],
        }
        if settings.dldt_modeltype not in required:
            raise ValueError(
                f"'dldt_modeltype' is {settings.dldt_modeltype!r}, maar moet een van "
                f"{', '.join(required)} zijn"
            )

        for name in ["l_ini"] + required[settings.dldt_modeltype]:
            if name in stochasts:
                continue
            val = np.asarray(getattr(settings, name), dtype=float)
            if np.isnan(val).any():
                raise ValueError(
                    f"'{name}' is NaN, maar is nodig voor {settings.dldt_modeltype=}"
                )
            if name == "n_aq" and (val == 0).any():
                raise ValueError(
                    f"'{name}' is nul, maar mag niet nul zijn voor {settings.dldt_modeltype=}"
                )

        # Detectie is een instelling, geen stochast
        if "detect_true" in stochasts:
            raise ValueError("'detect_true' mag geen stochast zijn")
        detect = settings.detect_true
        if np.ndim(detect) != 0 or detect not in (0, 1):
            raise ValueError(
                f"'detect_true' is {detect!r}, maar moet True, False, 0 of 1 zijn"
            )
        settings.set_param("detect_true", bool(detect), verify_integrity=False)
        if (
            settings.detect_true
            and "intervention_hours" not in stochasts
            and np.isnan(np.asarray(settings.intervention_hours, dtype=float)).any()
        ):
            raise ValueError(
                "'intervention_hours' is NaN, maar is nodig als 'detect_true' aan staat"
            )

    @staticmethod
    def pipe_lengte_kritiek(settings: PipingSettings) -> float | np.ndarray:
        """
        Kritieke pipelengte behorend bij het kritieke verval.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Kritieke pipelengte [m].
        """
        return settings.L * 0.5 * np.tanh(2 * settings.D / settings.L)

    @staticmethod
    def inextrp1d(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        """
        Lineair inter- en extrapoleren van een 1D array.

        Parameters
        ----------
        x : np.ndarray
            Posities waarop wordt geinterpoleerd.
        xp : np.ndarray
            Posities van de bekende waarden, oplopend.
        fp : np.ndarray
            Bekende waarden op de posities xp.

        Returns
        -------
        np.ndarray
            Geinterpoleerde waarden.
        """
        # Bepaal de ondergrens
        intidx = np.minimum(np.maximum(0, np.searchsorted(xp, x) - 1), len(xp) - 2)
        # Bepaal de interpolatiefractie
        denom = xp[intidx + 1] - xp[intidx]
        # Nul wordt een zeer klein getal, zodat staptijden werken
        denom[denom == 0] = 1e-16
        fracs = (x - xp[intidx]) / denom
        # Interpoleer (1-frac) * f_low + frac * f_up
        f = (1 - fracs) * fp[intidx] + fp[intidx + 1] * fracs

        return f

    @staticmethod
    def H_equilibrium(
        Hc: float | np.ndarray,
        lc: float | np.ndarray,
        l_pipe: float | np.ndarray,
        settings: PipingSettings,
    ) -> float | np.ndarray:
        """
        Evenwichtsverval behorend bij de pipelengte.

        Parameters
        ----------
        Hc : float or np.ndarray
            Kritiek verval [m].
        lc : float or np.ndarray
            Kritieke pipelengte [m].
        l_pipe : float or np.ndarray
            Pipelengte [m].
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        float or np.ndarray
            Evenwichtsverval [m].
        """
        lc = np.atleast_1d(lc)
        l_pipe = np.atleast_1d(l_pipe)
        L = np.atleast_1d(settings.L)
        Hc = np.atleast_1d(Hc)
        m_p = np.atleast_1d(settings.m_p)
        Heq = np.empty_like(Hc, dtype=np.float64)
        for i in range(len(L)):
            Heq[i] = PipingEquations._H_equilibrium(
                m_p[i] * Hc[i], lc[i], l_pipe[i], L[i]
            )

        if np.ndim(settings.L) == 0:
            Heq = Heq[0]

        return Heq

    @staticmethod
    def _H_equilibrium(Hc: float, lc: float, l_pipe: float, L: float) -> float:
        """
        Evenwichtsverval voor een enkele waarde, inclusief modelfactor in Hc.

        Lineaire inter- en extrapolatie door (0, 0), (lc, Hc) en (L, 0.9 * Hc).
        """
        if l_pipe <= lc:
            l_low, l_up, Heq_low, Heq_up = 0.0, lc, 0.0, Hc
        else:
            l_low, l_up, Heq_low, Heq_up = lc, L, Hc, 0.9 * Hc
        denom = l_up - l_low
        if denom == 0:
            denom = 1e-16
        frac = (l_pipe - l_low) / denom
        Heq = (1 - frac) * Heq_low + Heq_up * frac

        return Heq

    @staticmethod
    def check_erosie_stationair(
        l_pipe: np.ndarray,
        h_timeseries: np.ndarray,
        stijghoogteverloop: np.ndarray,
        settings: PipingSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Controleer of een pipe kan groeien bij de maximale waterstand en stijghoogte.

        Erosie is niet mogelijk bij:
        - geen opbarsten bij maximale stijghoogte (en nog geen pipe aanwezig);
        - geen heave bij maximale stijghoogte;
        - geen pipegroei bij maximale waterstand (evenwichtsverval niet overschreden).

        Parameters
        ----------
        l_pipe : np.ndarray
            Beginlengte van de pipe per sample [m].
        h_timeseries : np.ndarray
            Waterstandsverloop [m+NAP], vorm n x 1.
        stijghoogteverloop : np.ndarray
            Stijghoogteverloop [m+NAP], vorm n x aantal samples.
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        erosion_potential : np.ndarray
            Per sample of erosie mogelijk is.
        Hc : np.ndarray
            Kritiek verval per sample [m].
        lc : np.ndarray
            Kritieke pipelengte per sample [m].
        """
        Hc = PipingEquations.H_c(settings)
        lc = PipingEquations.pipe_lengte_kritiek(settings)
        Heq = PipingEquations.H_equilibrium(Hc, lc, l_pipe, settings)
        max_stijghoogte = np.max(stijghoogteverloop, axis=0)
        max_waterstand = np.max(h_timeseries, axis=0)
        Z_u_max = PipingEquations._Z_u(
            settings.m_u,
            PipingEquations.delta_phi_cu(settings),
            max_stijghoogte,
            settings.h_exit,
        )
        Z_h_max = PipingEquations._Z_h(
            settings.i_ch, max_stijghoogte, settings.h_exit, settings.D_cover
        )
        Z_p_max = Heq - PipingEquations.verval(
            max_waterstand, settings.h_exit, settings.r_c, settings.D_cover
        )
        erosion_potential = ~(
            ((Z_u_max > 0) & (l_pipe == 0)) | (Z_h_max > 0) | (Z_p_max > 0)
        )

        return erosion_potential, Hc, lc

    @staticmethod
    def progressiesnelheid(
        H: np.ndarray, Heq: float, l_pipe: np.ndarray, settings: PipingSettings
    ) -> np.ndarray:
        """
        Berekening van de pipegroeisnelheid.

        Parameters
        ----------
        H : np.ndarray
            Verval over de kering [m].
        Heq : float
            Evenwichtsverval [m].
        l_pipe : np.ndarray
            Pipelengte [m].
        settings : PipingSettings
            Instellingen voor piping berekeningen, met `dldt_modeltype`
            "constant", "Pol2019" of "Pol2022".

        Returns
        -------
        np.ndarray
            Pipegroeisnelheid [m/s], niet negatief.

        Raises
        ------
        ValueError
            Als het model voor de pipegroeisnelheid onbekend is.
        """
        if settings.dldt_modeltype == "constant":
            dldt = settings.dldt_constant
        elif settings.dldt_modeltype == "Pol2019":
            l_pipe[l_pipe == settings.L] = l_pipe - 1e-8
            dldt = (
                settings.dldt_kezdi_mv
                * settings.dldt_kezdi_C
                * settings.k
                * H
                / (settings.L - l_pipe)
                / settings.n_aq
            )
        elif settings.dldt_modeltype == "Pol2022":  # (PhD thesis vgl 6.5)
            dldt = (
                89
                * settings.dldt_Ce
                * (settings.k * np.clip(H - Heq, 0, None) / settings.L) ** 0.81
            )
        else:
            raise ValueError("Onjuiste methode pipegroeisnelheid")

        # Als l_pipe > L kunnen negatieve snelheden ontstaan. clip deze.
        dldt = np.clip(dldt, 0, None)

        return dldt

    @staticmethod
    def unit_respons(
        x: float | np.ndarray, t: np.ndarray, settings: PipingSettings
    ) -> np.ndarray:
        """
        Eenheidsrespons van de stijghoogte op afstand x, t seconden na een stap in de waterstand.

        Nog niet beschikbaar: wordt niet aangeroepen, zie `mogelijke_erosie`.

        Parameters
        ----------
        x : float or np.ndarray
            Afstand tot het intredepunt [m].
        t : np.ndarray
            Tijd sinds de stap in de waterstand [s].
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        np.ndarray
            Eenheidsrespons [-], met de vorm van t.
        """
        # TODO: cv_al wordt hier gelezen, maar is met ticket 03 uit PipingSettings
        # gehaald. Zet cv_al terug (attribuut, invoer en controle) voor gebruik.
        # TODO: m_v = 1e-5 staat hier inline (k / 1e-5 / 9.81). Maak het een invoer.
        cv_aq = settings.k / 1e-5 / 9.81  # m2/s, benadering
        # TODO: het argument van tanh heeft de eenheid meter en mist D_al.
        term3 = np.sqrt(1 / 2 * settings.cv_al * t)
        term0 = 1 / (2 * cv_aq * t)
        term1 = settings.k_al / np.tanh(term3)
        term2 = settings.k * settings.D * np.sqrt(2 * settings.cv_al * t)
        lambda_t = 1 / np.sqrt(term0 + term1 / term2)

        return np.exp(-x / lambda_t)

    @staticmethod
    def stijghoogte_superpositie_barends(
        waterstandsverloop: np.ndarray, settings: PipingSettings
    ) -> np.ndarray:
        """
        Stijghoogteverloop door superpositie van eenheidsresponsen (Barends).

        Nog niet beschikbaar: wordt niet aangeroepen, zie `mogelijke_erosie`.

        Parameters
        ----------
        waterstandsverloop : np.ndarray
            Waterstandsverloop [m+NAP], vorm n x 1.
        settings : PipingSettings
            Instellingen voor piping berekeningen, met een tijdrooster.

        Returns
        -------
        np.ndarray
            Stijghoogteverloop [m+NAP], vorm n x 1.
        """
        # TODO: dit rekent één sample per aanroep. Maak het geschikt voor batches.
        # TODO: geheugen. De matrices R en dphi_x_t zijn beide N x N float64,
        # samen 16 * N**2 bytes bij N tijdstappen (ca. 1.6 GB bij 10.000 stappen).
        timesteps = settings.timegrid * 3600
        N = waterstandsverloop.shape[0]
        R = np.zeros(shape=(N, N))
        dphi_x_t = np.zeros(shape=(N, N))

        # TODO: x en phi0 zijn placeholders (settings.L en settings.h_exit).
        x = settings.L
        phi0 = settings.h_exit

        for i in range(1, N):
            dh = waterstandsverloop[i] - waterstandsverloop[i - 1]

            tau = timesteps - timesteps[i]
            tau[tau < 10e-20] = 10e-20
            R[i, :] = PipingEquations.unit_respons(x, tau, settings).flatten()
            dphi_x_t[i, :] = dh * R[i, :]
            dphi_x_t[i, timesteps.flatten() < timesteps[i, 0]] = 0

        phi_x_t_totaal = np.nansum(dphi_x_t, axis=0) + phi0

        return phi_x_t_totaal.reshape((N, 1))

    @staticmethod
    def mogelijke_erosie(
        h_timeseries: np.ndarray, l_pipe: np.ndarray, settings: PipingSettings
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Bepaal het stijghoogteverloop en of erosie mogelijk is.

        Parameters
        ----------
        h_timeseries : np.ndarray
            Waterstandsverloop [m+NAP], vorm n x 1.
        l_pipe : np.ndarray
            Beginlengte van de pipe per sample [m].
        settings : PipingSettings
            Instellingen voor piping berekeningen.

        Returns
        -------
        stijghoogteverloop : np.ndarray
            Stijghoogteverloop [m+NAP], vorm n x aantal samples.
        erosion_potential : np.ndarray
            Per sample of erosie mogelijk is.
        Hc : np.ndarray
            Kritiek verval per sample [m].
        lc : np.ndarray
            Kritieke pipelengte per sample [m].

        Raises
        ------
        ValueError
            Als de stijghoogte methode onjuist is.
        """
        if settings.methode_stijghoogte == "responsfactor":
            stijghoogteverloop = PipingEquations.stijghoogte_responsfactor(
                h_timeseries, settings
            )
        elif settings.methode_stijghoogte == "responsfactor_phi_gem":
            stijghoogteverloop = PipingEquations.stijghoogte_responsfactor_phi_gem(
                h_timeseries, settings
            )
        elif settings.methode_stijghoogte == "TRWD_4A":
            stijghoogteverloop = PipingEquations.stijghoogte_TRWD_4A(
                h_timeseries, settings
            )
        # TODO: voeg hier "superpositie_barends" toe met
        # stijghoogte_superpositie_barends(h_timeseries, settings), zodra die af is.
        # PipingSettings._check_methode_stijghoogte laat de naam dan weer toe.
        else:
            raise ValueError("Onjuiste methode stijghoogte")

        # Controle op mogelijke erosie
        erosion_potential, Hc, lc = PipingEquations.check_erosie_stationair(
            l_pipe, h_timeseries, stijghoogteverloop, settings
        )

        return stijghoogteverloop, erosion_potential, Hc, lc

    @staticmethod
    def z_pipelengte(settings: PipingSettings) -> float | np.ndarray:
        """
        Grenstoestand voor pipelengte: 1 - l_end / L.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen, met een waterstandsreeks.

        Returns
        -------
        float or np.ndarray
            Z-waarde, een float bij scalaire instellingen.
        """
        l_end = PipingEquations.pipelengte_ontwikkeling(settings).l_end
        z_fail = 1 - (l_end / np.atleast_1d(settings.L))
        if np.ndim(settings.L) == 0:
            z_fail = float(z_fail[0])

        return z_fail

    @staticmethod
    def pipelengte_ontwikkeling(settings: PipingSettings) -> PipeGrowth:
        """
        Berekening van de pipelengte over de tijd, gegeven een waterstandsverloop.

        Parameters
        ----------
        settings : PipingSettings
            Instellingen voor piping berekeningen, met `timegrid` en
            `h_timeseries`. Scalaire instellingen (een sample) worden als
            1-element arrays doorgerekend.

        Returns
        -------
        PipeGrowth
            Per sample: eindlengte, lengte over de tijd, gemiddelde en laatste
            groeisnelheid, en de tijdstippen van opbarsten, heave, Sellmeijer,
            eerste erosie, interventie en falen (NaN als dit niet optreedt).
        """
        if np.ndim(settings.L) == 0:
            # Een enkel sample, met scalaire instellingen
            values = settings.get_settings_as_dict()
            settings = settings.copy()
            settings.set_params_fromdict(
                {k: np.atleast_1d(v) for k, v in values.items()}
            )
        timegrid = settings.timegrid
        h_timeseries = settings.h_timeseries

        # Berekening stijghoogteverloop eenmalig
        l_pipe = settings.l_ini * settings.L
        vec_stijghoogteverloop, erosion_potential, vec_Hc, vec_lc = (
            PipingEquations.mogelijke_erosie(h_timeseries, l_pipe, settings)
        )

        # outputvariabelen
        l_time_vec = []
        l_end = l_pipe.copy()
        dldt_avg = np.zeros_like(l_pipe, dtype=np.float64)
        dldt_end = np.zeros_like(l_pipe, dtype=np.float64)
        t_uplift = np.full_like(l_pipe, np.nan, dtype=np.float64)
        t_heave = np.full_like(l_pipe, np.nan, dtype=np.float64)
        t_sell = np.full_like(l_pipe, np.nan, dtype=np.float64)
        t_ff = np.full_like(l_pipe, np.nan, dtype=np.float64)
        t_fail = np.full_like(l_pipe, np.nan, dtype=np.float64)
        firsttime_U_H_Heq = np.full_like(l_pipe, np.nan, dtype=np.float64)
        for i in range(len(l_pipe)):
            settingsi = settings.get_partial_settings(i, verify_integrity=False)
            Hc = vec_Hc[[i]]
            lc = vec_lc[[i]]
            stijghoogteverloop = vec_stijghoogteverloop[:, [i]]

            # Tijdsafhankelijke som als erosie kan optreden
            if erosion_potential[i]:
                # Tijdstappen
                dt = np.diff(timegrid, axis=0)

                # rekenvariabelen
                dldt = np.zeros_like(timegrid, dtype=np.float64)
                l_time = np.full_like(timegrid, l_pipe[i], dtype=np.float64)

                Z_uplift = PipingEquations._Z_u(
                    settings.m_u[i],
                    PipingEquations.delta_phi_cu(settingsi),
                    stijghoogteverloop,
                    settings.h_exit[i],
                )
                Z_heave = PipingEquations._Z_h(
                    settings.i_ch[i],
                    stijghoogteverloop,
                    settings.h_exit[i],
                    settings.D_cover[i],
                )
                Z_sell = PipingEquations._Z_p(
                    settings.m_p[i],
                    Hc,
                    h_timeseries,
                    settings.h_exit[i],
                    settings.r_c[i],
                    settings.D_cover[i],
                )

                H = PipingEquations.verval(
                    h_timeseries,
                    settings.h_exit[i],
                    settings.r_c[i],
                    settings.D_cover[i],
                )
                H[H < 0] = 0.0

                # Uplift
                if settings.l_ini[i] > 1e-6:
                    t_uplift[i] = timegrid[0, 0]  # first
                elif np.min(Z_uplift) < 0:
                    t_uplift[i] = np.min(timegrid[Z_uplift < 0])

                # Pipelengte en faaldefinitie
                intervention_active = False
                intervention_time = np.nan
                time_uplift = np.nan
                H_cummax = np.maximum.accumulate(
                    H[:, 0]
                )  # H_cummax[t] == np.max(H[: t + 1])
                for t in range(1, timegrid.shape[0]):
                    # Evenwichtscurve/verval
                    Heq = PipingEquations._H_equilibrium(
                        settings.m_p[i] * Hc[0], lc[0], l_time[t - 1, 0], settings.L[i]
                    )

                    # uplift
                    if math.isnan(time_uplift):
                        if settings.l_ini[i] > 0.000001:
                            time_uplift = timegrid[0, 0]  # first
                        elif Z_uplift[t, 0] < 0:
                            time_uplift = timegrid[t, 0]

                    # uitspoelen zand
                    if (
                        (not math.isnan(time_uplift))
                        and (Z_heave[t, 0] < 0)
                        and (H[t, 0] >= Heq)
                        and math.isnan(firsttime_U_H_Heq[i])
                    ):
                        firsttime_U_H_Heq[i] = timegrid[t, 0]

                    # Bepaal of interventie heeft plaatsgevonden
                    if (
                        (not math.isnan(firsttime_U_H_Heq[i]))
                        and settings.detect_true
                        and timegrid[t, 0]
                        > (firsttime_U_H_Heq[i] + settings.intervention_hours[i])
                    ):
                        if not intervention_active:  # first time active
                            intervention_time = (
                                firsttime_U_H_Heq[i] + settings.intervention_hours[i]
                            )
                            # Groei over het deel van de stap voor de interventie
                            frac = (intervention_time - timegrid[t - 1, 0]) / dt[
                                t - 1, 0
                            ]
                            if (
                                frac > 0
                                and (Z_heave[t, 0] < 0)
                                and (l_time[t - 1, 0] < settings.L[i])
                            ):
                                dldt[t] = frac * PipingEquations.progressiesnelheid(
                                    H[t], Heq, l_time[t - 1], settingsi
                                )
                        intervention_active = True

                    # Voorwaarde voor pipegroei
                    erosion_condition = (
                        (timegrid[t, 0] >= firsttime_U_H_Heq[i])
                        and (Z_heave[t, 0] < 0)
                        and (l_time[t - 1, 0] < settings.L[i])
                        and not intervention_active
                    )
                    if erosion_condition:
                        dldt[t] = PipingEquations.progressiesnelheid(
                            H[t], Heq, l_time[t - 1], settingsi
                        )

                    l_time[t, 0] = l_time[t - 1, 0] + dldt[t, 0] * (dt[t - 1, 0] * 3600)

                    # correctie om te voorkomen dat pipe in 1 tijdstap doorschiet terwijl Hc niet wordt overschreden
                    if (l_time[t, 0] > lc[0]) & (
                        H_cummax[t] < (settings.m_p[i] * Hc[0])
                    ):
                        l_time[t, 0] = np.max([lc[0], l_pipe[i]])

                Z_fail = settingsi.L - l_time
                l_end[i] = l_time[-1, 0]

                if (dldt > 0).any():
                    dldt_avg[i] = np.mean(dldt[dldt > 0])
                dldt_end[i] = dldt[-1, 0]

                l_time_vec.append(l_time.copy())

                t_heave[i] = min(timegrid[Z_heave < 0], default=np.nan)
                t_sell[i] = min(timegrid[Z_sell < 0], default=np.nan)
                t_fail[i] = min(timegrid[Z_fail < 0], default=np.nan)
                t_ff[i] = intervention_time
            else:
                l_time_vec.append(np.array([l_pipe[i]]))

        return PipeGrowth(
            l_end,
            l_time_vec,
            dldt_avg,
            dldt_end,
            t_uplift,
            t_heave,
            t_sell,
            firsttime_U_H_Heq,
            t_ff,
            t_fail,
        )
