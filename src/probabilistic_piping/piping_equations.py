from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from probabilistic_piping.piping_settings import PipingSettings


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
    def unit_respons(
        x: float | np.ndarray, t: np.ndarray, settings: PipingSettings
    ) -> np.ndarray:
        """
        Eenheidsrespons van de stijghoogte op afstand x, t seconden na een stap in de waterstand.

        Nog niet beschikbaar: wordt niet aangeroepen, Barends is geblokkeerd in PipingSettings.

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

        Nog niet beschikbaar: wordt niet aangeroepen, Barends is geblokkeerd in PipingSettings.

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


@njit(cache=True, fastmath=False, error_model="numpy")
def _H_equilibrium(Hc: float, lc: float, l_pipe: float, L: float) -> float:
    """Evenwichtsverval voor één sample, inclusief modelfactor in Hc.

    Lineaire inter- en extrapolatie door (0, 0), (lc, Hc) en (L, 0.9 * Hc).

    Parameters
    ----------
    Hc : float
        Kritiek verval [m], inclusief de modelfactor piping.
    lc : float
        Kritieke pipelengte [m], waar de evenwichtscurve haar maximum bereikt.
    l_pipe : float
        Huidige pipelengte [m].
    L : float
        Kwelweglengte [m].

    Returns
    -------
    float
        Evenwichtsverval bij de huidige pipelengte [m].

    Notes
    -----
    De modelfactor is al verwerkt in Hc en wordt hier niet opnieuw toegepast.
    Een interval met lengte nul krijgt een noemer van 1e-16, zoals in de
    oorspronkelijke interpolatie.
    """
    if l_pipe <= lc:
        l_low, l_up, Heq_low, Heq_up = 0.0, lc, 0.0, Hc
    else:
        l_low, l_up, Heq_low, Heq_up = lc, L, Hc, 0.9 * Hc
    denominator = l_up - l_low
    if denominator == 0:
        denominator = 1e-16
    fraction = (l_pipe - l_low) / denominator
    return (1 - fraction) * Heq_low + Heq_up * fraction


@njit(cache=True, fastmath=False, error_model="numpy")
def progression_rate(
    H: float,
    Heq: float,
    l_pipe: float,
    L: float,
    modeltype: str,
    k: float,
    n_aq: float,
    dldt_constant: float,
    dldt_Ce: float,
    dldt_kezdi_mv: float,
    dldt_kezdi_C: float,
) -> float:
    """Pipegroeisnelheid voor één sample en één tijdstip, in m/s.

    Parameters
    ----------
    H : float
        Verval over de kering [m].
    Heq : float
        Evenwichtsverval bij de huidige pipelengte [m].
    l_pipe : float
        Huidige pipelengte [m]. De integratie roept deze functie alleen aan
        zolang de pipe korter is dan L.
    L : float
        Kwelweglengte [m].
    modeltype : str
        Groeimodel: "constant", "Pol2019" of "Pol2022".
    k : float
        Doorlatendheid van de watervoerende laag [m/s].
    n_aq : float
        Porositeit van de aquifer [-], ongelijk aan nul bij "Pol2019".
    dldt_constant : float
        Constante pipegroeisnelheid bij "constant" [m/s].
    dldt_Ce : float
        Groeicoëfficiënt Ce bij "Pol2022" [-].
    dldt_kezdi_mv : float
        Constante mv in de formule van Kezdi bij "Pol2019" [-].
    dldt_kezdi_C : float
        Constante C in de formule van Kezdi bij "Pol2019" [-].

    Returns
    -------
    float
        Niet-negatieve pipegroeisnelheid [m/s].

    Raises
    ------
    ValueError
        Als het model voor de pipegroeisnelheid onbekend is.

    Notes
    -----
    "constant" gebruikt dldt_constant. "Pol2019" gebruikt
    mv * C * k * H / ((L - l_pipe) * n_aq). "Pol2022" gebruikt
    89 * Ce * (k * max(H - Heq, 0) / L) ** 0.81.

    "Pol2022" volgt Pol (2022), vergelijking 6.5, en dezelfde formule in
    Pol et al. (2024). Zie `pipe_growth` voor de volledige modelreferenties.
    Parameters van niet-gekozen groeimodellen worden niet gebruikt.
    """
    if modeltype == "constant":
        rate = dldt_constant
    elif modeltype == "Pol2019":
        rate = dldt_kezdi_mv * dldt_kezdi_C * k * H / (L - l_pipe) / n_aq
    elif modeltype == "Pol2022":
        difference = np.maximum(H - Heq, 0.0)
        rate = 89 * dldt_Ce * (k * difference / L) ** 0.81
    else:
        raise ValueError("Onjuiste methode pipegroeisnelheid")

    # Als l_pipe > L kunnen negatieve snelheden ontstaan. Kap deze af.
    return np.maximum(rate, 0.0)


@njit(cache=True, fastmath=False, error_model="numpy")
def erosion_potential(
    max_phi: np.ndarray,
    max_waterlevel: float,
    l_initial: np.ndarray,
    L: np.ndarray,
    lc: np.ndarray,
    Hc: np.ndarray,
    m_u: np.ndarray,
    delta_phi: np.ndarray,
    h_exit: np.ndarray,
    i_ch: np.ndarray,
    D_cover: np.ndarray,
    r_c: np.ndarray,
) -> np.ndarray:
    """Controleer of een pipe kan groeien bij de maximale belasting.

    Erosie is niet mogelijk bij:
    - geen opbarsten bij maximale stijghoogte en geen aanwezige beginpipe;
    - geen heave bij maximale stijghoogte;
    - geen pipegroei bij maximale waterstand.

    Parameters
    ----------
    max_phi : np.ndarray
        Maximale stijghoogte per sample [m+NAP], vorm (samples,).
    max_waterlevel : float
        Maximale buitenwaterstand van de gedeelde waterstandsreeks [m+NAP].
    l_initial : np.ndarray
        Beginlengte van de pipe per sample [m], vorm (samples,).
    L : np.ndarray
        Kwelweglengte per sample [m], vorm (samples,).
    lc : np.ndarray
        Kritieke pipelengte per sample [m], vorm (samples,).
    Hc : np.ndarray
        Kritiek verval per sample [m], inclusief de modelfactor piping,
        vorm (samples,).
    m_u : np.ndarray
        Modelfactor opbarsten per sample [-], vorm (samples,).
    delta_phi : np.ndarray
        Kritiek stijghoogteverschil voor opbarsten per sample [m], zonder
        modelfactor opbarsten, vorm (samples,).
    h_exit : np.ndarray
        Kwelslootpeil bij het uittredepunt per sample [m+NAP], vorm (samples,).
    i_ch : np.ndarray
        Kritieke heavegradiënt per sample [-], vorm (samples,).
    D_cover : np.ndarray
        Dikte van de deklaag per sample [m], vorm (samples,).
    r_c : np.ndarray
        Reductiefactor voor het verval per sample [-], vorm (samples,).

    Returns
    -------
    np.ndarray
        Booleanmasker met vorm (samples,): True als erosie mogelijk is.

    Notes
    -----
    Dit is een voorselectie op basis van de maxima, die niet gelijktijdig
    hoeven op te treden. De integratie controleert de voorwaarden per tijdstap.
    De voorselectie sluit alleen strikt positieve Z-waarden uit; een waarde
    gelijk aan nul wordt hier niet als veilig uitgesloten.
    """
    can_erode = np.empty(len(L), dtype=np.bool_)
    for i in range(len(L)):
        Heq = _H_equilibrium(Hc[i], lc[i], l_initial[i], L[i])
        Z_u = m_u[i] * delta_phi[i] - (max_phi[i] - h_exit[i])
        Z_h = i_ch[i] - (max_phi[i] - h_exit[i]) / D_cover[i]
        Z_p = Heq - (max_waterlevel - h_exit[i] - r_c[i] * D_cover[i])
        can_erode[i] = not ((Z_u > 0 and l_initial[i] == 0) or Z_h > 0 or Z_p > 0)
    return can_erode


@njit(cache=True, fastmath=False, error_model="numpy")
def integrate_pipe_growth(
    timegrid: np.ndarray,
    H: np.ndarray,
    H_cummax: np.ndarray,
    Z_uplift: np.ndarray,
    Z_heave: np.ndarray,
    L: np.ndarray,
    lc: np.ndarray,
    Hc: np.ndarray,
    l_initial: np.ndarray,
    has_initial_pipe: np.ndarray,
    can_erode: np.ndarray,
    modeltype: str,
    k: np.ndarray,
    n_aq: np.ndarray,
    dldt_constant: np.ndarray,
    dldt_Ce: np.ndarray,
    dldt_kezdi_mv: np.ndarray,
    dldt_kezdi_C: np.ndarray,
    detect: bool,
    intervention_hours: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bereken de pipelengte en groeisnelheid over de tijd voor alle samples.

    Parameters
    ----------
    timegrid : np.ndarray
        Strikt toenemende tijdstippen [uur], vorm (tijdstippen,). Eén reeks
        voor alle samples; tijdstappen hoeven niet even groot te zijn.
    H : np.ndarray
        Niet-negatief verval over de kering [m], vorm (tijdstippen, samples).
    H_cummax : np.ndarray
        Cumulatief maximum van H tot en met de huidige tijdstap [m],
        vorm (tijdstippen, samples).
    Z_uplift : np.ndarray
        Grenstoestandsfunctie opbarsten [m], vorm (tijdstippen, samples).
        Opbarsten treedt op bij een strikt negatieve waarde.
    Z_heave : np.ndarray
        Grenstoestandsfunctie heave [-], vorm (tijdstippen, samples).
        Heave treedt op bij een strikt negatieve waarde.
    L : np.ndarray
        Kwelweglengte per sample [m], vorm (samples,).
    lc : np.ndarray
        Kritieke pipelengte per sample [m], vorm (samples,).
    Hc : np.ndarray
        Kritiek verval per sample [m], inclusief de modelfactor piping,
        vorm (samples,).
    l_initial : np.ndarray
        Beginlengte van de pipe per sample [m], vorm (samples,).
    has_initial_pipe : np.ndarray
        Booleanmasker voor een aanwezige beginpipe, vorm (samples,).
        Deze samples gelden vanaf het eerste tijdstip als opgebarsten.
    can_erode : np.ndarray
        Booleanmasker van de erosievoorselectie, vorm (samples,).
        Samples met False behouden op ieder tijdstip hun beginlengte.
    modeltype : str
        Groeimodel voor alle samples: "constant", "Pol2019" of "Pol2022".
    k : np.ndarray
        Doorlatendheid van de watervoerende laag per sample [m/s],
        vorm (samples,).
    n_aq : np.ndarray
        Porositeit per sample [-], ongelijk aan nul bij "Pol2019",
        vorm (samples,).
    dldt_constant : np.ndarray
        Constante pipegroeisnelheid per sample bij "constant" [m/s],
        vorm (samples,).
    dldt_Ce : np.ndarray
        Groeicoëfficiënt Ce per sample bij "Pol2022" [-], vorm (samples,).
    dldt_kezdi_mv : np.ndarray
        Constante mv per sample bij "Pol2019" [-], vorm (samples,).
    dldt_kezdi_C : np.ndarray
        Constante C per sample bij "Pol2019" [-], vorm (samples,).
    detect : bool
        Of detectie en interventie worden meegenomen; geldt voor alle samples.
    intervention_hours : np.ndarray
        Tijd tussen het eerste uitspoelen en de interventie per sample [uur],
        vorm (samples,). Alleen gebruikt als detect True is.

    Returns
    -------
    l_time : np.ndarray
        Pipelengte [m], vorm (tijdstippen, samples), inclusief de beginlengte.
    dldt : np.ndarray
        Toegepaste groeisnelheid [m/s], vorm (tijdstippen, samples).
        De eerste rij is nul. Bij interventie bevat de betreffende stap de
        snelheid vermenigvuldigd met de fractie vóór het interventietijdstip.
    first_erosion : np.ndarray
        Eerste tijdstip van uitspoelen per sample [uur], vorm (samples,).
        NaN als uitspoelen niet optreedt.
    intervention_time : np.ndarray
        Exact interventietijdstip per sample [uur], vorm (samples,).
        NaN als geen interventie wordt verwerkt binnen de reeks.

    Notes
    -----
    De invoer is voorbereid en gecontroleerd door `pipe_growth`. Numerieke
    arrays zijn één- of tweedimensionaal zoals hierboven beschreven; maskers
    bevatten booleans. De integratie werkt de pipelengte vanaf de tweede rij
    bij en rekent de tijdstap van uren om naar seconden.

    Het eerste uitspoelen vereist opbarsten, heave en H >= Heq. Een eenmaal
    opgetreden opbarsting blijft geldig. De pipe groeit niet verder zodra de
    vorige lengte L heeft bereikt; de laatste groeistap kan L overschrijden.
    Zolang het cumulatieve verval onder Hc blijft, wordt de lengte begrensd
    op max(lc, l_initial).

    De interventieklok start bij het eerste uitspoelen. Interventie wordt pas
    verwerkt in een stap strikt na het geplande interventietijdstip, met een
    deelstap tot aan dat exacte tijdstip. Daarna stopt de groei permanent.
    Een interventie exact op het laatste tijdstip wordt dus niet geregistreerd.

    Groeiformules en evenwichtsinterpolatie staan in gedeelde scalaire helpers.
    Deze blijven in dezelfde module zodat Numba's cache wijzigingen meeneemt.
    """
    n_steps, n_samples = H.shape
    # Rekenvariabelen: pipelengte, groeisnelheid en tijdstippen per sample.
    l_time = np.empty((n_steps, n_samples))
    dldt = np.zeros((n_steps, n_samples))
    l_time[0] = l_initial
    time_uplift = np.full(n_samples, np.nan)
    first_erosion = np.full(n_samples, np.nan)
    intervention_time = np.full(n_samples, np.nan)
    intervention_active = np.zeros(n_samples, dtype=np.bool_)
    for i in range(n_samples):
        if has_initial_pipe[i]:
            # Een bestaande pipe veronderstelt opbarsten vanaf het eerste tijdstip.
            time_uplift[i] = timegrid[0]

    # Tijdstappen: de nieuwe pipelengte hangt af van de vorige stap.
    for t in range(1, n_steps):
        for i in range(n_samples):
            previous_length = l_time[t - 1, i]
            if not can_erode[i]:
                # Zonder erosie blijft de volledige kolom op de beginlengte.
                l_time[t, i] = previous_length
                continue
            # Evenwichtscurve/verval, inclusief de modelfactor piping in Hc.
            H_equilibrium = _H_equilibrium(Hc[i], lc[i], previous_length, L[i])

            # Opbarsten: het eerste optreden blijft geldig in latere tijdstappen.
            if math.isnan(time_uplift[i]) and Z_uplift[t, i] < 0:
                time_uplift[i] = timegrid[t]
            heave = Z_heave[t, i] < 0
            # Eerste uitspoelen van zand: opbarsten, heave en voldoende verval.
            if (
                not math.isnan(time_uplift[i])
                and heave
                and H[t, i] >= H_equilibrium
                and math.isnan(first_erosion[i])
            ):
                first_erosion[i] = timegrid[t]

            partial_step = False
            fraction = 1.0
            # Bepaal of interventie heeft plaatsgevonden. De klok start bij het
            # eerste uitspoelen; na interventie stopt de groei permanent.
            if (
                detect
                and not math.isnan(first_erosion[i])
                and timegrid[t] > first_erosion[i] + intervention_hours[i]
            ):
                if not intervention_active[i]:
                    intervention_time[i] = first_erosion[i] + intervention_hours[i]
                    # Groei over het deel van de stap voor de interventie.
                    fraction = (intervention_time[i] - timegrid[t - 1]) / (
                        timegrid[t] - timegrid[t - 1]
                    )
                    partial_step = fraction > 0 and heave and previous_length < L[i]
                intervention_active[i] = True

            # Voorwaarde voor pipegroei na het eerste uitspoelen.
            # De pipe groeit niet verder zodra L is bereikt of interventie actief is.
            grow = (
                timegrid[t] >= first_erosion[i]
                and heave
                and previous_length < L[i]
                and not intervention_active[i]
            )
            if grow or partial_step:
                rate = progression_rate(
                    H[t, i],
                    H_equilibrium,
                    previous_length,
                    L[i],
                    modeltype,
                    k[i],
                    n_aq[i],
                    dldt_constant[i],
                    dldt_Ce[i],
                    dldt_kezdi_mv[i],
                    dldt_kezdi_C[i],
                )
                dldt[t, i] = fraction * rate if partial_step else rate

            # Pipelengte bijwerken; de tijdstap is in uren en de snelheid in m/s.
            length = previous_length + dldt[t, i] * (
                (timegrid[t] - timegrid[t - 1]) * 3600
            )
            # Voorkom doorschieten voorbij lc zolang Hc nog niet is bereikt.
            # H_cummax[t, i] == max(H[: t + 1, i]), dus inclusief de huidige stap.
            # Behoud minstens de beginlengte van de pipe.
            if length > lc[i] and H_cummax[t, i] < Hc[i]:
                length = max(lc[i], l_initial[i])
            l_time[t, i] = length

    return l_time, dldt, first_erosion, intervention_time


class PipeGrowth(NamedTuple):
    """Pipegroei per sample, met één vaste vorm voor de volledige tijdreeks.

    Attributes
    ----------
    l_end : np.ndarray
        Eindlengte per sample [m], vorm (samples,).
    l_time : np.ndarray
        Pipelengte [m], vorm (tijdstippen, samples). Samples zonder erosie
        houden op ieder tijdstip hun beginlengte.
    dldt_avg : np.ndarray
        Gemiddelde snelheid over stappen met positieve groei [m/s].
    dldt_end : np.ndarray
        Groeisnelheid in de laatste tijdstap [m/s].
    t_uplift : np.ndarray
        Eerste tijdstip van opbarsten [uur].
    t_heave : np.ndarray
        Eerste tijdstip van heave [uur].
    t_sell : np.ndarray
        Eerste overschrijding van het kritieke verval [uur].
    t_first_erosion : np.ndarray
        Eerste uitspoelen van zand [uur].
    t_intervention : np.ndarray
        Exact interventietijdstip [uur].
    t_fail : np.ndarray
        Eerste tijdstip waarop de pipe langer is dan L [uur].

    Notes
    -----
    Alle velden behalve `l_time` hebben vorm (samples,). Tijdstippen zijn NaN
    als de gebeurtenis niet optreedt of erosie niet mogelijk is. Eén scalair
    sample heeft dezelfde uitvoervorm als een batch met één sample.
    """

    l_end: np.ndarray
    l_time: np.ndarray
    dldt_avg: np.ndarray
    dldt_end: np.ndarray
    t_uplift: np.ndarray
    t_heave: np.ndarray
    t_sell: np.ndarray
    t_first_erosion: np.ndarray
    t_intervention: np.ndarray
    t_fail: np.ndarray


def _check_growth_inputs(
    settings: PipingSettings, stochasts: set[str] = frozenset()
) -> None:
    """Controleer vaste groei-invoer; stochasts worden na sampling ingevuld."""
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
        value = np.asarray(getattr(settings, name), dtype=float)
        if np.isnan(value).any():
            raise ValueError(
                f"'{name}' is NaN, maar is nodig voor {settings.dldt_modeltype=}"
            )
        if name == "n_aq" and (value == 0).any():
            raise ValueError(
                f"'{name}' is nul, maar mag niet nul zijn voor {settings.dldt_modeltype=}"
            )

    # Detectie is een instelling, geen stochast.
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


def pipe_growth(settings: PipingSettings) -> PipeGrowth:
    """Bereken timedep pipegroei voor alle samples in één waterstandsreeks.

    Parameters
    ----------
    settings : PipingSettings
        Scalaire instellingen of een batch met één array per modelparameter.
        Zet de gedeelde reeks met `settings.set_waterstand_tijdreeks`.

    Returns
    -------
    PipeGrowth
        Eindlengtes, volledige lengtematrix, groeisnelheden en tijdstippen.

    Notes
    -----
    Er is één gecachte Numba-integratie voor alle groeimodellen. De eerste
    aanroep kan langer duren door compilatie. Detectie start bij het eerste
    uitspoelen; een interventie stopt de groei permanent, met een deelstap
    tot aan het exacte interventietijdstip.

    `Pol2022` volgt Pol (2022), vergelijking 6.5, en is dezelfde formule als
    in Pol et al. (2024):

    - Pol, J.C. (2022). *Time-dependent development of Backward Erosion
      Piping*. PhD thesis, TU Delft.
      doi:10.4233/uuid:eb5ed3b2-4210-489e-b329-59722a0c50a0 (equation 6.5)
    - Pol, J.C., Noordam, A. and Kanning, W. (2024). A 3D time-dependent
      backward erosion piping model. *Computers and Geotechnics*.
    """
    if settings.timegrid is None or settings.h_timeseries is None:
        raise ValueError("Zet eerst de waterstandsreeks met set_waterstand_tijdreeks")
    # Eén scalair sample wordt dezelfde batchvorm als ieder ander sample.
    if np.ndim(settings.L) == 0:
        settings = settings.copy()
        settings.set_param("L", np.atleast_1d(settings.L))
    _check_growth_inputs(settings)
    PipingEquations.check_stijghoogte(settings)

    time = settings.timegrid[:, 0]
    waterlevel = settings.h_timeseries
    if settings.methode_stijghoogte == "responsfactor":
        phi = PipingEquations.stijghoogte_responsfactor(waterlevel, settings)
    elif settings.methode_stijghoogte == "responsfactor_phi_gem":
        phi = PipingEquations.stijghoogte_responsfactor_phi_gem(waterlevel, settings)
    else:  # PipingSettings en check_stijghoogte controleren de methode.
        phi = PipingEquations.stijghoogte_TRWD_4A(waterlevel, settings)

    # Fysische grootheden één keer per sample, vóór de tijdintegratie.
    l_initial = settings.l_ini * settings.L
    lc = settings.L * 0.5 * np.tanh(2 * settings.D / settings.L)
    Hc = settings.m_p * PipingEquations.H_c(settings)
    delta_phi = PipingEquations.delta_phi_cu(settings)
    can_erode = erosion_potential(
        np.max(phi, axis=0),
        float(np.max(waterlevel)),
        l_initial,
        settings.L,
        lc,
        Hc,
        settings.m_u,
        delta_phi,
        settings.h_exit,
        settings.i_ch,
        settings.D_cover,
        settings.r_c,
    )
    Z_uplift = PipingEquations._Z_u(settings.m_u, delta_phi, phi, settings.h_exit)
    Z_heave = PipingEquations._Z_h(
        settings.i_ch, phi, settings.h_exit, settings.D_cover
    )
    H = PipingEquations.verval(
        waterlevel, settings.h_exit, settings.r_c, settings.D_cover
    )
    Z_sell = Hc - H
    H[H < 0] = 0.0
    has_initial_pipe = settings.l_ini > 1e-6
    lengths, rates, first, intervention = integrate_pipe_growth(
        time,
        H,
        np.maximum.accumulate(H, axis=0),
        Z_uplift,
        Z_heave,
        settings.L,
        lc,
        Hc,
        l_initial,
        has_initial_pipe,
        can_erode,
        settings.dldt_modeltype,
        settings.k,
        settings.n_aq,
        settings.dldt_constant,
        settings.dldt_Ce,
        settings.dldt_kezdi_mv,
        settings.dldt_kezdi_C,
        bool(settings.detect_true),
        settings.intervention_hours,
    )

    def first_time(mask):
        return np.where(
            can_erode & mask.any(axis=0), time[np.argmax(mask, axis=0)], np.nan
        )

    # Opbarsten kan al op het eerste tijdstip optreden. Een aanwezige pipe
    # geldt vanaf dat tijdstip als opgebarsten; niet-eroderende samples blijven NaN.
    uplift = np.where(
        can_erode,
        np.where(
            has_initial_pipe,
            time[0],
            np.where(np.min(Z_uplift, axis=0) < 0, first_time(Z_uplift < 0), np.nan),
        ),
        np.nan,
    )
    # Gemiddelde snelheid alleen over stappen met positieve groei, ook wanneer
    # de correctie op de kritieke lengte de uiteindelijke groei heeft begrensd.
    positive = rates > 0
    average_rate = np.where(positive, rates, 0).sum(axis=0) / np.maximum(
        positive.sum(axis=0), 1
    )
    return PipeGrowth(
        lengths[-1].copy(),
        lengths,
        average_rate,
        rates[-1].copy(),
        uplift,
        first_time(Z_heave < 0),
        first_time(Z_sell < 0),
        first,
        intervention,
        first_time((settings.L - lengths) < 0),
    )
