import numpy as np

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
