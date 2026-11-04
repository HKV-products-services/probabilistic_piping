from typing import Self

import numpy as np
from pydantic.dataclasses import dataclass

from probabilistic_piping.piping_equations import PipingEquations

# Stijghoogte methodes die beschikbaar zijn
METHODES_STIJGHOOGTE = ("responsfactor", "responsfactor_phi_gem", "TRWD_4A")


@dataclass(config={"arbitrary_types_allowed": True})
class PipingSettings:
    """
    A dataclass to store various settings and parameters related to piping calculations.

    Attributes
    ----------
    t_norm : float or np.ndarray
        Terugkeertijd norm voor de berekening (alleen voor semiprob).
    h : float or np.ndarray
        Buitenwaterstand [m+NAP].
    h_exit : float or np.ndarray
        Kwelslootpeil bij uitgang [m+NAP].
    D : float or np.ndarray
        Dikte watervoerende laag [m].
    D_cover : float or np.ndarray
        Dikte deklaag [m].
    D_vl : float or np.ndarray
        Dikte deklaag voorland [m].
    D_al : float or np.ndarray
        Dikte deklaag achterland [m].
    k : float or np.ndarray
        Doorlatendheid deklaag [m/s].
    k_vl : float or np.ndarray
        Doorlatendheid voorland [m/s].
    k_al : float or np.ndarray
        Doorlatendheid achterland [m/s].
    b : float or np.ndarray
        b-waarde in TRWD model 4D (halve breedte rivierbed) [m]
    n_aq : float or np.ndarray
        Porositeit van de aquifer [-].
    L : float or np.ndarray
        Kwelweglengte [m].
    L_dijkbasis : float or np.ndarray
        Lengte dijkbasis, onderdeel van de kwelweg bij TRWD model 4A [m].
    L_vl : float or np.ndarray
        Lengte voorland [m].
    L_al : float or np.ndarray
        Lengte achterland [m].
    gamma_sat : float or np.ndarray
         Verzadigd gewicht deklaag, gewogen gemiddelde voor meerdere lagen. [kN/m^3].
    r_exit : float or np.ndarray
        Dempingsfactor bij uitgang (dempingsfactor over de afstand intredepunt-uittredepunt) [-].
    m_u : float or np.ndarray
        Modelfactor uplift [-].
    i_ch : float or np.ndarray
        Kritieke heavegradiënt [-].
    d_70 : float or np.ndarray
        70% kwantiel korrelgrootte van pipinggevoelige zandlaag [m].
    eta : float or np.ndarray
        White's weerstandscoefficient (sleepkrachtfactor, constante van White) [-].
    theta : float or np.ndarray
        Rolweerstandshoek [graden].
    d_70m : float or np.ndarray
        Gemiddelde waarde korrelgrootte van 70e kwantiel [m].
    g : float or np.ndarray
        Zwaartekrachtversnelling [kN/m^3].
    gamma_water : float or np.ndarray
        Dichtheid water (volumiek gewicht water) [kN/m^3].
    r_c : float or np.ndarray
        Reductiefactor [-].
    v : float or np.ndarray
        Kinematische viscositeit [m^2/s]
    gamma_sp : float or np.ndarray
        Volumieke dichtheid zand onder water [kN/m^3]
    krit_verval_factor : float or np.ndarray
        Vermenigvuldigingsfactor voor kritiek verval, by default 1.0.
    m_p : float or np.ndarray
        Modelfactor piping [-].
    methode_stijghoogte : str
        Stijghoogte methode ("responsfactor", "responsfactor_phi_gem" of
        "TRWD_4A"), by default "responsfactor"
    phi_gem : float or np.ndarray
        Gemiddelde stijghoogte bij het uittredepunt, bij "responsfactor_phi_gem" [m+NAP].
    h_gem : float or np.ndarray
        Gemiddelde buitenwaterstand, bij "responsfactor_phi_gem" [m+NAP].
    dldt_modeltype : str or None
        Model voor de pipegroeisnelheid ("constant", "Pol2019" of "Pol2022"),
        verplicht voor timedep berekeningen, by default None. Geldt voor alle
        samples.
    dldt_constant : float or np.ndarray
        Constante pipegroeisnelheid, bij "constant" [m/s].
    dldt_Ce : float or np.ndarray
        Groeicoefficient, bij "Pol2022" [-].
    dldt_kezdi_C : float or np.ndarray
        Constante C in de formule van Kezdi, bij "Pol2019" [-].
    dldt_kezdi_mv : float or np.ndarray
        Constante mv in de formule van Kezdi, bij "Pol2019" [-].
    l_ini : float or np.ndarray
        Beginlengte van de pipe als fractie van de kwelweglengte L [-].
    detect_true : bool
        Of een pipe wordt opgemerkt en er dus een interventie volgt, by default
        False. Geldt voor alle samples.
    intervention_hours : float or np.ndarray
        Aantal uren tussen het begin van het uitspoelen en de interventie [uur].
    timegrid : np.ndarray or None
        Tijdstippen van de waterstandsreeks [uur], vorm n x 1. Een reeks voor
        alle samples.
    h_timeseries : np.ndarray or None
        Buitenwaterstand op de tijdstippen van timegrid [m+NAP], vorm n x 1.
    """

    t_norm: float | np.ndarray = np.nan
    h: float | np.ndarray = np.nan
    h_exit: float | np.ndarray = np.nan
    D: float | np.ndarray = np.nan
    D_cover: float | np.ndarray = np.nan
    D_vl: float | np.ndarray = np.nan
    D_al: float | np.ndarray = np.nan
    k: float | np.ndarray = np.nan
    k_vl: float | np.ndarray = np.nan
    k_al: float | np.ndarray = np.nan
    b: float | np.ndarray = np.nan
    n_aq: float | np.ndarray = np.nan
    L: float | np.ndarray = np.nan
    L_dijkbasis: float | np.ndarray = np.nan
    L_vl: float | np.ndarray = np.nan
    L_al: float | np.ndarray = np.nan
    gamma_sat: float | np.ndarray = np.nan
    r_exit: float | np.ndarray = np.nan
    m_u: float | np.ndarray = np.nan
    i_ch: float | np.ndarray = np.nan
    d_70: float | np.ndarray = np.nan
    eta: float | np.ndarray = np.nan
    theta: float | np.ndarray = np.nan
    d_70m: float | np.ndarray = np.nan
    g: float | np.ndarray = np.nan
    gamma_water: float | np.ndarray = np.nan
    r_c: float | np.ndarray = np.nan
    v: float | np.ndarray = np.nan
    gamma_sp: float | np.ndarray = np.nan
    krit_verval_factor: float | np.ndarray = 1.0
    m_p: float | np.ndarray = np.nan
    methode_stijghoogte: str = "responsfactor"
    phi_gem: float | np.ndarray = np.nan
    h_gem: float | np.ndarray = np.nan
    dldt_modeltype: str | None = None
    dldt_constant: float | np.ndarray = np.nan
    dldt_Ce: float | np.ndarray = np.nan
    dldt_kezdi_C: float | np.ndarray = np.nan
    dldt_kezdi_mv: float | np.ndarray = np.nan
    l_ini: float | np.ndarray = np.nan
    detect_true: bool = False
    intervention_hours: float | np.ndarray = np.nan
    timegrid: np.ndarray | None = None
    h_timeseries: np.ndarray | None = None

    def __post_init__(self) -> None:
        """
        Post-initialization to set up attributes and their dimensions.
        """
        self._check_methode_stijghoogte(self.methode_stijghoogte)

        # Determine attributes which need to be checked, store info in a dict
        # Instellingen die niet per sample worden uitgebreid
        ignore = [
            "methode_stijghoogte",
            "dldt_modeltype",
            "detect_true",
            "timegrid",
            "h_timeseries",
        ]
        attrs = []
        for a in dir(self):
            if (
                not a.startswith("__")
                and a not in ignore
                and not callable(getattr(self, a))
            ):
                attrs.append(a)

        self.__attrs = {}
        for attr in attrs:
            val = getattr(self, attr)
            ndim = np.ndim(val)
            self.__attrs[attr] = (val, ndim)

        self.__ignore = ignore

    @staticmethod
    def _check_methode_stijghoogte(methode: str) -> None:
        """
        Controleer of de stijghoogte methode beschikbaar is.

        Raises
        ------
        NotImplementedError
            Als de methode "superpositie_barends" is, die nog niet beschikbaar is.
        ValueError
            Als de methode onbekend is.
        """
        if methode == "superpositie_barends":
            raise NotImplementedError(
                "methode_stijghoogte 'superpositie_barends' is nog niet beschikbaar"
            )
        if methode not in METHODES_STIJGHOOGTE:
            raise ValueError(
                f"methode_stijghoogte {methode!r} is onbekend, kies uit "
                f"{', '.join(METHODES_STIJGHOOGTE)}"
            )

    def set_params_fromdict(
        self,
        piping_dict: dict[str, float | int | str | np.ndarray],
        verify_integrity: bool = True,
    ) -> None:
        """
        Set parameters from a dictionary.

        Parameters
        ----------
        piping_dict : dict
            Dictionary containing parameter names and their values.
        verify_integrity : bool, optional
            Whether to verify the integrity of the parameters after setting them, by default True.
        """
        # Set params, verify integrity after
        for k, v in piping_dict.items():
            self.set_param(k, v, verify_integrity=False)
        if verify_integrity:
            self.verify_integrity()

    def set_param(
        self,
        attr: str,
        val: float | int | str | np.ndarray,
        verify_integrity: bool = True,
    ) -> None:
        """
        Set a single parameter.

        Parameters
        ----------
        attr : str
            The name of the attribute to set.
        val : float or np.ndarray
            The value to set for the attribute.
        verify_integrity : bool, optional
            Whether to verify the integrity of the parameters after setting them, by default True.

        Raises
        ------
        ValueError
            If the attribute name is unknown, or methode_stijghoogte is unknown.
        NotImplementedError
            If methode_stijghoogte is "superpositie_barends".
        """
        if attr == "methode_stijghoogte":
            self._check_methode_stijghoogte(val)
        if hasattr(self, attr):
            ndim = np.ndim(val)
            if attr not in self.__ignore:
                self.__attrs[attr] = (val, ndim)
            setattr(self, attr, val)
        # else:
        #     raise ValueError(f"variabele {attr=} is onbekend")
        if verify_integrity:
            self.verify_integrity()

    def verify_integrity(self) -> None:
        """
        Verify the integrity of the parameters by ensuring they are numpy arrays and have consistent sizes.

        Raises
        ------
        ValueError
            If any parameter has more than one dimension.
        """
        # Enforce numpy arrays and determine max size on first axis
        max_val = 0
        for attr, (val, ndim) in self.__attrs.items():
            if ndim == 1:
                if not isinstance(val, np.ndarray):
                    # Enforce numpy arrays
                    val = np.array(val, dtype=np.float64)
                    self.__attrs[attr] = (val, np.ndim(val))
                    setattr(self, attr, val)
                max_val = max(max_val, len(val))
            elif ndim > 1:
                raise ValueError("Meer dan 1 dimensie is niet ondersteund")

        # Make number of elements per parameter consistent
        if max_val >= 1:
            for attr, (val, ndim) in self.__attrs.items():
                if ndim == 0:
                    # Fill floating points with numpy arrays
                    val = np.full((max_val,), val)
                    self.__attrs[attr] = (val, np.ndim(val))
                    setattr(self, attr, val)
                elif ndim == 1 and max_val > 1 and len(val) == 1:
                    # Expand numpy 1D array uit
                    val = np.repeat(val, max_val)
                    self.__attrs[attr] = (val, ndim)
                    setattr(self, attr, val)

                # Check that the number of elements is equal to max_val
                assert len(val) == max_val

    def copy(self, verify_integrity: bool = True) -> Self:
        """
        Create a copy of the current PipingSettings object.

        Parameters
        ----------
        verify_integrity : bool, optional
            Whether to verify the integrity of the parameters after copying, by default True.

        Returns
        -------
        PipingSettings
            A new instance of PipingSettings with copied parameters.
        """
        # Initial new settings object.
        ps = PipingSettings()

        # Copy ignored settings
        settings_dict = {}
        for attr in self.__ignore:
            val = getattr(self, attr)
            settings_dict[attr] = val.copy() if isinstance(val, np.ndarray) else val

        # Copy all other settings
        for attr, (val, ndim) in self.__attrs.items():
            if ndim == 0:
                settings_dict[attr] = val
            else:
                settings_dict[attr] = val.copy()

        # Set settings from dict
        ps.set_params_fromdict(settings_dict, verify_integrity=verify_integrity)

        return ps

    def get_partial_settings(self, i: int, verify_integrity: bool = True) -> Self:
        """
        Get a partial settings object for the i-th slice.

        Parameters
        ----------
        i : int
            The index of the slice to extract.
        verify_integrity : bool, optional
            Whether to verify the integrity of the parameters after extraction, by default True.

        Returns
        -------
        PipingSettings
            A new instance of PipingSettings with the i-th slice of parameters.
        """
        # Create new settings object
        ps = PipingSettings()

        # Copy ignored settings
        settings_dict = {}
        for attr in self.__ignore:
            val = getattr(self, attr)
            settings_dict[attr] = val.copy() if isinstance(val, np.ndarray) else val

        # Copy i-th slice for all other settings
        for attr, (val, _) in self.__attrs.items():
            settings_dict[attr] = val[[i]].copy()

        ps.set_params_fromdict(settings_dict, verify_integrity=verify_integrity)

        return ps

    def get_settings_as_dict(self) -> dict[str, float | np.ndarray]:
        """
        Get the settings as a dictionary.

        Returns
        -------
        dict
            A dictionary containing the parameter names and their values.
        """
        settings = {}
        for attr, (val, _) in self.__attrs.items():
            settings[attr] = val
        return settings

    def set_waterstand_tijdreeks(
        self,
        tijd: np.ndarray | list[float],
        waterstand: np.ndarray | list[float],
        dt: float | None = None,
    ) -> None:
        """
        Zet een waterstandsreeks, gelijk voor alle samples.

        Parameters
        ----------
        tijd : np.ndarray or list of float
            Tijdstippen [uur].
        waterstand : np.ndarray or list of float
            Buitenwaterstand op de tijdstippen [m+NAP].
        dt : float or None, optional
            Tijdstap [uur] waarmee de reeks wordt geinterpoleerd, by default None
            (de reeks blijft zoals opgegeven). De reeks eindigt exact op het
            laatste opgegeven tijdstip. Past dt niet een geheel aantal keren in de
            reeks, dan is de laatste stap korter.

        Raises
        ------
        ValueError
            Als tijd en waterstand niet even lang zijn, als er minder dan 2
            punten zijn, als de tijdstippen niet strikt stijgen of als dt niet
            positief is.
        """
        tijd = np.asarray(tijd, dtype=np.float64).reshape(-1, 1)
        waterstand = np.asarray(waterstand, dtype=np.float64).reshape(-1, 1)
        if tijd.shape != waterstand.shape:
            raise ValueError("Tijd en waterstand arrays zijn niet even lang")
        if len(tijd) < 2:
            raise ValueError("Een waterstandsreeks heeft minimaal 2 punten nodig")
        if not np.all(np.diff(tijd[:, 0]) > 0):
            raise ValueError("Tijdstippen moeten strikt stijgend zijn")
        if dt is not None and not dt > 0:
            raise ValueError("dt moet groter dan 0 zijn")

        if dt is not None:
            # Het rooster stopt exact op het laatste tijdstip, zonder extrapolatie.
            # Een punt dat door afronding vlak bij het einde valt vervalt.
            t_end = tijd[-1, 0]
            t_new = np.arange(tijd[0, 0], t_end, dt)
            t_new = np.append(t_new[t_end - t_new > 1e-9 * dt], t_end)
            waterstand = PipingEquations.inextrp1d(
                t_new, tijd[:, 0], waterstand[:, 0]
            ).reshape(-1, 1)
            tijd = t_new.reshape(-1, 1)

        self.set_param("timegrid", tijd, verify_integrity=False)
        self.set_param("h_timeseries", waterstand, verify_integrity=False)
