from typing import Self

import numpy as np
from pydantic.dataclasses import dataclass


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
    cv_vl : float or np.ndarray
        consolidatiecoëfficiënt klei voorland [m^2/s].
    cv_al : float or np.ndarray
        consolidatiecoëfficiënt klei achterland [m^2/s].
    b : float or np.ndarray
        b-waarde in TRWD model 4D (halve breedte rivierbed) [m]
    n_aq : float or np.ndarray
        Porositeit van de aquifer [-].
    L : float or np.ndarray
        Kwelweglengte [m].
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
        Stijghoogte methode, by default "responsfactor"
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
    cv_vl: float | np.ndarray = np.nan
    cv_al: float | np.ndarray = np.nan
    b: float | np.ndarray = np.nan
    n_aq: float | np.ndarray = np.nan
    L: float | np.ndarray = np.nan
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

    def __post_init__(self) -> None:
        """
        Post-initialization to set up attributes and their dimensions.
        """
        # Determine attributes which need to be checked, store info in a dict
        ignore = ["methode_stijghoogte"]
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
            If the attribute name is unknown.
        """
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
            settings_dict[attr] = getattr(self, attr)

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
            settings_dict[attr] = getattr(self, attr)

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
