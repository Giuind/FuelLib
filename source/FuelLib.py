import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
if FUELLIB_DIR not in sys.path:
    sys.path.append(FUELLIB_DIR)
from paths import *

from typing import Optional

class fuel:
    """
    Class for handling group contribution calculations of thermodynamic and mixture properties.

    :param name: Name of the mixture as it appears in its gcData file.
    :type name: str
    :param decompName: Name of the groupDecomposition file if different from name. Defaults to None.
    :type decompName: str, optional
    :param fuelDataDir: Directory where the fuel data is stored. Defaults to FuelLib/fuelData.
    :type fuelDataDir: str, optional
    """

    # Number of first and second order groups from Constantinou and Gani
    N_g1 = 78
    N_g2 = 43

    # Boltzmann's constant J/K
    k_B = 1.380649e-23

    def __init__(self, name=None, decompName=None, fuelDataDir=FUELDATA_DIR, only_compounds=False, keywords_dict={}):
        """
        Initialize the fuel object and read tables.

        :param name: Name of the mixture as it appears in its gcData file or list of compounds strings.
        :type name: str
        :param decompName: Name of the groupDecomposition file if different from name.
        :type decompName: str, optional
        :param fuelDataDir: Directory where the fuel data is stored.
        :type fuelDataDir: str, optional
        """
        self.name = name
        if  not only_compounds:
            if decompName is None:
                decompName = name
        else:
            self.only_compounds = only_compounds

        if fuelDataDir != FUELDATA_DIR:
            self.fuelDataDir = fuelDataDir
            self.fuelDataGcDir = os.path.join(self.fuelDataDir, "gcData")
            self.fuelDataDecompDir = os.path.join(
                self.fuelDataDir, "groupDecompositionData"
            )
        else:
            self.fuelDataDir = FUELDATA_DIR
            self.fuelDataGcDir = FUELDATA_GC_DIR
            self.fuelDataDecompDir = FUELDATA_DECOMP_DIR

        if only_compounds:
            self.groupDecompFile = os.path.join(self.fuelDataDecompDir, "refCompounds.csv")
            # Composition is passed in input when only_compounds
            self.gcxgcFile = None
        else:
            self.groupDecompFile = os.path.join(self.fuelDataDecompDir, f"{decompName}.csv")
            self.gcxgcFile = os.path.join(self.fuelDataGcDir, f"{name}_init.csv")
        self.gcmTableFile = os.path.join(GCMTABLE_DIR, "gcmTable.csv")

        # Read functional group data for mixture (num_compounds,num_groups)
        self.df_Nij = pd.read_csv(self.groupDecompFile)
        self.df_Nij = self.df_Nij.set_index(self.df_Nij.columns[0])
        # Only look-up occurrences of selected compounds from df_Nij
        # At this point GC controbutons not looked up yet--> do not calc prop
        self.lookup_occurrences(calc_properties=False)

        # Read energy interaction coefficients matrix
        self.keywords_dict = keywords_dict
        if self.keywords_dict.get("activity_coeffs"):
            self.aFile = os.path.join(GCMTABLE_DIR, "a.csv")
            df_a = pd.read_csv(self.aFile, header=None)
            self.a = df_a.to_numpy()

            if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
                raise ValueError(
                    f"Invalid interaction matrix in {self.aFile}:\n"
                    f"Matrix must be square, but shape is {self.a.shape}."
                )

            if self.a.shape[0] != self.num_groups:
                raise ValueError(
                    f"Inconsistent UNIFAC data:\n"
                    f"The number of functional groups in {self.groupDecompFile} "
                    f"({self.num_groups}) does not match the size of the interaction "
                    f"matrix in {self.aFile} ({self.a.shape[0]})."
                )

        # Read and store GCM table properties
        df_table = pd.read_csv(self.gcmTableFile)
        df_table = df_table.drop(columns=["Units"])

        def get_row(property_name):
            """
            Get property row from GCM table.

            :param property_name: Name of the property to retrieve.
            :type property_name: str
            :return: Property values for all functional groups.
            :rtype: np.ndarray
            :raises ValueError: If property not found in GCM table.
            """
            row = df_table[df_table["Property"] == property_name]
            if row.empty:
                raise ValueError(f"Property '{property_name}' not found in GCM table.")
            return row.iloc[:, 1:].to_numpy().flatten()

        # Table data for functional groups (num_compounds,)
        self.mwk = get_row("MW")  # molecular weights (g/mol)
        if self.keywords_dict.get("critical_temp"): 
            self.Tck = get_row("tck")  # critical temperature (K)
        if self.keywords_dict.get("critical_press"): 
            self.Pck = get_row("pck")  # critical pressure (bar)
        if self.keywords_dict.get("critical_vol"): 
            self.Vck = get_row("vck")  # critical volume (m^3/kmol)
        if self.keywords_dict.get("boiling_temp"): 
            self.Tbk = get_row("tbk")  # boiling temperature (K)
        if self.keywords_dict.get("melting_temp"): 
            self.Tmk = get_row("tmk")  # melting point temperature (K)
        if self.keywords_dict.get("formation_enth"): 
            self.hfk = get_row("hfk")  # enthalpy of formation, (kJ/mol)
        if self.keywords_dict.get("formation_gibbs"): 
            self.gfk = get_row("gfk")  # Gibbs energy (kJ/mol)
        if self.keywords_dict.get("latent_heat"): 
            self.hvk = get_row("hvk")  # latent heat of vaporization (kJ/mol)
        if self.keywords_dict.get("acentric_factor"): 
            self.wk = get_row("wk")  # accentric factor (1)
        if self.keywords_dict.get("liquid_vol"): 
            self.Vmk = get_row("vmk")  # (molar liquid volume at 298 K) (m^3/kmol)
        if self.keywords_dict.get("specific_heat"): 
            self.cpak = get_row("CpAk")  # specific heat values (J/mol/K)
            self.cpbk = get_row("CpBk")  # specific heat values (J/mol/K)
            self.cpck = get_row("CpCk")  # specific heat values (J/mol/K)
        if self.keywords_dict.get("activity_coeffs"):
            self.Rk = get_row("Rk")   # group surface area parameters (1)
            self.Qk = get_row("Qk")   # group volume contributions (1)
        
        # Calc properties based on look-up GC contributions
        self.get_properties()


    def get_properties(self):
        """
        Compute properties at std conditions (num_compounds,)
        """

        # Molecular weights
        self.MW = np.matmul(self.Nij, self.mwk)  # g/mol
        self.MW *= 1e-3  # Convert to kg/mol

        if self.keywords_dict.get("critical_temp"): 
            self.Tc = 181.128 * np.log(np.matmul(self.Nij, self.Tck))  # K
        if self.keywords_dict.get("critical_press"): 
            self.Pc = 1.3705 + (np.matmul(self.Nij, self.Pck) + 0.10022) ** (-2)  # bar
            self.Pc *= 1e5  # Convert to Pa from bar
        if self.keywords_dict.get("critical_vol"): 
            self.Vc = -0.00435 + (np.matmul(self.Nij, self.Vck))  # m^3/kmol
            self.Vc *= 1e-3  # Convert to m^3/mol
        if self.keywords_dict.get("boiling_temp"): 
            self.Tb = 204.359 * np.log(np.matmul(self.Nij, self.Tbk))  # K
        if self.keywords_dict.get("melting_temp"): 
            self.Tm = 102.425 * np.log(np.matmul(self.Nij, self.Tmk))  # K
        if self.keywords_dict.get("formation_enth"): 
            self.Hf = 10.835 + np.matmul(self.Nij, self.hfk)  # kJ/mol
            self.Hf *= 1e3  # Convert to J/mol
        if self.keywords_dict.get("formation_gibbs"): 
            self.Gf = -14.828 + np.matmul(self.Nij, self.gfk)  # kJ/mol
            self.Gf *= 1e3  # Convert to J/mol
        if self.keywords_dict.get("latent_heat"): 
            self.Hv_stp = 6.829 + (np.matmul(self.Nij, self.hvk))  # kJ/mol
            self.Hv_stp *= 1e3  # Convert to J/mol
            # L_v,stp (latent heat of vaporization at 298 K)
            self.Lv_stp = self.Hv_stp / self.MW  # J/kg
        if self.keywords_dict.get("acentric_factor"): 
            self.omega = 0.4085 * np.log(np.matmul(self.Nij, self.wk) + 1.1507) ** (1.0 / 0.5050)
        if self.keywords_dict.get("liquid_vol"): 
            self.Vm_stp = 0.01211 + np.matmul(self.Nij, self.Vmk)  # m^3/kmol
            self.Vm_stp *= 1e-3  # Convert to m^3/mol
        if self.keywords_dict.get("specific_heat"): 
            # C_p,stp (specific heat at 298 K)
            self.Cp_stp = np.matmul(self.Nij, self.cpak) - 19.7779  # J/mol/K

            # Temperature corrections for C_p
            self.Cp_B = np.matmul(self.Nij, self.cpbk)
            self.Cp_C = np.matmul(self.Nij, self.cpck)

        required = ["critical_temp", "critical_press", "acentric_factor"]
        if all(self.keywords_dict.get(key, False) for key in required):
            # Lennard-Jones parameters for diffusion calculations (Tee et al. 1966)
            self.epsilonByKB = (0.7915 + 0.1693 * self.omega) * self.Tc  # K
            Pc_atm = self.Pc / 101325  # atm
            self.sigma = (2.3551 - 0.0874 * self.omega) * (self.Tc / Pc_atm) ** (
                1.0 / 3
            )  # Angstroms
            self.sigma *= 1e-10  # Convert from Angstroms to m


        
    def lookup_occurrences(self, calc_properties=False):
        missing = set(self.name) - set(self.df_Nij.index)
        if missing:
            raise ValueError(f"Missing species in refCompounds.csv: {missing}")
        if not self.only_compounds:
            self.Nij = self.df_Nij.to_numpy()
        else:
            self.Nij = self.df_Nij.loc[self.name].to_numpy()
        self.num_compounds = self.Nij.shape[0]
        self.num_groups = self.Nij.shape[1]

        # Classify hydrocarbon by family (used in thermal conductivity)
        # 0: saturated hydrocarbons
        # 1: aromatics
        # 2: cycloparaffins
        # 3: olefins
        self.fam = np.zeros(self.num_compounds, dtype=int)
        aromatics = 10  # starting index for aromatic groups
        num_aromatics = 5
        cyclos = 84  # starting index for membered ring groups
        num_cyclos = 5
        olefins = 4  # starting index for double bound groups
        num_olefins = 6
        for i in range(self.num_compounds):
            # Check if aromatic: does it contain AC's?
            if sum(self.Nij[i, aromatics : aromatics + num_aromatics]) > 0:
                self.fam[i] = 1
            # Check if cycloparaffin: does it contain rings?
            elif sum(self.Nij[i, cyclos : cyclos + num_cyclos]) > 0:
                self.fam[i] = 2
            # Check if olefin: does it contain double bonds?
            elif sum(self.Nij[i, olefins : olefins + num_olefins]) > 0:
                self.fam[i] = 3

        # recaclulate properties due to Nij update
        if calc_properties:
            self.get_properties()


    def set_composition(self, compounds, Y : Optional[np.ndarray] = None):
        """
        Calculate GCM properties
        """
        # Only look-up occurrences of selected compounds from df_Nij
        self.name = compounds
        # Anytime set_composition is called, Nij might be updated. We need to update properties then
        self.lookup_occurrences(calc_properties=True)

        if  not self.only_compounds:
            # Read initial liquid composition of mixture and normalize to get mass frac
            df_gcxgc = pd.read_csv(self.gcxgcFile)
            self.compounds = [
                compound.strip() for compound in df_gcxgc["Compound"].to_list()
            ]
            if "PelePhysics Key" in df_gcxgc.columns:
                self.pelephysics_keys = [
                    key.strip() for key in df_gcxgc["PelePhysics Key"].to_list()
                ]
            else:
                self.pelephysics_keys = None
            self.Y_0 = df_gcxgc["Weight %"].to_numpy().flatten().astype(float)
            self.Y_0 /= np.sum(self.Y_0)
        else:
            self.compounds = self.name
            self.pelephysics_keys = None
            self.Y_0 = Y
            #self.Y_0 = np.zeros(self.num_compounds, dtype=float)
            #for i in range(len(self.Y_0)):
            #    self.Y_0[i] = Y[i]


        # Make sure mixture data is consistent:
        if self.num_groups < self.N_g1:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of columns in {self.groupDecompFile} is less than "
                f"the required number of first-order groups (N_g1 = {self.N_g1})."
            )
        if self.Y_0.shape[0] != self.num_compounds:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of compounds in {self.groupDecompFile} does not "
                f"equal the number of compounds in {self.gcxgcFile}."
            )

        self.X_0 = self.Y2X(self.Y_0)
        # TODO: this must be calc/called only once (everytime Nij changes!)
        if self.keywords_dict.get("activity_coeffs"):
            self.initialize_activity_coefficients(self.X_0)


    def initialize_activity_coefficients(self, X : Optional[np.ndarray] = None):

        # Activity coefficients
        r = np.matmul(self.Nij, self.Rk)
        q = np.matmul(self.Nij, self.Qk)
        z = 10.0
        LVec = (z/2)*(r - q) - (r - 1)
        thetaVec = X * q / np.dot(X, q)
        phiVec   = X * r / np.dot(X, r)
        # TODO: treat following divisions in log() as np.divide(numerator, arg_log, out=np.zeros_like(numerator), where=arg_log != 0)
        self.gammaCVec = np.exp(
            np.log(phiVec / X)
            + (z / 2.0) * q * np.log(thetaVec / phiVec)
            + LVec
            - (phiVec / X) * np.dot(X, LVec)
        )
        # Replace NaNs (fully vaporized species) with 1
        self.gammaCVec = np.nan_to_num(self.gammaCVec, nan=1.0)
        # -----------------------------
        # Residual part
        # -----------------------------
        # Group mole fractions
        # TODO: double check XVec.shape == (num_groups,)
        XVec = X @ self.Nij
        XVec = XVec / np.sum(XVec)
        self.ThetaVec = self.Qk * XVec / np.dot(self.Qk, XVec)

        


    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(self, Yi):
        """
        Calculate the mean molecular weight of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mean molecular weight of the mixture in kg/mol.
        :rtype: float
        """
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / self.MW)  # mean molar weight of the mixture
        else:
            Mbar = 0.0

        return Mbar

    def mass2Y(self, mass):
        """
        Calculate the mass fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Normalize to get group mole fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW)

        return Yi

    def mass2X(self, mass):
        """
        Calculate the mole fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the number of moles for each compound
        num_mole = mass / self.MW

        # Normalize to get group mole fractions
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            Xi = num_mole / total_moles
        else:
            Xi = np.zeros_like(self.MW)

        return Xi

    def X2Y(self, Xi):
        """
        Calculate the mass fractions from the mole fractions of each component.

        :param Xi: Mole fractions of each compound.
        :type Xi: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the mass for each compound
        mass = Xi * self.MW

        # Normalize to get group mass fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = np.zeros_like(self.MW)

        return Yi

    def Y2X(self, Yi):
        """
        Calculate the mole fractions from the mass fractions of each component.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        Mbar = self.mean_molecular_weight(Yi)
        if np.sum(Yi) != 0:
            Xi = Mbar * Yi / self.MW
        else:
            Xi = np.zeros_like(self.MW)

        return Xi

    def density(self, T, comp_idx=None):
        """
        Calculate the density of each component at temperature T.

        :param T: Temperature of the mixture in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Density of each compound in kg/m^3.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            MW = self.MW  # kg/mol
            Vm = self.molar_liquid_vol(T)  # m^3/mol
        else:
            MW = self.MW[comp_idx]  # kg/mol
            Vm = self.molar_liquid_vol(T, comp_idx=comp_idx)  # m^3/mol

        rho = MW / Vm  # kg/m^3
        return rho

    def viscosity_kinematic(self, T, comp_idx=None):
        """
        Calculate the viscosity using Dutt's equation.

        :meta private: This uses Dutt's equation (4.23) from "Viscosity of Liquids".
        :meta private: The equation predicts viscosity in mm^2/s and is converted to SI units.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Viscosity of each component in m^2/s.
        :rtype: np.ndarray
        """

        # Convert temperature to Celsius
        T_cels = K2C(T)
        if comp_idx is None:
            Tb_cels = K2C(self.Tb)
        else:
            Tb_cels = K2C(self.Tb[comp_idx])

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239 - 0.19 * Tb_cels)
        nu_i = np.exp(rhs)  # Viscosity in mm^2/s

        # Convert to SI (m^2/s)
        nu_i = nu_i * 1e-6

        return nu_i

    def viscosity_dynamic(self, T, comp_idx=None):
        """
        Calculate liquid dynamic viscosity based on droplet temperature and density.

        :meta private: Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Dynamic viscosity in Pa*s.
        :rtype: np.ndarray
        """

        nu_i = self.viscosity_kinematic(T, comp_idx=comp_idx)  # m^2/s
        rho_i = self.density(T, comp_idx=comp_idx)  # kg/m^3
        mu_i = nu_i * rho_i  # Pa*s
        return mu_i

    def Cp(self, T, comp_idx=None):
        """
        Compute specific heat capacity at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Specific heat capacity in J/mol/K.
        :rtype: np.ndarray
        """

        theta = (T - 298) / 700
        if comp_idx is None:
            Cp_stp = self.Cp_stp
            Cp_B = self.Cp_B
            Cp_C = self.Cp_C
        else:
            Cp_stp = self.Cp_stp[comp_idx]
            Cp_B = self.Cp_B[comp_idx]
            Cp_C = self.Cp_C[comp_idx]

        cp = Cp_stp + Cp_B * theta + Cp_C * theta**2

        return cp

    def Cl(self, T, comp_idx=None):
        """
        Compute liquid specific heat capacity in J/kg/K at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Specific heat capacity in J/kg/K.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            MW = self.MW
        else:
            MW = self.MW[comp_idx]
        cp = self.Cp(T, comp_idx=comp_idx)
        return cp / MW

    def psat(self, T, comp_idx=None, correlation="Lee-Kesler"):
        """
        Compute saturated vapor pressure.

        :meta private: Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Saturated vapor pressure in Pa.
        :rtype: np.ndarray
        """

        if comp_idx is None:
            Tr = T / self.Tc
            Pc = self.Pc
            omega = self.omega
        else:
            Tr = T / self.Tc[comp_idx]
            Pc = self.Pc[comp_idx]
            omega = self.omega[comp_idx]

        if correlation.casefold() == "Ambrose-Walton".casefold():
            # May cause trouble at high temperatures
            tau = 1 - Tr
            f0 = (
                -5.97616 * tau
                + 1.29874 * tau**1.5
                - 0.60394 * tau**2.5
                - 1.06841 * tau**5.0
            )
            f0 /= Tr
            f1 = (
                -5.03365 * tau
                + 1.11505 * tau**1.5
                - 5.41217 * tau**2.5
                - 7.46628 * tau**5.0
            )
            f1 /= Tr
            f2 = (
                -0.64771 * tau
                + 2.41539 * tau**1.5
                - 4.26979 * tau**2.5
                - 3.25259 * tau**5.0
            )
            f2 /= Tr
            rhs = np.exp(f0 + omega * f1 + omega**2 * f2)

        else:  # Default correlation is Lee-Kesler
            f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
            f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
            rhs = np.exp(f0 + omega * f1)

        psat = Pc * rhs
        return psat

    def psat_antoine_coeffs(self, Tvals=None, units="mks", correlation="Lee-Kesler"):
        """
        Estimate Antoine coefficients for vapor pressure of an individual compound.

        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, Tb_i]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: 4 np.ndarrays
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = np.zeros(20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """Antoine equation for vapor pressure."""
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        # Fit Antoine coefficients for each compound
        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T = np.linspace(273.15, self.Tb[i], 20)
            Pvals = np.zeros_like(T)
            for k in range(len(T)):
                Pvals[k] = 1 / D * self.psat(T[k], correlation=correlation)[i]

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def molar_liquid_vol(self, T, comp_idx=None):
        """
        Compute molar liquid volume with temperature correction.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar liquid volume in m^3/mol.
        :rtype: np.ndarray
        """

        Tstp = 298.0
        if comp_idx is None:
            Tc = self.Tc
            omega = self.omega
            Vm_stp = self.Vm_stp
        else:
            Tc = np.array([self.Tc[comp_idx]])
            omega = np.array([self.omega[comp_idx]])
            Vm_stp = np.array([self.Vm_stp[comp_idx]])
        phi = np.zeros_like(Tc)
        for i in range(len(Tc)):
            if T > Tc[i]:
                phi[i] = -((1 - (Tstp / Tc[i])) ** (2.0 / 7.0))
            else:
                phi[i] = ((1 - (T / Tc[i])) ** (2.0 / 7.0)) - (
                    (1 - (Tstp / Tc[i])) ** (2.0 / 7.0)
                )
        z = 0.29056 - 0.08775 * omega
        Vmi = Vm_stp * np.power(z, phi)
        if comp_idx is not None:
            Vmi = Vmi[0]
        return Vmi

    def latent_heat_vaporization(self, T, comp_idx=None):
        """
        Calculate latent heat of vaporization adjusted for temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Latent heat of vaporization in J/kg.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            Tc = self.Tc
            Tb = self.Tb
            Lv_stp = self.Lv_stp
        else:
            Tc = np.array([self.Tc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            Lv_stp = np.array([self.Lv_stp[comp_idx]])

        # Reduced temperatures
        Tr = T / Tc
        Trb = Tb / Tc

        Lvi = np.zeros_like(Tc)
        for i in range(len(Tc)):
            if T > Tc[i]:
                Lvi[i] = 0.0
            else:
                Lvi[i] = Lv_stp[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)

        if comp_idx is not None:
            Lvi = Lvi[0]
        return Lvi

    def diffusion_coeff(
        self,
        p,
        T,
        sigma_gas=3.62e-10,
        epsilonByKB_gas=97.0,
        MW_gas=28.97e-3,
        correlation="Tee",
    ):
        """
        Compute diffusion coefficients using Lennard-Jones parameters.

        :meta private: Uses Wilke and Lee method (Poling, equation 11-4.1).
        :meta private: Ambient gas defaults to air parameters.

        :param p: Pressure in Pa.
        :type p: float
        :param T: Temperature in Kelvin.
        :type T: float
        :param sigma_gas: Collision diameter in m.
        :type sigma_gas: float, optional
        :param epsilonByKB_gas: Well depth over Boltzmann constant, in K.
        :type epsilonByKB_gas: float, optional
        :param MW_gas: Mean molecular weight of ambient gas in kg/mol.
        :type MW_gas: float, optional
        :param correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
        :type correlation: str, optional
        :return: Diffusion coefficient.
        :rtype: np.ndarray
        """

        # Method of Tee for calculating liquid sigma and epsilon
        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma * 1e10  # convert from m to Angstroms
            epsilonByKB_i = self.epsilonByKB  # K
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = np.zeros_like(self.Tb)
            for n in range(self.num_compounds):
                Vmb_i[n] = self.molar_liquid_vol(self.Tb[n])[n] * 1e6  # cm^3/mol
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * self.Tb  # K , Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigma_gas = sigma_gas * 1e10  # convert from m to Angstroms
        sigmaAB_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = (
            epsilonByKB_gas * epsilonByKB_i
        ) ** 0.5  # K, Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        Tstar_i = T / epsilonAB_byKB_i  # [1]
        A = 1.06036
        B = 0.15610
        C = 0.193
        D = 0.47635
        E = 1.03587
        F = 1.52996
        G = 1.76474
        H = 3.89411
        omegaD_i = (
            A / (Tstar_i**B)
            + C / np.exp(D * Tstar_i)
            + E / np.exp(F * Tstar_i)
            + G / np.exp(H * Tstar_i)
        )

        # Convert molecular weights from kg/mol to g/mol then calculate M_AB
        MW_gas = MW_gas * 1e3
        MW_i = self.MW * 1e3
        M_AB_i = 2 * (MW_i * MW_gas) / (MW_i + MW_gas)  # g/mol, see Poling (11-3.1)

        # Convert pressure from Pa to bar
        p = p * 1e-5  # bar

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T**1.5)
            / (p * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm^2/s
        D_AB_i = D_AB_i * 1e-4  # Convert to m^2/s

        return D_AB_i

    def surface_tension(self, T, comp_idx=None, correlation="Brock-Bird"):
        """
        Calculate surface tension of each compound at a given temperature.

        :meta private: Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Brock-Bird" or "Pitzer").
        :type correlation: str, optional
        :return: Surface tension in N/m.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            Tc = self.Tc
            Pc = self.Pc
            Tb = self.Tb
            omega = self.omega
        else:
            Tc = np.array([self.Tc[comp_idx]])
            Pc = np.array([self.Pc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            omega = np.array([self.omega[comp_idx]])
        Tr = T / Tc
        Pc = Pc * 1e-5  # convert from Pa to bar

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb / Tc
            Q = 0.1196 * (1.0 + (Tbr * np.log(Pc / 1.01325)) / (1.0 - Tbr)) - 0.279
        else:
            w = omega
            Q = (
                (1.86 + 1.18 * w)
                / 19.05
                * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
            )

        st = Pc ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)

        st = st * 1e-3  # Convert from dyn/cm to N/m
        if comp_idx is not None:
            st = st[0]

        return st

    def thermal_conductivity(self, T, comp_idx=None):
        """
        Calculate thermal conductivity at a given temperature.

        :meta private: Uses Latini et al. method (Poling equation 10-9.1).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Thermal conductivity in W/m/K.
        :rtype: np.ndarray
        """
        if comp_idx is None:
            MW = self.MW
            Tc = self.Tc
            Tb = self.Tb
            fam = self.fam
        else:
            MW = np.array([self.MW[comp_idx]])
            Tc = np.array([self.Tc[comp_idx]])
            Tb = np.array([self.Tb[comp_idx]])
            fam = np.array([self.fam[comp_idx]])

        Astar = 0.00350 + np.zeros_like(Tc)
        alpha = 1.2
        beta = 0.5 + np.zeros_like(Tc)
        gamma = 0.167
        MW_beta = MW * 1e3  # convert from kg/mol to g/mol
        Tr = T / Tc

        for i in range(len(Tc)):
            if fam[i] == 1:
                # Aromatics
                Astar[i] = 0.0346
                beta[i] = 1.0
            elif fam[i] == 2:
                # Cycloparaffins
                Astar[i] = 0.0310
                beta[i] = 1.0
            elif fam[i] == 3:
                # Olefins
                Astar[i] = 0.0361
                beta[i] = 1.0
            MW_beta[i] = MW_beta[i] ** beta[i]

        A = Astar * Tb**alpha / (MW_beta * Tc**gamma)
        tc = A * (1 - Tr) ** (0.38) / (Tr ** (1 / 6))

        if comp_idx is not None:
            tc = tc[0]
        return tc

    def activity(self, T, comp_idx=None):

        #  ENERGY INTERACTION PARAMETER
        Psi = np.exp(-self.a / T)

        # Avoid division by zero
        denom = self.ThetaVec @ Psi
        divRes = np.divide(self.ThetaVec, denom, out=np.zeros_like(self.ThetaVec), where=denom != 0)

        GammaVec = self.Qk * (1 - np.log(denom) - divRes @ Psi.T)

        # --------------------------------
        # Isolated group gammas per comp
        # --------------------------------
        GammaIVec = np.zeros((self.num_compounds, self.num_groups))

        for i in range(self.num_compounds):
            XIVec = self.Nij[i, :] / np.sum(self.Nij[i, :])
            ThetaIVec = self.Qk * XIVec / np.dot(self.Qk, XIVec)

            denom_i = ThetaIVec @ Psi
            divRes_i = np.divide(
                ThetaIVec,
                denom_i,
                out=np.zeros_like(ThetaIVec),
                where=denom_i != 0,
            )

            GammaIVec[i, :] = self.Qk * (
                1 - np.log(denom_i) - divRes_i @ Psi.T
            )

        # -----------------------------
        # Residual activity coefficients
        # -----------------------------
        # TODO: gammaCVec also has as many elements as num_compounds (inherited from Xi)
        gammaRVec = np.zeros(self.num_compounds)

        for i in range(self.num_compounds):
            gammaRVec[i] = np.exp(
                np.dot(self.Nij[i, :], GammaVec - GammaIVec[i, :])
            )

        # -----------------------------
        # Total activity coefficient
        # -----------------------------
        result = self.gammaCVec * gammaRVec

        if comp_idx is not None:
            return result[comp_idx]

        return result


    # --- Mixture functions ---
    def mixture_density(self, Yi, T):
        """
        Calculate mixture density at a given temperature.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Mixture density in kg/m^3.
        :rtype: float
        """
        MW = self.MW  # Molecular weights of each component (kg/mol)
        Vmi = self.molar_liquid_vol(T)  # Molar volume of each component (m^3/mol)

        # Calculate density (kg/m^3)
        rho = Yi @ (MW / Vmi)

        return rho

    def mixture_kinematic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate kinematic viscosity of the mixture.

        :meta private: Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture kinematic viscosity in m^2/s.
        :rtype: float
        """
        nu_i = self.viscosity_kinematic(T)  # Viscosities of individual components

        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius mixing correlation
            nu = np.exp(np.sum(Xi * np.log(nu_i)))
        else:
            # Default: Kendall-Monroe mixing correlation
            nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** (3.0)

        return nu

    def mixture_dynamic_viscosity(self, Yi, T, correlation="Kendall-Monroe"):
        """
        Calculate dynamic viscosity of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture dynamic viscosity in Pa*s.
        :rtype: float
        """

        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)

        return rho * nu

    def mixture_vapor_pressure(self, Yi, T, correlation="Lee-Kesler"):
        """
        Calculate vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Mixture vapor pressure in Pa.
        :rtype: float
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Saturated vapor pressure for each compound (Pa)
        p_sati = self.psat(T, correlation=correlation)

        # Mixture vapor pressure via Raoult's law
        # TODO: activity coefficients implicitly assumed unity
        p_v = p_sati @ Xi

        return p_v

    def mixture_vapor_pressure_antoine_coeffs(
        self, Yi, Tvals=None, units="mks", correlation="Lee-Kesler"
    ):
        """
        Estimate Antoine coefficients for vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, min(Tb)]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: float
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            # Initialize as zeros for now, calculated for each compound later
            X = self.Y2X(Yi)
            Tb = mixing_rule(self.Tb, X)
            T = np.linspace(273.15, np.min(Tb), 20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """
            Antoine equation for vapor pressure.

            :param T: Temperature.
            :type T: float or np.ndarray
            :param A: Antoine coefficient A.
            :type A: float
            :param B: Antoine coefficient B.
            :type B: float
            :param C: Antoine coefficient C.
            :type C: float
            :return: log10(pressure).
            :rtype: float or np.ndarray
            """
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        Pvals = np.zeros_like(T)
        for k in range(len(T)):
            Pvals[k] = (
                self.mixture_vapor_pressure(Yi, T[k], correlation=correlation) / D
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])  # initial guess
        A, B, C = popt

        return A, B, C, D

    def mixture_surface_tension(self, Yi, T, correlation="Brock-Bird"):
        """
        Calculate surface tension of the mixture.

        :meta private: Uses arithmetic pseudo-property method recommended by Hugill and van Welsenes (1986).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Correlation method ("Pitzer" or "Brock-Bird").
        :type correlation: str, optional
        :return: Mixture surface tension in N/m.
        :rtype: float
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Surface tension for each compound (N/m)
        sti = self.surface_tension(T, correlation=correlation)

        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        st = mixing_rule(sti, Xi, "arithmetic")

        return st

    def mixture_thermal_conductivity(self, Yi, T):
        """
        Calculate thermal conductivity of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Thermal conductivity in W/m/K.
        :rtype: float
        """
        tc = self.thermal_conductivity(T)
        return np.sum(Yi * tc ** (-2)) ** (-0.5)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def C2K(T):
    """
    Convert temperature from Celsius to Kelvin.

    :param T: Temperature in Celsius.
    :type T: float or np.ndarray
    :return: Temperature in Kelvin.
    :rtype: float or np.ndarray
    """
    return T + 273.15


def K2C(T):
    """
    Convert temperature from Kelvin to Celsius.

    :param T: Temperature in Kelvin.
    :type T: float or np.ndarray
    :return: Temperature in Celsius.
    :rtype: float or np.ndarray
    """
    return T - 273.15


def mixing_rule(var_n, X, pseudo_prop="arithmetic"):
    """
    Mixing rules for computing mixture properties.

    :param var_n: Individual compound properties.
    :type var_n: np.ndarray
    :param X: Mole fractions of the compounds.
    :type X: np.ndarray
    :param pseudo_prop: Type of mean ("arithmetic" or "geometric").
    :type pseudo_prop: str, optional
    :return: Mixture property value.
    :rtype: float
    """
    num_comps = len(var_n)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            if pseudo_prop.casefold() == "geometric":
                # Use geometric mean definition for the pseudo property
                var_ij = (var_n[i] * var_n[j]) ** (0.5)
            else:
                # Use arithmetic definition for the pseudo property
                var_ij = (var_n[i] + var_n[j]) / 2
            var_mix += X[i] * X[j] * var_ij
    return var_mix


def droplet_volume(r):
    """
    Calculate spherical volume of a droplet given the radius.

    :param r: Radius of the droplet in meters.
    :type r: float
    :return: Spherical volume of droplet in cubic meters.
    :rtype: float
    """
    return 4.0 / 3.0 * np.pi * r**3


def droplet_mass(fuel, r, Yi, T):
    """
    Calculate the mass of each compound in the fuel provided the radius of the droplet.

    :param fuel: An instance of the groupContribution class.
    :type fuel: groupContribution object
    :param r: Radius of the droplet in meters.
    :type r: float
    :param Yi: Mass fractions of each compound.
    :type Yi: np.ndarray
    :param T: Droplet temperature in Kelvin.
    :type T: float
    :return: Mass of each compound in droplet in kg.
    :rtype: np.ndarray
    """
    volume = droplet_volume(r)  # m^3
    if volume > 0:
        return volume / (fuel.molar_liquid_vol(T) @ Yi) * Yi * fuel.MW
    else:
        return np.zeros_like(fuel.MW)
