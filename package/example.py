

"""
Using "pip install obtain_HEA_features-0.1.2-py3-none-any.whl" to install the obtain_HEA_features module.
"""

from obtain_HEA_features import Wdk_made_feature



if __name__ == "__main__":
    # The elements for which features need to be computed, along with their corresponding mass percentages
    element_list = ["Nb", "W", "Mo", "Zr", "C"]
    composition_list = [94.71, 2.64, 1.69, 0.73, 0.24]
    # The full list of physical property names.
    name_list_all = [
              "vec", "cohesive_energy", "bulk_modulus", "average_electronegativity",
              "electronegativity_difference", "average_atomic_size", "atomic_size_difference",
               "mixed_entropy","mixed_enthalpy", "Tm", "density", "elastic_modulus", "melting_enthalpy",
                "thermal_expansion", "thermal_conductivity", "specific_heat", "lattice_constant",
                "hardness", "electron_density", "G", "omega"
            ]
    # name_list are subset of the name_list_all
    name_list = ["atomic_size_difference", "average_atomic_size", "electronegativity_difference",
                              "mixed_entropy", "mixed_enthalpy", "average_electronegativity","electronegativity_difference"]
    # If the entries in composition_list are given as mass percentages, the parameter mole_fraction should be set to False.                       
    output = Wdk_made_feature(element_list, composition_list, name_list, mole_fraction=False).get_features()
    print(output, len(output))