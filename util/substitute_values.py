import numpy as np


def substitute_values(original_array, values_to_substitute, substitutions):
    substitutions_values_order = np.argsort(values_to_substitute)
    values_to_substitute_ordered = values_to_substitute[substitutions_values_order]
    substitutions_ordered = substitutions[substitutions_values_order]
    indices_to_substitute = np.nonzero(np.isin(original_array, values_to_substitute))
    values_at_substitutions = original_array[*indices_to_substitute]
    substitutions_assignments = np.searchsorted(values_to_substitute_ordered, values_at_substitutions)
    original_array[*indices_to_substitute] = substitutions_ordered[substitutions_assignments]