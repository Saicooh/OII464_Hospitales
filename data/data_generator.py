# /data_generator.py
import numpy as np

# Datos reales (2 operaciones: Anestesia y Cirugía)
BASE_DAY_SURGERIES_DATA = {
    1: {1: 30, 2: 60},
    2: {1: 40, 2: 60},
    3: {1: 35, 2: 80},
    4: {1: 65, 2: 190},
    5: {1: 70, 2: 190},
    6: {1: 75, 2: 190},
    7: {1: 80, 2: 150},
    8: {1: 70, 2: 110},
    9: {1: 35, 2: 80},
    10: {1: 30, 2: 80},
    11: {1: 60, 2: 110},
    12: {1: 80, 2: 110},
    13: {1: 65, 2: 210},
    14: {1: 40, 2: 70},
    15: {1: 70, 2: 160}
}

def generate_day_surgeries_data(job_ids, std_factor=0.0):
    """
    Generates surgery processing-time data with optional variability.
    """
    data = {}
    for j in job_ids:
        if j in BASE_DAY_SURGERIES_DATA:
            # Shallow copy is sufficient for {int: int} dicts (no nested mutables)
            data[j] = {op: val for op, val in BASE_DAY_SURGERIES_DATA[j].items()}
            
            if std_factor > 0:
                for op in [1, 2]:
                    base_val = BASE_DAY_SURGERIES_DATA[j][op]
                    std_val = std_factor * base_val
                    value = np.random.normal(base_val, std_val)
                    data[j][op] = max(1, round(value, 2))
        else:
            # Default value for undefined jobs (2 operations)
            data[j] = {1: 30, 2: 60}
    return data