from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from data.instance_loader import load_catalog, load_instance
from data.instance_model import canonical_digest
from tools.build_instance_catalog import build_catalog, required_replica_count

DIDACTIC_CATALOG = (('HOSP-DIDACT-03-01',
  'didactic/HOSP-DIDACT-03-01.yaml',
  3,
  '612f77fbad6d14029b7e15aacd14333f56af4914c67ec8c00d9d3e6f787a4d28'),
 ('HOSP-DIDACT-05-01',
  'didactic/HOSP-DIDACT-05-01.yaml',
  5,
  'ef386c09c3f5458cab006e8ecf0e4f5f7d8444696b9005438ec61d705b59c8e2'),
 ('HOSP-DIDACT-08-01',
  'didactic/HOSP-DIDACT-08-01.yaml',
  8,
  'ab3735ff8bc7d9620f739499788de7e2c336202fb686764b37fa0ca6a153a694'),
 ('HOSP-DIDACT-10-01',
  'didactic/HOSP-DIDACT-10-01.yaml',
  10,
  '401af56ee294776955ff842ffb0d0a730b7ca7e6c515fa145aeb34efe06ae7d6'))

STANDARD_15_CATALOG = (('HOSP-STD-15-01', 'standard/HOSP-STD-15-01.yaml'),
 ('HOSP-STD-15-02', 'standard/HOSP-STD-15-02.yaml'),
 ('HOSP-STD-15-03', 'standard/HOSP-STD-15-03.yaml'))

STANDARD_15_EVIDENCE = (('HOSP-STD-15-01',
  '961d3d2c3eef83296e641ca556961d3dcd1c5322dd32e37e43d498596b46e264',
  1501,
  'balanced load with moderate wait limits',
  0.72),
 ('HOSP-STD-15-02',
  'f315107711baef80dca74f77944f3ddf1643e5a1d35edd4016bbea6bbc495e03',
  1502,
  'surgery-heavy load with tighter wait limits',
  0.84),
 ('HOSP-STD-15-03',
  '1c59e8d211dafb58fed0e0f6cb6acc4ce21b77aa0b6c3ea02ea5f28743328d4b',
  1503,
  'mixed case load with wider wait limits',
  0.69))

STANDARD_20_CATALOG = (('HOSP-STD-20-01',
  'standard/HOSP-STD-20-01.yaml',
  2001,
  'balanced throughput with moderate wait limits',
  10.0,
  32.0,
  0.78),
 ('HOSP-STD-20-02',
  'standard/HOSP-STD-20-02.yaml',
  2002,
  'surgery-intensive throughput with tight wait limits',
  13.0,
  24.0,
  0.89),
 ('HOSP-STD-20-03',
  'standard/HOSP-STD-20-03.yaml',
  2003,
  'mixed-duration throughput with wider wait limits',
  8.0,
  44.0,
  0.73))

STANDARD_20_EVIDENCE = (('HOSP-STD-20-01',
  '4020267125f32fb837f503d4f8454f15a1c2eaaa040c81268ca390cc0cb3914b',
  2001,
  'balanced throughput with moderate wait limits',
  0.78),
 ('HOSP-STD-20-02',
  'f4d7eb68c49ca8471778e951b75929acfd33ba8efe201fc46198b6734b219e6c',
  2002,
  'surgery-intensive throughput with tight wait limits',
  0.89),
 ('HOSP-STD-20-03',
  'cf8acc0cbb3e3f835da745a06eee3784454065eef9e3b7278add7d92128fa270',
  2003,
  'mixed-duration throughput with wider wait limits',
  0.73))

STANDARD_25_CATALOG = (('HOSP-STD-25-01',
  'standard/HOSP-STD-25-01.yaml',
  2501,
  'balanced capacity with moderate wait limits',
  11.0,
  34.0,
  0.82),
 ('HOSP-STD-25-02',
  'standard/HOSP-STD-25-02.yaml',
  2502,
  'surgery-intensive capacity with tight wait limits',
  14.0,
  26.0,
  0.92),
 ('HOSP-STD-25-03',
  'standard/HOSP-STD-25-03.yaml',
  2503,
  'mixed-duration capacity with wider wait limits',
  8.5,
  48.0,
  0.76))

STANDARD_25_EVIDENCE = (('HOSP-STD-25-01',
  '0ae95d5e511979da55ccca585a1eb6136928c1e313c5f44fbda4766889092b8e',
  2501,
  'balanced capacity with moderate wait limits',
  0.82),
 ('HOSP-STD-25-02',
  '111e3dbf700a52182fb348e3992e1ca35fa16ea56d3fdab97ee16fdc5a83af70',
  2502,
  'surgery-intensive capacity with tight wait limits',
  0.92),
 ('HOSP-STD-25-03',
  '2d6e6d85cbec6219069261e929485a64f4ffcd10e2507b980fb2121f19a54ec7',
  2503,
  'mixed-duration capacity with wider wait limits',
  0.76))

STANDARD_30_CATALOG = (('HOSP-STD-30-01',
  'standard/HOSP-STD-30-01.yaml',
  '2f04176e649b56a7972631b0814e3932667b13b8da7df0c9bd837592c1400685',
  3001,
  'balanced scale with moderate wait limits',
  12.0,
  36.0,
  0.85),
 ('HOSP-STD-30-02',
  'standard/HOSP-STD-30-02.yaml',
  'ae1dbf281b4091bae360735878ddd7e10635385b4d42b9720ddcbcdc944e2c38',
  3002,
  'surgery-constrained peak with tight wait limits',
  15.0,
  28.0,
  0.95),
 ('HOSP-STD-30-03',
  'standard/HOSP-STD-30-03.yaml',
  '5c14c469ed592eb550dfc07626cff1cb866a9c8062b0ed4fd25d9fedcbdb1808',
  3003,
  'mixed-duration resilience with wide wait limits',
  9.0,
  52.0,
  0.79))

STANDARD_30_VALIDATION_EVIDENCE = {'HOSP-STD-30-01': {'duration_quantiles': {'p10': 9.5, 'p50': 22.0, 'p90': 51.0},
                    'dispersion': {'coefficient_of_variation': 0.54},
                    'rank_correlations': {'anesthesia_surgery': 0.83},
                    'workload_capacity_ratios': {'rooms': 0.85,
                                                 'anesthetists': 0.69,
                                                 'surgeons': 0.43}},
 'HOSP-STD-30-02': {'duration_quantiles': {'p10': 9.0, 'p50': 27.0, 'p90': 62.0},
                    'dispersion': {'coefficient_of_variation': 0.59},
                    'rank_correlations': {'anesthesia_surgery': 0.89},
                    'workload_capacity_ratios': {'rooms': 0.95,
                                                 'anesthetists': 0.78,
                                                 'surgeons': 0.52}},
 'HOSP-STD-30-03': {'duration_quantiles': {'p10': 8.5, 'p50': 23.0, 'p90': 56.0},
                    'dispersion': {'coefficient_of_variation': 0.64},
                    'rank_correlations': {'anesthesia_surgery': 0.86},
                    'workload_capacity_ratios': {'rooms': 0.79,
                                                 'anesthetists': 0.58,
                                                 'surgeons': 0.36}}}

HOSPITAL_12ROOMS = tuple(f"OR-{number}" for number in range(1, 13))

HOSPITAL_12ROOMS_15_CATALOG = (('HOSP-12R-15-01',
  'hospital_12rooms/HOSP-12R-15-01.yaml',
  121501,
  'twelve-room balanced teaching load',
  10.0,
  34.0,
  0.58),
 ('HOSP-12R-15-02',
  'hospital_12rooms/HOSP-12R-15-02.yaml',
  121502,
  'twelve-room surgery-intensive teaching load',
  13.0,
  26.0,
  0.71),
 ('HOSP-12R-15-03',
  'hospital_12rooms/HOSP-12R-15-03.yaml',
  121503,
  'twelve-room mixed-duration teaching load',
  8.0,
  46.0,
  0.49))

HOSPITAL_12ROOMS_15_VALIDATION_EVIDENCE = {'HOSP-12R-15-01': {'duration_quantiles': {'p10': 8.0, 'p50': 20.0, 'p90': 46.0},
                    'dispersion': {'coefficient_of_variation': 0.5},
                    'rank_correlations': {'anesthesia_surgery': 0.8},
                    'workload_capacity_ratios': {'rooms': 0.58,
                                                 'anesthetists': 0.39,
                                                 'surgeons': 0.3}},
 'HOSP-12R-15-02': {'duration_quantiles': {'p10': 8.5, 'p50': 24.0, 'p90': 58.0},
                    'dispersion': {'coefficient_of_variation': 0.57},
                    'rank_correlations': {'anesthesia_surgery': 0.88},
                    'workload_capacity_ratios': {'rooms': 0.71,
                                                 'anesthetists': 0.45,
                                                 'surgeons': 0.38}},
 'HOSP-12R-15-03': {'duration_quantiles': {'p10': 7.5, 'p50': 21.0, 'p90': 54.0},
                    'dispersion': {'coefficient_of_variation': 0.63},
                    'rank_correlations': {'anesthesia_surgery': 0.84},
                    'workload_capacity_ratios': {'rooms': 0.49,
                                                 'anesthetists': 0.34,
                                                 'surgeons': 0.26}}}

HOSPITAL_12ROOMS_15_DIGESTS = {'HOSP-12R-15-01': 'c9a3fe0727620927a786a7d0f9bc85a01bb368d4af9dfdc2cd9d868167ee8dc2',
 'HOSP-12R-15-02': 'dcf9f4332d06128fcf69ab726739e9a138ab65757e506e3a3a0b403f47368bc8',
 'HOSP-12R-15-03': 'ae01a001e0b9f765e49317464c2a151fb6d11899af1d85c064f9befed2dbd034'}

HOSPITAL_12ROOMS_20_CATALOG = (('HOSP-12R-20-01',
  'hospital_12rooms/HOSP-12R-20-01.yaml',
  122001,
  'twelve-room balanced twenty-job load',
  11.0,
  36.0,
  0.64),
 ('HOSP-12R-20-02',
  'hospital_12rooms/HOSP-12R-20-02.yaml',
  122002,
  'twelve-room surgery-intensive twenty-job load',
  14.0,
  28.0,
  0.78),
 ('HOSP-12R-20-03',
  'hospital_12rooms/HOSP-12R-20-03.yaml',
  122003,
  'twelve-room mixed-duration twenty-job load',
  8.5,
  50.0,
  0.56))

HOSPITAL_12ROOMS_20_VALIDATION_EVIDENCE = {'HOSP-12R-20-01': {'duration_quantiles': {'p10': 8.5, 'p50': 20.0, 'p90': 48.0},
                    'dispersion': {'coefficient_of_variation': 0.52},
                    'rank_correlations': {'anesthesia_surgery': 0.82},
                    'workload_capacity_ratios': {'rooms': 0.64,
                                                 'anesthetists': 0.46,
                                                 'surgeons': 0.31}},
 'HOSP-12R-20-02': {'duration_quantiles': {'p10': 9.0, 'p50': 24.0, 'p90': 58.0},
                    'dispersion': {'coefficient_of_variation': 0.58},
                    'rank_correlations': {'anesthesia_surgery': 0.89},
                    'workload_capacity_ratios': {'rooms': 0.78,
                                                 'anesthetists': 0.54,
                                                 'surgeons': 0.38}},
 'HOSP-12R-20-03': {'duration_quantiles': {'p10': 8.0, 'p50': 21.0, 'p90': 53.0},
                    'dispersion': {'coefficient_of_variation': 0.65},
                    'rank_correlations': {'anesthesia_surgery': 0.85},
                    'workload_capacity_ratios': {'rooms': 0.56,
                                                 'anesthetists': 0.4,
                                                 'surgeons': 0.28}}}

HOSPITAL_12ROOMS_20_DIGESTS = {'HOSP-12R-20-01': '63dc85069f3bd27fa244a807cd01877e8cdbc46d7576602d414ea3816c75f4c4',
 'HOSP-12R-20-02': '1c9d08b4a131dda436b5b6336057aaf930ae7d6c7ddc5d5ba4fb4e0248add40c',
 'HOSP-12R-20-03': 'cf0560d2d3259c752c90c671313ff3b3ec3856618e34b763fc8b1c1f7aed8b23'}

HOSPITAL_12ROOMS_25_CATALOG = (('HOSP-12R-25-01',
  'hospital_12rooms/HOSP-12R-25-01.yaml',
  122501,
  'twelve-room balanced twenty-five-job load',
  12.0,
  38.0,
  0.69),
 ('HOSP-12R-25-02',
  'hospital_12rooms/HOSP-12R-25-02.yaml',
  122502,
  'twelve-room surgery-intensive twenty-five-job load',
  15.0,
  30.0,
  0.84),
 ('HOSP-12R-25-03',
  'hospital_12rooms/HOSP-12R-25-03.yaml',
  122503,
  'twelve-room mixed-duration twenty-five-job load',
  9.0,
  54.0,
  0.61))

HOSPITAL_12ROOMS_25_FROZEN = (('HOSP-12R-25-01',
  '8d3ac1babb71ed42fa8807f4bc0c5ff6e8f4e08c133a1829a09781930098b810',
  122501,
  {'duration_quantiles': {'p10': 9.0, 'p50': 22.0, 'p90': 50.0},
   'dispersion': {'coefficient_of_variation': 0.54},
   'rank_correlations': {'anesthesia_surgery': 0.83},
   'workload_capacity_ratios': {'rooms': 0.69, 'anesthetists': 0.51, 'surgeons': 0.32}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18')),
 ('HOSP-12R-25-02',
  'a769f26651a981b6ee6392d5ce82a500396490de89d70e8c95cced203d4aa6d7',
  122502,
  {'duration_quantiles': {'p10': 9.5, 'p50': 26.0, 'p90': 61.0},
   'dispersion': {'coefficient_of_variation': 0.6},
   'rank_correlations': {'anesthesia_surgery': 0.9},
   'workload_capacity_ratios': {'rooms': 0.84, 'anesthetists': 0.61, 'surgeons': 0.39}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16')),
 ('HOSP-12R-25-03',
  'be174f888e3abd728cbe046e615c3b603b72e1695c382a357bb23f75ce98629f',
  122503,
  {'duration_quantiles': {'p10': 8.0, 'p50': 22.0, 'p90': 56.0},
   'dispersion': {'coefficient_of_variation': 0.67},
   'rank_correlations': {'anesthesia_surgery': 0.86},
   'workload_capacity_ratios': {'rooms': 0.61, 'anesthetists': 0.46, 'surgeons': 0.29}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20')))

HOSPITAL_12ROOMS_30_CATALOG = (('HOSP-12R-30-01',
  'hospital_12rooms/HOSP-12R-30-01.yaml',
  123001,
  'twelve-room balanced thirty-job load',
  12.5,
  40.0,
  0.74),
 ('HOSP-12R-30-02',
  'hospital_12rooms/HOSP-12R-30-02.yaml',
  123002,
  'twelve-room surgery-intensive thirty-job load',
  15.5,
  31.0,
  0.89),
 ('HOSP-12R-30-03',
  'hospital_12rooms/HOSP-12R-30-03.yaml',
  123003,
  'twelve-room mixed-duration thirty-job load',
  9.5,
  58.0,
  0.66))

HOSPITAL_12ROOMS_30_DIGESTS = {'HOSP-12R-30-01': '9b21cb7ddfcf1b5aa57a239c2c3088f0638b2a6c4ff782e64ed90b917e47fd14',
 'HOSP-12R-30-02': 'c38951d7f55ae8aef6c321cc49eea638fb873940255b6ddbe882fbdf978acae0',
 'HOSP-12R-30-03': '198955d077d18cd2dda47531d07a1768c9e73b237b07d8a40f5a8f12366487e7'}

HOSPITAL_12ROOMS_30_VALIDATION_EVIDENCE = {'HOSP-12R-30-01': {'duration_quantiles': {'p10': 9.5, 'p50': 23.0, 'p90': 53.0},
                    'dispersion': {'coefficient_of_variation': 0.56},
                    'rank_correlations': {'anesthesia_surgery': 0.84},
                    'workload_capacity_ratios': {'rooms': 0.74,
                                                 'anesthetists': 0.56,
                                                 'surgeons': 0.32}},
 'HOSP-12R-30-02': {'duration_quantiles': {'p10': 10.0, 'p50': 28.0, 'p90': 65.0},
                    'dispersion': {'coefficient_of_variation': 0.61},
                    'rank_correlations': {'anesthesia_surgery': 0.91},
                    'workload_capacity_ratios': {'rooms': 0.89,
                                                 'anesthetists': 0.67,
                                                 'surgeons': 0.37}},
 'HOSP-12R-30-03': {'duration_quantiles': {'p10': 8.5, 'p50': 24.0, 'p90': 60.0},
                    'dispersion': {'coefficient_of_variation': 0.69},
                    'rank_correlations': {'anesthesia_surgery': 0.87},
                    'workload_capacity_ratios': {'rooms': 0.66,
                                                 'anesthetists': 0.51,
                                                 'surgeons': 0.29}}}

HOSPITAL_12ROOMS_30_PERSONNEL = {'HOSP-12R-30-01': (('AN-1',
                     'AN-2',
                     'AN-3',
                     'AN-4',
                     'AN-5',
                     'AN-6',
                     'AN-7',
                     'AN-8',
                     'AN-9',
                     'AN-10'),
                    ('SU-1',
                     'SU-2',
                     'SU-3',
                     'SU-4',
                     'SU-5',
                     'SU-6',
                     'SU-7',
                     'SU-8',
                     'SU-9',
                     'SU-10',
                     'SU-11',
                     'SU-12',
                     'SU-13',
                     'SU-14',
                     'SU-15',
                     'SU-16',
                     'SU-17',
                     'SU-18',
                     'SU-19',
                     'SU-20',
                     'SU-21',
                     'SU-22')),
 'HOSP-12R-30-02': (('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9'),
                    ('SU-1',
                     'SU-2',
                     'SU-3',
                     'SU-4',
                     'SU-5',
                     'SU-6',
                     'SU-7',
                     'SU-8',
                     'SU-9',
                     'SU-10',
                     'SU-11',
                     'SU-12',
                     'SU-13',
                     'SU-14',
                     'SU-15',
                     'SU-16',
                     'SU-17',
                     'SU-18',
                     'SU-19',
                     'SU-20')),
 'HOSP-12R-30-03': (('AN-1',
                     'AN-2',
                     'AN-3',
                     'AN-4',
                     'AN-5',
                     'AN-6',
                     'AN-7',
                     'AN-8',
                     'AN-9',
                     'AN-10',
                     'AN-11'),
                    ('SU-1',
                     'SU-2',
                     'SU-3',
                     'SU-4',
                     'SU-5',
                     'SU-6',
                     'SU-7',
                     'SU-8',
                     'SU-9',
                     'SU-10',
                     'SU-11',
                     'SU-12',
                     'SU-13',
                     'SU-14',
                     'SU-15',
                     'SU-16',
                     'SU-17',
                     'SU-18',
                     'SU-19',
                     'SU-20',
                     'SU-21',
                     'SU-22',
                     'SU-23',
                     'SU-24'))}

HOSPITAL_12ROOMS_40_FROZEN = (('HOSP-12R-40-01',
  'hospital_12rooms/HOSP-12R-40-01.yaml',
  '6f1eeaaf74b1fe61027e67f9b4ef976ac370a838e9b97ac64c7dab3ccde5977e',
  124001,
  'twelve-room balanced forty-job load',
  13.0,
  42.0,
  {'duration_quantiles': {'p10': 10.0, 'p50': 24.0, 'p90': 55.0},
   'dispersion': {'coefficient_of_variation': 0.57},
   'rank_correlations': {'anesthesia_surgery': 0.85},
   'workload_capacity_ratios': {'rooms': 0.79, 'anesthetists': 0.61, 'surgeons': 0.33}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10', 'AN-11'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25')),
 ('HOSP-12R-40-02',
  'hospital_12rooms/HOSP-12R-40-02.yaml',
  '243a65ec04c1d24075dd785d4dbdbd518a93a6b05b718dd4988aa9e742681819',
  124002,
  'twelve-room surgery-intensive forty-job load',
  16.0,
  32.0,
  {'duration_quantiles': {'p10': 10.5, 'p50': 29.5, 'p90': 68.0},
   'dispersion': {'coefficient_of_variation': 0.62},
   'rank_correlations': {'anesthesia_surgery': 0.92},
   'workload_capacity_ratios': {'rooms': 0.94, 'anesthetists': 0.72, 'surgeons': 0.38}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23')),
 ('HOSP-12R-40-03',
  'hospital_12rooms/HOSP-12R-40-03.yaml',
  '3952eb2fb038ad6b455528df57fe6b914f5dc3ea6f323dee9dff8668f57314ce',
  124003,
  'twelve-room mixed-duration forty-job load',
  10.0,
  60.0,
  {'duration_quantiles': {'p10': 9.0, 'p50': 25.0, 'p90': 62.0},
   'dispersion': {'coefficient_of_variation': 0.7},
   'rank_correlations': {'anesthesia_surgery': 0.88},
   'workload_capacity_ratios': {'rooms': 0.71, 'anesthetists': 0.61, 'surgeons': 0.3}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10', 'AN-11'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25',
   'SU-26',
   'SU-27')))

HOSPITAL_12ROOMS_50_FROZEN = (('HOSP-12R-50-01',
  'hospital_12rooms/HOSP-12R-50-01.yaml',
  '5576b9dfd9f606e8710e272d5852999f5cc9035a9dc37684e63d7ce40db515a6',
  125001,
  'twelve-room balanced fifty-job load',
  13.0,
  42.0,
  {'duration_quantiles': {'p10': 10.5, 'p50': 25.0, 'p90': 57.0},
   'dispersion': {'coefficient_of_variation': 0.58},
   'rank_correlations': {'anesthesia_surgery': 0.86},
   'workload_capacity_ratios': {'rooms': 0.84, 'anesthetists': 0.65, 'surgeons': 0.32}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10', 'AN-11'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25',
   'SU-26',
   'SU-27')),
 ('HOSP-12R-50-02',
  'hospital_12rooms/HOSP-12R-50-02.yaml',
  'ec69043554d87f4a8704249db6c998666565ed391a717f3a5bafa22c227dc7ae',
  125002,
  'twelve-room surgery-intensive fifty-job load',
  16.0,
  32.0,
  {'duration_quantiles': {'p10': 11.0, 'p50': 30.5, 'p90': 70.0},
   'dispersion': {'coefficient_of_variation': 0.63},
   'rank_correlations': {'anesthesia_surgery': 0.93},
   'workload_capacity_ratios': {'rooms': 0.98, 'anesthetists': 0.75, 'surgeons': 0.36}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25')),
 ('HOSP-12R-50-03',
  'hospital_12rooms/HOSP-12R-50-03.yaml',
  '4fa310f19084b489382a80f64abe04cdaa27da0a51c026476bd273383fb36835',
  125003,
  'twelve-room mixed-duration fifty-job load',
  10.0,
  60.0,
  {'duration_quantiles': {'p10': 9.5, 'p50': 26.0, 'p90': 64.0},
   'dispersion': {'coefficient_of_variation': 0.71},
   'rank_correlations': {'anesthesia_surgery': 0.89},
   'workload_capacity_ratios': {'rooms': 0.76, 'anesthetists': 0.65, 'surgeons': 0.3}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10', 'AN-11'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25',
   'SU-26',
   'SU-27',
   'SU-28',
   'SU-29')))

HOSPITAL_12ROOMS_60_FROZEN = (('HOSP-12R-60-01',
  'hospital_12rooms/HOSP-12R-60-01.yaml',
  '3ab2c1ed1c0607b5ed8e5403dae81970fca0d046c6052c9d54753f1d73ea3a93',
  126001,
  'twelve-room balanced sixty-job load',
  13.0,
  42.0,
  {'duration_quantiles': {'p10': 10.0, 'p50': 26.0, 'p90': 61.0},
   'dispersion': {'coefficient_of_variation': 0.59},
   'rank_correlations': {'anesthesia_surgery': 0.87},
   'workload_capacity_ratios': {'rooms': 0.91, 'anesthetists': 0.7, 'surgeons': 0.34}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10', 'AN-11'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25',
   'SU-26',
   'SU-27')),
 ('HOSP-12R-60-02',
  'hospital_12rooms/HOSP-12R-60-02.yaml',
  'fb22d3a047b4ef1de2165a0480d64964c98d7b734df99d130ee33ead80e9b570',
  126002,
  'twelve-room surgery-intensive sixty-job load',
  16.0,
  32.0,
  {'duration_quantiles': {'p10': 11.0, 'p50': 31.0, 'p90': 74.0},
   'dispersion': {'coefficient_of_variation': 0.64},
   'rank_correlations': {'anesthesia_surgery': 0.94},
   'workload_capacity_ratios': {'rooms': 1.08, 'anesthetists': 0.82, 'surgeons': 0.39}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25')),
 ('HOSP-12R-60-03',
  'hospital_12rooms/HOSP-12R-60-03.yaml',
  '73a1814961d7b51539e4ef3e2cfc4b39049325c57ddeb8516c62df0478c8f5a4',
  126003,
  'twelve-room mixed-duration sixty-job load',
  10.0,
  60.0,
  {'duration_quantiles': {'p10': 9.0, 'p50': 27.0, 'p90': 69.0},
   'dispersion': {'coefficient_of_variation': 0.72},
   'rank_correlations': {'anesthesia_surgery': 0.9},
   'workload_capacity_ratios': {'rooms': 0.83, 'anesthetists': 0.72, 'surgeons': 0.33}},
  ('AN-1', 'AN-2', 'AN-3', 'AN-4', 'AN-5', 'AN-6', 'AN-7', 'AN-8', 'AN-9', 'AN-10', 'AN-11'),
  ('SU-1',
   'SU-2',
   'SU-3',
   'SU-4',
   'SU-5',
   'SU-6',
   'SU-7',
   'SU-8',
   'SU-9',
   'SU-10',
   'SU-11',
   'SU-12',
   'SU-13',
   'SU-14',
   'SU-15',
   'SU-16',
   'SU-17',
   'SU-18',
   'SU-19',
   'SU-20',
   'SU-21',
   'SU-22',
   'SU-23',
   'SU-24',
   'SU-25',
   'SU-26',
   'SU-27',
   'SU-28',
   'SU-29')))

def test_catalog_uses_approved_increased_personnel_and_eligibility():
    root = Path(__file__).parents[2] / "instances"
    expected_counts = {
        **dict(zip(
            [f"HOSP-DIDACT-{size:02d}-01" for size in (3, 5, 8, 10)],
            ((3, 4), (3, 5), (4, 6), (4, 7)),
            strict=True,
        )),
        **dict(zip(
            [f"HOSP-STD-{size}-{replica:02d}" for size in (15, 20, 25, 30) for replica in range(1, 4)],
            (
                (5, 8), (5, 7), (6, 10),
                (6, 10), (6, 9), (7, 12),
                (7, 13), (7, 12), (8, 15),
                (8, 16), (8, 15), (9, 18),
            ),
            strict=True,
        )),
        **dict(zip(
            [f"HOSP-12R-{size}-{replica:02d}" for size in (15, 20, 25, 30, 40, 50, 60) for replica in range(1, 4)],
            (
                (7, 12), (6, 10), (8, 14),
                (8, 15), (7, 13), (9, 17),
                (9, 18), (8, 16), (10, 20),
                (10, 22), (9, 20), (11, 24),
                (11, 25), (10, 23), (11, 27),
                (11, 27), (10, 25), (11, 29),
                (11, 27), (10, 25), (11, 29),
            ),
            strict=True,
        )),
    }

    documents = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/*.yaml"))
    ]

    assert {document["instance_id"] for document in documents} == set(expected_counts)
    for document in documents:
        identifier = document["instance_id"]
        anesthetists, surgeons = expected_counts[identifier]
        personnel = document["resources"]["personnel"]
        assert personnel["1"] == [f"AN-{index}" for index in range(1, anesthetists + 1)]
        assert personnel["2"] == [f"SU-{index}" for index in range(1, surgeons + 1)]
        minimum_eligible = (
            2
            if document["family"] == "didactic"
            else 4
            if len(document["jobs"]) >= 50
            else 3
        )
        assert all(
            len(operation["eligible_personnel"]) >= minimum_eligible
            for job in document["jobs"]
            for operation in job["operations"]
        )


def test_materialized_didactic_catalog_has_honest_fixed_metadata():
    root = Path(__file__).parents[2] / "instances"
    contexts = load_catalog(root)
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    entries = metadata["instances"]
    didactic_entries = entries[:4]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, _, _ in DIDACTIC_CATALOG
    ]

    assert metadata["catalog_status"] == "partial"
    assert [entry["instance_id"] for entry in didactic_entries] == [item[0] for item in DIDACTIC_CATALOG]
    assert [entry["file"] for entry in didactic_entries] == [item[1] for item in DIDACTIC_CATALOG]
    assert [(context.instance_id, len(context.jobs)) for context in contexts[:4]] == [
        (identifier, job_count) for identifier, _, job_count, _ in DIDACTIC_CATALOG
    ]
    expected_digests = [item[3] for item in DIDACTIC_CATALOG]
    assert [context.digest for context in contexts[:4]] == expected_digests
    assert [entry["digest"] for entry in didactic_entries] == expected_digests
    assert all(document["digest"] == canonical_digest(document) for document in documents)
    assert all(document["classification"] == "fully synthetic instance" for document in documents)
    assert all(document["bounds"]["status"] == "pending" for document in documents)
    assert all(
        entry[field] == document[field]
        for entry, document in zip(didactic_entries, documents, strict=True)
        for field in (
            "provenance", "generation", "dimensions", "resources", "classification",
            "bounds", "validation", "digest",
        )
    )
    assert all("course-designed" in document["provenance"]["source"] for document in documents)


def test_standard_15_catalog_admits_three_intentionally_distinct_instances():
    root = Path(__file__).parents[2] / "instances"
    standard_root = root / "standard"
    expected_files = [Path(item[1]).name for item in STANDARD_15_CATALOG]

    assert sorted(path.name for path in standard_root.glob("HOSP-STD-15-*.yaml")) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    standard_entries = entries[4:7]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path in STANDARD_15_CATALOG
    ]

    assert metadata["catalog_status"] == "partial"
    assert [entry["instance_id"] for entry in standard_entries] == [
        item[0] for item in STANDARD_15_CATALOG
    ]
    assert [entry["file"] for entry in standard_entries] == [
        item[1] for item in STANDARD_15_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[:7]] == [3, 5, 8, 10, 15, 15, 15]
    assert len({context.digest for context in contexts[:7]}) == 7
    assert [document["digest"] for document in documents] == [
        context.digest for context in contexts[4:7]
    ]
    assert all(
        document["digest"] == canonical_digest(document) for document in documents
    )
    assert all(
        entry[field] == document[field]
        for entry, document in zip(standard_entries, documents, strict=True)
        for field in (
            "provenance",
            "generation",
            "dimensions",
            "resources",
            "classification",
            "bounds",
            "validation",
            "digest",
        )
    )
    intentional_configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
        )
        for document in documents
    }
    assert len(intentional_configurations) == 3


def test_standard_20_catalog_extends_partial_catalog_with_distinct_instances():
    root = Path(__file__).parents[2] / "instances"
    expected_files = [Path(item[1]).name for item in STANDARD_20_CATALOG]

    assert sorted(
        path.name for path in (root / "standard").glob("HOSP-STD-20-*.yaml")
    ) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    standard_entries = entries[7:10]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, *_ in STANDARD_20_CATALOG
    ]

    assert metadata["catalog_status"] == "partial"
    assert [entry["instance_id"] for entry in standard_entries] == [
        item[0] for item in STANDARD_20_CATALOG
    ]
    assert [entry["file"] for entry in standard_entries] == [
        item[1] for item in STANDARD_20_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[:10]] == [
        3, 5, 8, 10, 15, 15, 15, 20, 20, 20
    ]
    assert len({context.digest for context in contexts[:10]}) == 10
    assert [document["digest"] for document in documents] == [
        context.digest for context in contexts[7:10]
    ]
    assert all(document["digest"] == canonical_digest(document) for document in documents)
    assert all(
        entry[field] == document[field]
        for entry, document in zip(standard_entries, documents, strict=True)
        for field in (
            "provenance",
            "generation",
            "dimensions",
            "resources",
            "classification",
            "bounds",
            "validation",
            "digest",
        )
    )
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in documents
    }
    assert configurations == {
        (seed, congestion_band, first_duration, first_wait, room_ratio)
        for (
            _, _, seed, congestion_band, first_duration, first_wait, room_ratio
        ) in STANDARD_20_CATALOG
    }


def test_standard_25_catalog_extends_partial_catalog_with_distinct_instances():
    root = Path(__file__).parents[2] / "instances"
    expected_files = [Path(item[1]).name for item in STANDARD_25_CATALOG]

    assert sorted(
        path.name for path in (root / "standard").glob("HOSP-STD-25-*.yaml")
    ) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    standard_entries = entries[10:13]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, *_ in STANDARD_25_CATALOG
    ]

    assert metadata["catalog_status"] == "partial"
    assert [entry["instance_id"] for entry in standard_entries] == [
        item[0] for item in STANDARD_25_CATALOG
    ]
    assert [entry["file"] for entry in standard_entries] == [
        item[1] for item in STANDARD_25_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[:13]] == [
        3, 5, 8, 10, 15, 15, 15, 20, 20, 20, 25, 25, 25
    ]
    assert len({context.digest for context in contexts[:13]}) == 13
    assert [document["digest"] for document in documents] == [
        context.digest for context in contexts[10:13]
    ]
    assert all(document["digest"] == canonical_digest(document) for document in documents)
    assert all(
        entry[field] == document[field]
        for entry, document in zip(standard_entries, documents, strict=True)
        for field in (
            "provenance",
            "generation",
            "dimensions",
            "resources",
            "classification",
            "bounds",
            "validation",
            "digest",
        )
    )
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in documents
    }
    assert configurations == {
        (seed, congestion_band, first_duration, first_wait, room_ratio)
        for (
            _, _, seed, congestion_band, first_duration, first_wait, room_ratio
        ) in STANDARD_25_CATALOG
    }


def test_standard_30_catalog_extends_partial_catalog_with_distinct_instances():
    root = Path(__file__).parents[2] / "instances"
    expected_files = [Path(item[1]).name for item in STANDARD_30_CATALOG]

    assert sorted(
        path.name for path in (root / "standard").glob("HOSP-STD-30-*.yaml")
    ) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    standard_entries = entries[13:16]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, *_ in STANDARD_30_CATALOG
    ]

    assert metadata["catalog_status"] == "partial"
    assert [entry["instance_id"] for entry in standard_entries] == [
        item[0] for item in STANDARD_30_CATALOG
    ]
    assert [entry["file"] for entry in standard_entries] == [
        item[1] for item in STANDARD_30_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[:16]] == [
        3, 5, 8, 10, 15, 15, 15, 20, 20, 20, 25, 25, 25, 30, 30, 30
    ]
    assert len({context.digest for context in contexts[:16]}) == 16
    assert [document["digest"] for document in documents] == [
        context.digest for context in contexts[13:16]
    ] == [entry["digest"] for entry in standard_entries]
    assert all(document["digest"] == canonical_digest(document) for document in documents)
    assert all(
        entry[field] == document[field]
        for entry, document in zip(standard_entries, documents, strict=True)
        for field in (
            "provenance",
            "generation",
            "dimensions",
            "resources",
            "classification",
            "bounds",
            "validation",
            "digest",
        )
    )
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in documents
    }
    assert configurations == {
        (seed, congestion_band, first_duration, first_wait, room_ratio)
        for (
            _, _, _, seed, congestion_band, first_duration, first_wait, room_ratio
        ) in STANDARD_30_CATALOG
    }


def test_hospital_12rooms_15_catalog_owns_twelve_room_resources_and_eligibility():
    root = Path(__file__).parents[2] / "instances"
    expected_files = [Path(item[1]).name for item in HOSPITAL_12ROOMS_15_CATALOG]

    assert sorted(
        path.name
        for path in (root / "hospital_12rooms").glob("HOSP-12R-15-*.yaml")
    ) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    hospital_entries = entries[16:19]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, *_ in HOSPITAL_12ROOMS_15_CATALOG
    ]
    expected_rooms = list(HOSPITAL_12ROOMS)

    assert metadata["catalog_status"] == "partial"
    assert len(entries[16:19]) == 3
    assert [entry["instance_id"] for entry in hospital_entries] == [
        item[0] for item in HOSPITAL_12ROOMS_15_CATALOG
    ]
    assert [entry["file"] for entry in hospital_entries] == [
        item[1] for item in HOSPITAL_12ROOMS_15_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[16:19]] == [15, 15, 15]
    assert all(document["family"] == "hospital_12rooms" for document in documents)
    assert all(document["resources"]["rooms"] == expected_rooms for document in documents)
    assert all(list(context.rooms) == expected_rooms for context in contexts[16:19])
    assert all(
        operation["eligible_rooms"] == expected_rooms
        for document in documents
        for job in document["jobs"]
        for operation in job["operations"]
    )
    assert all(
        entry[field] == document[field]
        for entry, document in zip(hospital_entries, documents, strict=True)
        for field in (
            "provenance",
            "generation",
            "dimensions",
            "resources",
            "classification",
            "bounds",
            "validation",
            "digest",
        )
    )
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in documents
    }
    assert configurations == {
        (seed, congestion_band, first_duration, first_wait, room_ratio)
        for (
            _, _, seed, congestion_band, first_duration, first_wait, room_ratio
        ) in HOSPITAL_12ROOMS_15_CATALOG
    }


@pytest.mark.parametrize(
    ("identifier", "seed"),
    [(identifier, seed) for identifier, _, seed, *_ in HOSPITAL_12ROOMS_15_CATALOG],
)
def test_hospital_12rooms_15_evidence_digest_and_room_semantics_are_frozen(
    identifier, seed
):
    path = (
        Path(__file__).parents[2]
        / "instances"
        / "hospital_12rooms"
        / f"{identifier}.yaml"
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)
    digest = HOSPITAL_12ROOMS_15_DIGESTS[identifier]

    assert (context.instance_id, context.generation_seed, context.digest) == (
        identifier,
        seed,
        digest,
    )
    assert document["family"] == "hospital_12rooms"
    assert document["classification"] == "fully synthetic instance"
    assert canonical_digest(document) == document["digest"] == digest
    assert document["validation"]["evidence"] == (
        HOSPITAL_12ROOMS_15_VALIDATION_EVIDENCE[identifier]
    )
    assert context.rooms == HOSPITAL_12ROOMS
    assert document["generation"]["profile"]["room_eligibility"] == (
        "all declared rooms"
    )
    assert all(
        operation.eligible_rooms == HOSPITAL_12ROOMS
        for job in context.jobs
        for operation in job.operations
    )


def test_hospital_12rooms_20_catalog_owns_twelve_room_resources_and_eligibility():
    root = Path(__file__).parents[2] / "instances"
    expected_files = [Path(item[1]).name for item in HOSPITAL_12ROOMS_20_CATALOG]

    assert sorted(
        path.name
        for path in (root / "hospital_12rooms").glob("HOSP-12R-20-*.yaml")
    ) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    hospital_entries = entries[19:22]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, *_ in HOSPITAL_12ROOMS_20_CATALOG
    ]
    expected_rooms = list(HOSPITAL_12ROOMS)

    assert metadata["catalog_status"] == "partial"
    assert metadata["instance_count"] == len(entries)
    assert len(entries[19:22]) == 3
    assert [entry["instance_id"] for entry in hospital_entries] == [
        item[0] for item in HOSPITAL_12ROOMS_20_CATALOG
    ]
    assert [entry["file"] for entry in hospital_entries] == [
        item[1] for item in HOSPITAL_12ROOMS_20_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[19:22]] == [20, 20, 20]
    assert all(document["family"] == "hospital_12rooms" for document in documents)
    assert all(document["resources"]["rooms"] == expected_rooms for document in documents)
    assert all(list(context.rooms) == expected_rooms for context in contexts[19:22])
    assert all(
        operation["eligible_rooms"] == expected_rooms
        for document in documents
        for job in document["jobs"]
        for operation in job["operations"]
    )
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in documents
    }
    assert configurations == {
        (seed, congestion_band, first_duration, first_wait, room_ratio)
        for (
            _, _, seed, congestion_band, first_duration, first_wait, room_ratio
        ) in HOSPITAL_12ROOMS_20_CATALOG
    }


@pytest.mark.parametrize(
    ("identifier", "seed"),
    [(identifier, seed) for identifier, _, seed, *_ in HOSPITAL_12ROOMS_20_CATALOG],
)
def test_hospital_12rooms_20_evidence_digest_and_room_semantics_are_frozen(
    identifier, seed
):
    root = Path(__file__).parents[2] / "instances"
    path = root / "hospital_12rooms" / f"{identifier}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)
    digest = HOSPITAL_12ROOMS_20_DIGESTS[identifier]
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in metadata["instances"] if item["instance_id"] == identifier)

    assert (context.instance_id, context.generation_seed, context.digest) == (
        identifier,
        seed,
        digest,
    )
    assert canonical_digest(document) == document["digest"] == entry["digest"] == digest
    assert document["validation"]["evidence"] == (
        HOSPITAL_12ROOMS_20_VALIDATION_EVIDENCE[identifier]
    )
    assert all(
        entry[field] == document[field]
        for field in (
            "provenance",
            "generation",
            "dimensions",
            "resources",
            "classification",
            "bounds",
            "validation",
            "digest",
        )
    )
    assert context.rooms == HOSPITAL_12ROOMS
    assert document["generation"]["profile"]["room_eligibility"] == "all declared rooms"
    assert all(
        operation.eligible_rooms == HOSPITAL_12ROOMS
        for job in context.jobs
        for operation in job.operations
    )


def test_hospital_12rooms_25_catalog_owns_twelve_room_resources_and_eligibility():
    root = Path(__file__).parents[2] / "instances"
    expected_files = [Path(item[1]).name for item in HOSPITAL_12ROOMS_25_CATALOG]

    assert sorted(
        path.name
        for path in (root / "hospital_12rooms").glob("HOSP-12R-25-*.yaml")
    ) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    hospital_entries = entries[22:25]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, *_ in HOSPITAL_12ROOMS_25_CATALOG
    ]
    expected_rooms = list(HOSPITAL_12ROOMS)

    assert metadata["catalog_status"] == "partial"
    assert metadata["instance_count"] == len(entries)
    assert len(entries[22:25]) == 3
    assert [entry["instance_id"] for entry in hospital_entries] == [
        item[0] for item in HOSPITAL_12ROOMS_25_CATALOG
    ]
    assert [entry["file"] for entry in hospital_entries] == [
        item[1] for item in HOSPITAL_12ROOMS_25_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[22:25]] == [25, 25, 25]
    assert all(document["family"] == "hospital_12rooms" for document in documents)
    assert all(document["resources"]["rooms"] == expected_rooms for document in documents)
    assert all(list(context.rooms) == expected_rooms for context in contexts[22:25])
    assert all(
        operation["eligible_rooms"] == expected_rooms
        for document in documents
        for job in document["jobs"]
        for operation in job["operations"]
    )
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in documents
    }
    assert configurations == {
        (seed, congestion_band, first_duration, first_wait, room_ratio)
        for (
            _, _, seed, congestion_band, first_duration, first_wait, room_ratio
        ) in HOSPITAL_12ROOMS_25_CATALOG
    }


@pytest.mark.parametrize(
    ("identifier", "digest", "seed", "evidence", "anesthetists", "surgeons"),
    HOSPITAL_12ROOMS_25_FROZEN,
)
def test_hospital_12rooms_25_full_evidence_resources_and_digest_are_frozen(
    identifier, digest, seed, evidence, anesthetists, surgeons
):
    root = Path(__file__).parents[2] / "instances"
    path = root / "hospital_12rooms" / f"{identifier}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in metadata["instances"] if item["instance_id"] == identifier)
    resources = {
        "rooms": list(HOSPITAL_12ROOMS),
        "personnel": {"1": list(anesthetists), "2": list(surgeons)},
    }

    assert (context.instance_id, context.generation_seed, context.digest) == (
        identifier,
        seed,
        digest,
    )
    assert canonical_digest(document) == document["digest"] == entry["digest"] == digest
    assert document["validation"]["evidence"] == evidence
    assert document["resources"] == resources
    assert context.personnel_by_operation == ((1, anesthetists), (2, surgeons))
    assert all(
        entry[field] == document[field]
        for field in (
            "provenance", "generation", "dimensions", "resources", "classification",
            "bounds", "validation", "digest",
        )
    )
    assert context.rooms == HOSPITAL_12ROOMS
    assert all(
        operation.eligible_rooms == HOSPITAL_12ROOMS
        for job in context.jobs
        for operation in job.operations
    )


def test_hospital_12rooms_30_catalog_owns_twelve_room_resources_and_eligibility():
    root = Path(__file__).parents[2] / "instances"
    expected_files = [Path(item[1]).name for item in HOSPITAL_12ROOMS_30_CATALOG]

    assert sorted(
        path.name
        for path in (root / "hospital_12rooms").glob("HOSP-12R-30-*.yaml")
    ) == expected_files

    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    contexts = load_catalog(root)
    entries = metadata["instances"]
    hospital_entries = entries[25:28]
    documents = [
        yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
        for _, relative_path, *_ in HOSPITAL_12ROOMS_30_CATALOG
    ]
    expected_rooms = list(HOSPITAL_12ROOMS)

    assert metadata["catalog_status"] == "partial"
    assert metadata["instance_count"] == len(entries)
    assert len(entries[25:28]) == 3
    assert [entry["instance_id"] for entry in hospital_entries] == [
        item[0] for item in HOSPITAL_12ROOMS_30_CATALOG
    ]
    assert [entry["file"] for entry in hospital_entries] == [
        item[1] for item in HOSPITAL_12ROOMS_30_CATALOG
    ]
    assert [len(context.jobs) for context in contexts[25:28]] == [30, 30, 30]
    assert all(document["family"] == "hospital_12rooms" for document in documents)
    assert all(document["resources"]["rooms"] == expected_rooms for document in documents)
    assert all(list(context.rooms) == expected_rooms for context in contexts[25:28])
    assert all(
        operation["eligible_rooms"] == expected_rooms
        for document in documents
        for job in document["jobs"]
        for operation in job["operations"]
    )
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in documents
    }
    assert configurations == {
        (seed, congestion_band, first_duration, first_wait, room_ratio)
        for (
            _, _, seed, congestion_band, first_duration, first_wait, room_ratio
        ) in HOSPITAL_12ROOMS_30_CATALOG
    }


@pytest.mark.parametrize(
    ("identifier", "seed"),
    [(identifier, seed) for identifier, _, seed, *_ in HOSPITAL_12ROOMS_30_CATALOG],
)
def test_hospital_12rooms_30_full_evidence_resources_and_digest_are_frozen(
    identifier, seed
):
    root = Path(__file__).parents[2] / "instances"
    path = root / "hospital_12rooms" / f"{identifier}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in metadata["instances"] if item["instance_id"] == identifier)
    digest = HOSPITAL_12ROOMS_30_DIGESTS[identifier]
    anesthetists, surgeons = HOSPITAL_12ROOMS_30_PERSONNEL[identifier]
    resources = {
        "rooms": list(HOSPITAL_12ROOMS),
        "personnel": {"1": list(anesthetists), "2": list(surgeons)},
    }

    assert (context.instance_id, context.generation_seed, context.digest) == (
        identifier,
        seed,
        digest,
    )
    assert canonical_digest(document) == document["digest"] == entry["digest"] == digest
    assert document["validation"]["evidence"] == (
        HOSPITAL_12ROOMS_30_VALIDATION_EVIDENCE[identifier]
    )
    assert document["resources"] == resources
    assert context.personnel_by_operation == ((1, anesthetists), (2, surgeons))
    assert all(
        entry[field] == document[field]
        for field in (
            "provenance", "generation", "dimensions", "resources", "classification",
            "bounds", "validation", "digest",
        )
    )
    assert context.rooms == HOSPITAL_12ROOMS
    assert all(
        operation.eligible_rooms == HOSPITAL_12ROOMS
        for job in context.jobs
        for operation in job.operations
    )


@pytest.mark.parametrize(
    ("job_count", "start", "frozen"),
    [
        (40, 28, HOSPITAL_12ROOMS_40_FROZEN),
        (50, 31, HOSPITAL_12ROOMS_50_FROZEN),
        (60, 34, HOSPITAL_12ROOMS_60_FROZEN),
    ],
)
def test_hospital_12rooms_large_catalog_has_global_parity_and_fixed_slice(job_count, start, frozen):
    root = Path(__file__).parents[2] / "instances"
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    entries = metadata["instances"]
    contexts = load_catalog(root)
    documents = [
        yaml.safe_load((root / entry["file"]).read_text(encoding="utf-8"))
        for entry in entries
    ]
    hospital_entries = entries[start : start + 3]
    hospital_documents = documents[start : start + 3]
    expected_files = [item[1] for item in frozen]
    expected_ids = [item[0] for item in frozen]
    actual_files = {
        path.relative_to(root).as_posix() for path in root.glob("*/*.yaml")
    }
    parity_fields = (
        "provenance", "generation", "dimensions", "resources", "classification",
        "bounds", "validation", "digest",
    )

    assert metadata["catalog_status"] == "partial"
    assert metadata["instance_count"] == len(entries) == len(contexts)
    assert {entry["file"] for entry in entries} == actual_files
    assert [context.instance_id for context in contexts] == [
        entry["instance_id"] for entry in entries
    ] == [document["instance_id"] for document in documents]
    assert [context.digest for context in contexts] == [
        entry["digest"] for entry in entries
    ] == [document["digest"] for document in documents]
    assert len({context.digest for context in contexts}) == len(contexts)
    assert all(canonical_digest(document) == document["digest"] for document in documents)
    assert all(
        entry[field] == document[field]
        for entry, document in zip(entries, documents, strict=True)
        for field in parity_fields
    )
    assert len(hospital_entries) == 3
    assert [entry["file"] for entry in hospital_entries] == expected_files
    assert [entry["instance_id"] for entry in hospital_entries] == expected_ids
    assert [len(context.jobs) for context in contexts[start : start + 3]] == [job_count] * 3
    assert all(document["family"] == "hospital_12rooms" for document in hospital_documents)
    assert all(document["resources"]["rooms"] == list(HOSPITAL_12ROOMS) for document in hospital_documents)
    configurations = {
        (
            document["generation"]["seed"],
            document["generation"]["profile"]["congestion_bands"],
            document["jobs"][0]["operations"][0]["duration"],
            document["jobs"][0]["operations"][1]["max_wait"],
            document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"],
        )
        for document in hospital_documents
    }
    assert configurations == {
        (seed, profile, first_duration, first_wait, evidence["workload_capacity_ratios"]["rooms"])
        for _, _, _, seed, profile, first_duration, first_wait, evidence, *_
        in frozen
    }


@pytest.mark.parametrize(
    ("identifier", "relative_path", "digest", "seed", "profile", "first_duration", "first_wait", "evidence", "anesthetists", "surgeons"),
    HOSPITAL_12ROOMS_40_FROZEN + HOSPITAL_12ROOMS_50_FROZEN + HOSPITAL_12ROOMS_60_FROZEN,
)
def test_hospital_12rooms_large_full_evidence_resources_and_digest_are_frozen(
    identifier, relative_path, digest, seed, profile, first_duration, first_wait,
    evidence, anesthetists, surgeons,
):
    root = Path(__file__).parents[2] / "instances"
    path = root / relative_path
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in metadata["instances"] if item["instance_id"] == identifier)
    resources = {
        "rooms": list(HOSPITAL_12ROOMS),
        "personnel": {"1": list(anesthetists), "2": list(surgeons)},
    }

    assert (context.instance_id, context.generation_seed, context.digest) == (
        identifier, seed, digest,
    )
    assert canonical_digest(document) == document["digest"] == entry["digest"] == digest
    assert document["generation"]["profile"]["congestion_bands"] == profile
    assert document["jobs"][0]["operations"][0]["duration"] == first_duration
    assert document["jobs"][0]["operations"][1]["max_wait"] == first_wait
    assert document["validation"]["evidence"] == evidence
    assert document["resources"] == resources
    assert context.personnel_by_operation == ((1, anesthetists), (2, surgeons))
    assert context.rooms == HOSPITAL_12ROOMS
    assert all(
        operation.eligible_rooms == HOSPITAL_12ROOMS
        for job in context.jobs
        for operation in job.operations
    )


@pytest.mark.parametrize(
    ("identifier", "digest", "seed", "congestion_band", "room_ratio"),
    STANDARD_15_EVIDENCE,
)
def test_standard_15_replica_evidence_and_digest_are_frozen(
    identifier, digest, seed, congestion_band, room_ratio
):
    path = Path(__file__).parents[2] / "instances" / "standard" / f"{identifier}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)

    assert (context.instance_id, context.digest, context.generation_seed) == (
        identifier,
        digest,
        seed,
    )
    assert document["generation"]["profile"]["congestion_bands"] == congestion_band
    assert (
        document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"]
        == room_ratio
    )


@pytest.mark.parametrize(
    ("identifier", "digest", "seed", "congestion_band", "room_ratio"),
    STANDARD_20_EVIDENCE,
)
def test_standard_20_replica_evidence_and_digest_are_frozen(
    identifier, digest, seed, congestion_band, room_ratio
):
    path = Path(__file__).parents[2] / "instances" / "standard" / f"{identifier}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)

    assert (context.instance_id, context.digest, context.generation_seed) == (
        identifier,
        digest,
        seed,
    )
    assert document["generation"]["profile"]["congestion_bands"] == congestion_band
    assert (
        document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"]
        == room_ratio
    )


@pytest.mark.parametrize(
    ("identifier", "digest", "seed", "congestion_band", "room_ratio"),
    STANDARD_25_EVIDENCE,
)
def test_standard_25_replica_evidence_and_digest_are_frozen(
    identifier, digest, seed, congestion_band, room_ratio
):
    path = Path(__file__).parents[2] / "instances" / "standard" / f"{identifier}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)

    assert (context.instance_id, context.digest, context.generation_seed) == (
        identifier,
        digest,
        seed,
    )
    assert document["generation"]["profile"]["congestion_bands"] == congestion_band
    assert (
        document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"]
        == room_ratio
    )


@pytest.mark.parametrize(
    ("identifier", "digest", "seed", "congestion_band", "room_ratio"),
    [
        (identifier, digest, seed, congestion_band, room_ratio)
        for (
            identifier, _, digest, seed, congestion_band, _, _, room_ratio
        ) in STANDARD_30_CATALOG
    ],
)
def test_standard_30_replica_validation_evidence_and_digest_are_frozen(
    identifier, digest, seed, congestion_band, room_ratio
):
    path = Path(__file__).parents[2] / "instances" / "standard" / f"{identifier}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    context = load_instance(path)

    assert (context.instance_id, context.digest, context.generation_seed) == (
        identifier,
        digest,
        seed,
    )
    assert canonical_digest(document) == digest
    assert document["generation"]["profile"]["congestion_bands"] == congestion_band
    assert document["validation"]["evidence"] == STANDARD_30_VALIDATION_EVIDENCE[identifier]
    assert document["validation"]["evidence"]["workload_capacity_ratios"]["rooms"] == room_ratio


def instance_payload(identifier="HOSP-DIDACT-03-01", duration=12.0):
    payload = dict(
        schema_version=1, instance_id=identifier, family="didactic",
        classification="fully synthetic instance",
        provenance={"kind": "public pedagogical", "source": "course profile"},
        generation={
            "method": "fixed pedagogical", "version": "1", "seed": 7,
            "profile": {
                "duration_distributions": "positive-support constants",
                "dependence": "fixed public relationship",
                "assignment": "qualified subsets",
                "room_eligibility": "all declared rooms",
                "congestion_bands": "public profile bands",
                "replica_policy": "didactic singleton",
            },
        },
        dimensions={"jobs": 1, "operations_per_job": 2},
        resources={
            "rooms": ["R1", "R2"],
            "personnel": {"1": ["A1"], "2": ["S1"]},
        },
        jobs=[
            {
                "id": 1,
                "label": "Case 1",
                "operations": [
                    {"id": 1, "duration": duration, "setup": 2.0, "transition": 1.0, "cleanup": 0.0, "max_wait": 0.0, "eligible_rooms": ["R1", "R2"], "eligible_personnel": ["A1"]},
                    {"id": 2, "duration": 20.0, "setup": 0.0, "transition": 0.0, "cleanup": 3.0, "max_wait": 30.0, "eligible_rooms": ["R1", "R2"], "eligible_personnel": ["S1"]},
                ],
            }
        ],
        bounds={"status": "pending", "method_version": "course-lb-v1"},
        validation={"method": "instance-schema", "version": "1", "outcome": "passed"},
    )
    payload["digest"] = canonical_digest(payload)
    return payload
def test_catalog_admits_varied_instances_with_stable_digests(tmp_path):
    second = deepcopy(instance_payload("HOSP-DIDACT-03-02", 17.0))
    root = build_catalog(tmp_path, [instance_payload(), second])

    contexts = load_catalog(root)
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))

    assert tuple(item.instance_id for item in contexts) == (
        "HOSP-DIDACT-03-01", "HOSP-DIDACT-03-02"
    )
    assert contexts[0].digest != contexts[1].digest
    assert load_catalog(root) == contexts
    assert metadata["instance_count"] == 2
    assert set(metadata["instances"][0]) == {
        "instance_id", "file", "provenance", "generation", "dimensions",
        "resources", "classification", "bounds", "validation", "digest",
    }
@pytest.mark.parametrize(
    ("half_width", "current", "expected"),
    [(1.0, 3, 3), (1.01, 3, 5), (2.0, 9, 10)],
)
def test_replica_expansion_is_evidence_gated(half_width, current, expected):
    metrics = [{"mean": 10.0, "ci_half_width": half_width}]
    assert required_replica_count(metrics, current=current) == expected
