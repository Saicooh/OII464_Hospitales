# Guía para alumnos: planificación de cirugías

Este repositorio permite comparar cuatro algoritmos de planificación usando
instancias sintéticas de hospitales. No contiene pacientes ni datos de
hospitales reales.

El flujo es:

~~~text
instancia YAML → algoritmos → calendario de salas y personal → CSV y gráficos
~~~

## Instancias disponibles

El nombre de una instancia indica su familia, la cantidad de trabajos y la
réplica. Por ejemplo, `HOSP-12R-30-03` es una instancia de 30 trabajos, con
12 salas y réplica 03.

En las tablas, Personal indica la cantidad total de anestesiólogos/cirujanos
disponibles. Las listas de personal elegible pueden variar entre operaciones.

### Didácticas

| Instancia | Trabajos | Personal AN/SU |
|---|---:|---:|
| HOSP-DIDACT-03-01 | 3 | 3/4 |
| HOSP-DIDACT-05-01 | 5 | 3/5 |
| HOSP-DIDACT-08-01 | 8 | 4/6 |
| HOSP-DIDACT-10-01 | 10 | 4/7 |

### Estándar

| Instancia | Trabajos | Personal AN/SU |
|---|---:|---:|
| HOSP-STD-15-01 | 15 | 5/8 |
| HOSP-STD-15-02 | 15 | 5/7 |
| HOSP-STD-15-03 | 15 | 6/10 |
| HOSP-STD-20-01 | 20 | 6/10 |
| HOSP-STD-20-02 | 20 | 6/9 |
| HOSP-STD-20-03 | 20 | 7/12 |
| HOSP-STD-25-01 | 25 | 7/13 |
| HOSP-STD-25-02 | 25 | 7/12 |
| HOSP-STD-25-03 | 25 | 8/15 |
| HOSP-STD-30-01 | 30 | 8/16 |
| HOSP-STD-30-02 | 30 | 8/15 |
| HOSP-STD-30-03 | 30 | 9/18 |

### Hospital con 12 salas

Estas instancias utilizan las salas `OR-1` a `OR-12`.

| Instancia | Trabajos | Personal AN/SU |
|---|---:|---:|
| HOSP-12R-15-01 | 15 | 7/12 |
| HOSP-12R-15-02 | 15 | 6/10 |
| HOSP-12R-15-03 | 15 | 8/14 |
| HOSP-12R-20-01 | 20 | 8/15 |
| HOSP-12R-20-02 | 20 | 7/13 |
| HOSP-12R-20-03 | 20 | 9/17 |
| HOSP-12R-25-01 | 25 | 9/18 |
| HOSP-12R-25-02 | 25 | 8/16 |
| HOSP-12R-25-03 | 25 | 10/20 |
| HOSP-12R-30-01 | 30 | 10/22 |
| HOSP-12R-30-02 | 30 | 9/20 |
| HOSP-12R-30-03 | 30 | 11/24 |
| HOSP-12R-40-01 | 40 | 11/25 |
| HOSP-12R-40-02 | 40 | 10/23 |
| HOSP-12R-40-03 | 40 | 11/27 |
| HOSP-12R-50-01 | 50 | 11/27 |
| HOSP-12R-50-02 | 50 | 10/25 |
| HOSP-12R-50-03 | 50 | 11/29 |
| HOSP-12R-60-01 | 60 | 11/27 |
| HOSP-12R-60-02 | 60 | 10/25 |
| HOSP-12R-60-03 | 60 | 11/29 |

Cada trabajo tiene dos operaciones: anestesia (operación 1) y cirugía
(operación 2). El YAML define sus duraciones, salas permitidas, personal
elegible y tiempos auxiliares.

## Algoritmos

La comparación ejecuta, en este orden:

1. GA, algoritmo genético.
2. dPSO, versión discreta de Particle Swarm Optimization.
3. SBOA, Secretary Bird Optimization Algorithm.
4. dMShOA.

Los cuatro nombres aparecen igual en los CSV y en los gráficos. No se debe
interpretar `dMShOA` como una segunda variante: es la única entrada de ese
algoritmo.

### Espacio para la metaheurística de los alumnos

El repositorio incluye una quinta entrada opcional llamada `MH`. Su plantilla
está en `algorithms/mh.py` y usa una búsqueda aleatoria sencilla como punto de
partida. Cada grupo debe reemplazar esa lógica por su propia metaheurística.

Para probarla, cambia `enabled` a `true` en el bloque `mh` de
`config/config.yaml`. La función debe conservar esta interfaz:

~~~python
from simulation.result_model import SolverOutput

def run(context, seed, on_iteration=None) -> SolverOutput:
    ...
~~~

La solución debe usar los trabajos y recursos de `context` y devolver un
`SolverOutput`. Al activarla, aparecerá como `MH` en el resumen, los CSV y los
gráficos. Mientras esté desactivada, la ejecución normal conserva solo los
cuatro algoritmos base.

## Instalación

Desde la carpeta raíz del repositorio:

~~~powershell
python -m pip install -r requirements.txt
~~~

El proyecto está preparado para Python 3.13.

## Ejecutar el proyecto

La ejecución normal usa la instancia indicada en `config/config.yaml`:

~~~powershell
python -u main.py
~~~

Para una ejecución corta de comprobación:

~~~powershell
$env:HOSPITAL_CONFIG_PATH = "config/config.quick.yaml"
python -u main.py
Remove-Item Env:HOSPITAL_CONFIG_PATH -ErrorAction SilentlyContinue
~~~

Para seleccionar otra instancia sin editar el código:

~~~powershell
$env:HOSPITAL_INSTANCE_PATH = "instances/hospital_12rooms/HOSP-12R-30-03.yaml"
python -u main.py
Remove-Item Env:HOSPITAL_INSTANCE_PATH -ErrorAction SilentlyContinue
~~~

También se puede separar la salida de una ejecución:

~~~powershell
$env:HOSPITAL_OUTPUT_ROOT = "results/prueba-r30"
python -u main.py
Remove-Item Env:HOSPITAL_OUTPUT_ROOT -ErrorAction SilentlyContinue
~~~

## Resultados

Los archivos se guardan en `results/elective/`, dentro de `csv/` y `plots/`.

| Archivo | Qué muestra |
|---|---|
| `summary_results.csv` | Makespan y tiempo promedio por algoritmo |
| `elective_room_overtimes.csv` | Sobrecarga o tiempo extra por sala |
| `statistical_analysis.csv` | Comparaciones estadísticas |
| `elective_best_schedule_*.csv` | Mejor calendario de cada algoritmo |
| `elective_best_strategy_*.csv` | Secuencia de salas utilizada |
| `elective_routing_explanation_*.csv` | Esperas, salas y personal asignado |

Las carpetas de gráficos principales son `boxplot`, `barplot`, `gantt`,
`convergence`, `histograms` y `personnel`. También se generan versiones SVG
en las carpetas correspondientes.

## Cómo leer los resultados

- **Makespan:** momento en que termina el último trabajo. En una misma
  instancia, un valor menor normalmente representa un calendario más compacto.
- **Tiempo de ejecución:** cuánto tardó el algoritmo en encontrar su solución;
  no es lo mismo que el makespan.
- **Gantt:** cada fila es una sala, el eje horizontal es el tiempo y cada
  bloque de color representa un trabajo. Los trabajos se identifican por su
  número.
- **Convergencia:** muestra cómo cambia el fitness durante las iteraciones.
- **`valid_simulations`:** cantidad de simulaciones que terminaron con un
  calendario válido.

No conviene comparar directamente el makespan de instancias con distinta
cantidad de trabajos: primero hay que considerar el tamaño de cada problema.

## Problemas frecuentes

Si aparece una instancia distinta de la esperada, revisa las variables de
PowerShell:

~~~powershell
echo $env:HOSPITAL_INSTANCE_PATH
echo $env:HOSPITAL_CONFIG_PATH
~~~

Si faltan CSV o gráficos, revisa primero que la consola indique calendarios
válidos y que la carpeta de salida corresponda a la ejecución actual.

Si se modificó un YAML manualmente, vuelve a usar una instancia del catálogo:
el archivo contiene información de validación que debe mantenerse consistente.
