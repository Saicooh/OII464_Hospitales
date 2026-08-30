"""
Plotting utilities with DRY principles applied.
All visualization logic is abstracted into reusable components.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend for faster rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib import colors as mcolors
from typing import List, Dict, Optional, Tuple

from config.config import (
    JOB_TYPES,
    SETUP_TIMES,
    CLEANUP_TIMES,
    VERBOSE_MODE,
    get_job_type,
)
from utils.logger import logger

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

ALGORITHM_COLORS = {
    "GA": "lightblue",
    "dPSO": "lightsalmon",
    "SBOA": "lightgreen",
    "dMShOA": "mediumpurple",
}

BOXPLOT_COLORS = ["lightblue", "lightsalmon", "lightgreen", "mediumpurple"]
ALGORITHM_ORDER = ["GA", "dPSO", "SBOA", "dMShOA"]
ALGORITHM_DATA_KEYS = {"GA": "GA", "dPSO": "dPSO", "SBOA": "SBOA", "dMShOA": "dMShOA Old"}

# Gantt chart styling
GANTT_STYLE = {
    "bar_height": 0.25,
    "room_spacing": 0.60,
    "group_gap": 0.75,
    "edge_color": "black",
    "edge_width": 0.5,
    "alpha": 0.9,
    "setup_color": "yellow",
    "transition_color": "mediumpurple",  # nuevo: tramo Anestesia→Intervención
    "cleanup_color": "lightcoral",
    "unused_room_color": "lightgray",
    "unused_room_alpha": 0.2,
    # Connector lines between op1 and op2 of the same job
    "connector_color": "#555555",
    "connector_lw": 1.05,
    "connector_alpha": 0.65,
    "connector_linestyle": (0, (4, 3)),  # visible but still unobtrusive
}

# =============================================================================
# CORE PLOTTING COMPONENTS (DRY ABSTRACTIONS)
# =============================================================================

# Load PDF preference from config (default: disabled for speed)
try:
    from config.config import _CONFIG

    _SAVE_PDF = _CONFIG.get("experiment", {}).get("save_pdf", False)
except Exception:
    _SAVE_PDF = False


class PlotConfig:
    """Base configuration for all plots."""

    DEFAULT_DPI = 150
    DEFAULT_FIGSIZE = (10, 6)
    GANTT_FIGSIZE = (12, 6)

    DEFAULT_AXIS_LABEL_SIZE = 15.0
    DEFAULT_LEGEND_SIZE = 13.0

    _created_dirs = set()  # Cache to avoid repeated os.makedirs calls

    @staticmethod
    def get_output_paths(
        output_dir: str, subdir: str, filename: str
    ) -> Tuple[str, str]:
        """
        Returns (png_path, pdf_path) for a given plot.
        Creates directories only once per unique path.
        """
        plot_dir = os.path.join(output_dir, subdir)
        if plot_dir not in PlotConfig._created_dirs:
            os.makedirs(plot_dir, exist_ok=True)
            PlotConfig._created_dirs.add(plot_dir)

        pdf_path = None
        if _SAVE_PDF:
            pdf_dir = os.path.join(plot_dir, "pdf")
            if pdf_dir not in PlotConfig._created_dirs:
                os.makedirs(pdf_dir, exist_ok=True)
                PlotConfig._created_dirs.add(pdf_dir)
            pdf_path = os.path.join(pdf_dir, f"{filename}.pdf")

        png_path = os.path.join(plot_dir, f"{filename}.png")

        return png_path, pdf_path

    @staticmethod
    def save_and_close(
        fig,
        png_path: str = None,
        pdf_path: str = None,
        dpi: int = DEFAULT_DPI,
        bbox_inches=None,
        pad_inches: float = 0.1,
        trim_right: bool = False,
        trim_pad_px: int = 10,
    ):
        """Saves figure to PNG (and optionally PDF), then closes it.

        Args:
            png_path: Destination PNG path. Pass None to emit no raster output
                (and no SVG sidecar); use this for vector-only exports where an
                existing PNG must be left untouched.
            bbox_inches: Passed to savefig. Use 'tight' to let matplotlib
                attempt to trim whitespace (note: inset_axes elements outside
                the main axes may not be captured — use trim_right instead).
            pad_inches: Extra padding around the bounding box when
                bbox_inches='tight'. Ignored for other bbox_inches values.
            trim_right: If True, after saving the PNG, trims trailing
                whitespace columns from the right edge of the image using
                PIL/Pillow and re-saves. Use this when a side panel (inset_axes
                with clip_on=False) causes invisible right-side whitespace that
                matplotlib's tight layout cannot detect.
            trim_pad_px: Pixels of padding to add after trimming the right edge.
                Keeps a clean thin border instead of flush-to-edge.
        """
        if not getattr(fig, "_skip_tight_layout", False):
            plt.tight_layout(pad=0.2)
        save_kwargs = {"dpi": dpi}
        if bbox_inches is not None:
            save_kwargs["bbox_inches"] = bbox_inches
            if bbox_inches == "tight":
                save_kwargs["pad_inches"] = pad_inches
        if png_path:
            plt.savefig(png_path, **save_kwargs)
        if pdf_path:
            os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
            plt.savefig(pdf_path, **save_kwargs)

        # Always save SVG next to PNG (under an 'svg/' subdirectory).
        # Skipped entirely when no PNG target was requested.
        if png_path:
            try:
                plot_dir = os.path.dirname(png_path)
                svg_dir = os.path.join(plot_dir, "svg")
                os.makedirs(svg_dir, exist_ok=True)
                filename_no_ext = os.path.splitext(os.path.basename(png_path))[0]
                svg_path = os.path.join(svg_dir, f"{filename_no_ext}.svg")
                plt.savefig(svg_path, **save_kwargs)
            except Exception as e:
                logger.warning(f"Could not save SVG plot: {e}")

        plt.close(fig)

        # Post-process: trim right-side whitespace that matplotlib missed
        if trim_right and png_path:
            try:
                import numpy as np
                from PIL import Image as _PILImage

                img = _PILImage.open(png_path)
                arr = np.array(img)
                h, w = arr.shape[:2]
                # Find the rightmost column with any non-white (non-255) pixel
                last_content_col = 0
                for col in range(w - 1, -1, -1):
                    col_pixels = arr[:, col, :3]
                    if not (col_pixels == 255).all():
                        last_content_col = col
                        break
                # Crop to content + pad
                crop_right = min(last_content_col + trim_pad_px + 1, w)
                if crop_right < w:
                    cropped = img.crop((0, 0, crop_right, h))
                    cropped.save(png_path, dpi=(dpi, dpi))
            except Exception:
                pass  # Never fail silently — if PIL is missing, skip trimming


class GanttChartBuilder:
    """
    Builds Gantt charts with DRY principles.
    """

    # Minimum vertical separation (in y-axis units) between two external labels
    # on the same row before we start staggering them.
    _LABEL_STAGGER_THRESHOLD = 0.08

    # In-figure font sizes for the default Gantt geometry.
    _FONTS_SCREEN = {
        "job_label": 8,
        "unused": 8.5,
        "ytick": 10.5,
        "xtick": None,  # inherit rcParams
        "axis_label": 11.5,
        "title": 13,
        "legend": 9.5,
        "legend_ncol": 3,
        "legend_anchor_y": -0.13,
    }

    def __init__(
        self,
        schedule_details: List[Dict],
        rooms: List[str],
        job_label_map: Optional[Dict] = None,
        show_legend: bool = True,
    ):
        """
        Args:
            show_legend: Whether to draw the segment legend. Defaults to True
                so existing PNG/default renders retain their behavior.
        """
        self.show_legend = show_legend
        self.fonts = self._FONTS_SCREEN

        self.schedule_details = schedule_details
        self.rooms = rooms
        # job_label_map: {job_id -> label_str} for CIE10 display.
        # Falls back to str(job_id) when missing or None.
        self.job_label_map = job_label_map or {}

        # Computed properties
        self.job_colors = self._compute_job_colors()
        self.y_positions = self._compute_y_positions()
        self.max_time = self._compute_max_time()
        self.rooms_with_tasks = set(t["Resource"] for t in schedule_details)
        self.unused_rooms = set(rooms) - self.rooms_with_tasks

        # State for anti-overlap of external labels.
        # Maps room -> list of (x_anchor, y_text) already placed.
        self._ext_label_positions: Dict[str, List[Tuple[float, float]]] = {}

    def _compute_job_colors(self) -> Dict:
        """Assign colors to jobs deterministically."""
        all_jobs = set(t["Job"] for t in self.schedule_details)
        int_jobs = sorted([j for j in all_jobs if isinstance(j, int)])
        str_jobs = sorted([j for j in all_jobs if isinstance(j, str)])
        unique_jobs = int_jobs + str_jobs

        try:
            cmap = plt.get_cmap("tab20" if len(unique_jobs) <= 20 else "turbo")
        except Exception:
            cmap = plt.get_cmap("tab10")

        color_list = [
            cmap(i / max(1, len(unique_jobs) - 1)) for i in range(len(unique_jobs))
        ]
        job_colors = {job_id: color_list[i] for i, job_id in enumerate(unique_jobs)}

        return job_colors

    def _compute_y_positions(self) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
        """Computes y-axis positions for rooms."""
        y_pos, y_labels = {}, []
        y = 0

        # Sort rooms numerically if they end with a number (e.g., Pabellon_1)
        def sort_key(room_name):
            try:
                return int(room_name.split("_")[-1])
            except ValueError:
                return room_name

        for room in sorted(self.rooms, key=sort_key):
            y_pos[room] = y
            display_room = (
                str(room).replace("Pabellon_", "OR_").replace("Pabellón_", "OR_")
            )
            y_labels.append((room, display_room))
            y += GANTT_STYLE["room_spacing"]

        return y_pos, y_labels

    def _compute_max_time(self) -> float:
        """Computes makespan from schedule."""
        if not self.schedule_details:
            return 0
        return max(
            t.get("Finish", 0)
            for t in self.schedule_details
            if t.get("Finish", -1) >= 0
        )

    def _draw_task_bar(self, ax, task: Dict):
        """
        Draws the temporal segments of a task in the Gantt chart.

        Segmentos soportados:
            Setup       → yellow      (OR entry to anesthesia)
            Processing  → job color   (actual operation processing)
            Transition  → purple      (real-data Op2: anesthesia to intervention)
            Cleanup     → light red   (real-data Op2: intervention to OR exit)

        Resolución de tiempos (orden de prioridad):
            1. Campos reales del scheduler (SetupUsed, TransitionUsed, CleanupUsed)
               → propagados desde datos PKL reales.
            2. Fallback estático (SETUP_TIMES / CleanupUsed=0) para modo sintético
               o cualquier tarea sin campos dinámicos.

        Esto garantiza que el Gantt en modo real NO dependa de SETUP_TIMES estáticos.
        """
        job = task["Job"]
        resource = task["Resource"]
        start = task.get("Start", -1)
        processing_end = task.get("ProcessingEnd", -1)
        finish = task.get("Finish", -1)

        if (
            start < 0
            or processing_end < 0
            or finish < 0
            or resource not in self.y_positions[0]
        ):
            if VERBOSE_MODE:
                logger.warning(f"Skipping invalid task: Job={job}, Resource={resource}")
            return

        # --- Resolución de tiempos de setup/transition/cleanup ---
        # Prioridad 1: campos reales propagados por el scheduler desde PKL
        setup_used = task.get("SetupUsed")  # None si no está presente
        transition_used = task.get("TransitionUsed")  # None si op1 o modo sintético
        cleanup_used = task.get("CleanupUsed")  # None si no está presente

        is_parallel = (task.get("Operation") == 1)

        if setup_used is not None:
            # Modo real: tiempos dinámicos desde PKL
            setup_duration = setup_used
            cleanup_duration = (
                cleanup_used if cleanup_used is not None else (finish - processing_end)
            )
            # El "tiempo previo al procesamiento" es: setup (op1) o transición (op2)
            # En op1: time_before_proc = setup_duration; transition_duration = 0
            # En op2: time_before_proc = transition_used;  setup_duration = 0
            transition_duration = (
                transition_used
                if (transition_used is not None and transition_used > 1e-6)
                else 0.0
            )
            if is_parallel:
                # Discrepancia 5 corregida: En la operación 1 (Anestesia), la transición del paciente
                # y la preparación física del quirófano (setup_time) ocurren en paralelo.
                time_before_proc = max(setup_duration, transition_duration)
            else:
                time_before_proc = setup_duration + transition_duration
        else:
            # Fallback estático: modo sintético o tarea sin campos dinámicos
            job_type = get_job_type(job)
            setup_duration = SETUP_TIMES.get(job_type, 0)
            cleanup_duration = finish - processing_end
            transition_duration = 0.0
            time_before_proc = setup_duration

        # Duración de procesamiento: desde (start + time_before_proc) hasta processing_end
        proc_duration = processing_end - (start + time_before_proc)

        if min(setup_duration, proc_duration, cleanup_duration) < -1e-6:
            if VERBOSE_MODE:
                logger.warning(
                    f"Negative duration: Job={job} Op={task.get('Operation')}"
                )
            return

        y = self.y_positions[0][resource]
        edge_width = GANTT_STYLE["edge_width"]
        edge_color = GANTT_STYLE["edge_color"]

        # --- Segmento 1: Transición observada (hatch sin fondo) ---
        if transition_duration > 1e-6:
            ax.barh(
                y=y,
                width=transition_duration,
                left=start,
                height=GANTT_STYLE["bar_height"],
                facecolor="none",
                edgecolor=edge_color,
                linewidth=edge_width,
                hatch="\\\\\\",
            )

        # --- Segmento 2: Setup (hatch sin fondo) ---
        if setup_duration > 1e-6:
            # En la operación 1, el setup y la transición ocurren en paralelo (ambos inician en 'start')
            setup_start = start if is_parallel else start + transition_duration
            ax.barh(
                y=y,
                width=setup_duration,
                left=setup_start,
                height=GANTT_STYLE["bar_height"],
                facecolor="none",
                edgecolor=edge_color,
                linewidth=edge_width,
                hatch="///",
            )

        # --- Segmento 3: Procesamiento (color por job/CIE10) ---
        if proc_duration > 1e-6:
            proc_start = start + time_before_proc
            proc_color = self.job_colors.get(job, "gray")

            ax.barh(
                y=y,
                width=proc_duration,
                left=proc_start,
                height=GANTT_STYLE["bar_height"],
                color=proc_color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=GANTT_STYLE["alpha"],
            )

            # Etiqueta del job: colocada siempre arriba de la barra (evita solapamiento)
            job_label = self.job_label_map.get(job, str(job))
            label_x = proc_start + proc_duration / 2
            # El eje y está invertido, por lo que restar del centro coloca la etiqueta arriba de la barra
            label_y = y - GANTT_STYLE["bar_height"] / 2 - 0.08

            ax.text(
                label_x,
                label_y,
                job_label,
                ha="center",
                va="bottom",
                color="black",
                fontweight="bold",
                fontsize=self.fonts["job_label"],
                clip_on=True,
            )

        # --- Segmento 4: Cleanup (hatch sin fondo) ---
        if cleanup_duration > 1e-6:
            ax.barh(
                y=y,
                width=cleanup_duration,
                left=processing_end,
                height=GANTT_STYLE["bar_height"],
                facecolor="none",
                edgecolor=edge_color,
                linewidth=edge_width,
                hatch="|||",
            )

    def _draw_job_connectors(self, ax):
        """Dibuja líneas de conexión (desactivado en plan visual)."""
        return

    def _draw_unused_rooms(self, ax):
        """Draws placeholder bars for unused rooms."""
        if self.max_time <= 0:
            return

        for room in self.unused_rooms:
            if room in self.y_positions[0]:
                y = self.y_positions[0][room]
                ax.barh(
                    y=y,
                    width=self.max_time * 0.01,
                    left=0,
                    height=GANTT_STYLE["bar_height"],
                    color=GANTT_STYLE["unused_room_color"],
                    edgecolor="gray",
                    linewidth=0.3,
                    alpha=GANTT_STYLE["unused_room_alpha"],
                    linestyle="--",
                )

                ax.text(
                    self.max_time * 0.005,
                    y,
                    "UNUSED",
                    ha="left",
                    va="center",
                    color="gray",
                    fontsize=self.fonts["unused"],
                    style="italic",
                    alpha=0.6,
                )

    def _configure_axes(self, ax, title: Optional[str], sim_num: Optional[int] = None):
        """Configures axes labels, optional title, ticks, and grid."""
        y_pos, y_labels = self.y_positions

        ax.set_yticks([y_pos[room] for room, _display in y_labels])
        ax.set_yticklabels(
            [display for _room, display in y_labels], fontsize=self.fonts["ytick"]
        )
        ax.set_xlabel("Time (minutes)", fontsize=self.fonts["axis_label"])
        ax.set_ylabel("Operating Room", fontsize=self.fonts["axis_label"])
        if self.fonts["xtick"] is not None:
            ax.tick_params(axis="x", labelsize=self.fonts["xtick"])

        if title:
            # Title with simulation number
            if sim_num is not None:
                title_text = (
                    f"{title} (Best: Sim #{sim_num}) - Makespan: {self.max_time:.2f} min"
                )
            else:
                title_text = f"{title} - Makespan: {self.max_time:.2f} min"

            ax.set_title(title_text, fontsize=self.fonts["title"], fontweight="bold")

        # X-axis limits and ticks
        # Margen derecho interno del eje: pequeño colchón para leader lines/labels.
        # Se mantiene porcentaje moderado porque el usuario quiere revisar por separado
        # el espacio EXTERNO a la derecha del panel CIE10, no el interno del eje.
        if self.max_time > 0:
            ax.set_xlim(0, self.max_time * 1.03)

            # Dynamic tick interval
            if self.max_time <= 500:
                tick_interval = 50
            elif self.max_time <= 1000:
                tick_interval = 100
            elif self.max_time <= 2000:
                tick_interval = 200
            else:
                tick_interval = 250

            base_ticks = list(range(0, int(self.max_time) + 1, tick_interval))

            # Add makespan if far enough from last tick
            if (
                base_ticks
                and abs(self.max_time - base_ticks[-1]) >= tick_interval * 0.3
            ):
                base_ticks.append(self.max_time)
            elif not base_ticks:
                base_ticks.append(self.max_time)

            ax.set_xticks(base_ticks)
            ax.set_xticklabels(
                [
                    f"{int(t)}"
                    if abs(t - self.max_time) > 0.5
                    else f"{self.max_time:.0f}"
                    for t in base_ticks
                ]
            )
        else:
            ax.set_xlim(0, 100)

        ax.invert_yaxis()
        ax.grid(True, axis="x", linestyle=":", color="gray", alpha=0.6)

        # Group separators disabled (no longer separating by APR/OR/ARR)

    def _add_legend(self, ax):
        """Adds segment legend and an explanatory note for the transition model."""
        # Entrada de leyenda para las líneas de conexión op1→op2
        connector_legend_entry = plt.Line2D(
            [0],
            [0],
            color=GANTT_STYLE["connector_color"],
            linewidth=GANTT_STYLE["connector_lw"] + 0.3,
            linestyle=(0, (3, 3)),
            alpha=0.9,
            label="Op1 → Op2 connector (same job)",
        )

        handles = [
            Patch(
                facecolor="none",
                edgecolor="black",
                label="Setup (OR → Anesthesia)",
                hatch="///",
            ),
            Patch(
                facecolor="grey",
                edgecolor="black",
                label="Processing (Color per Job/CIE10)",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                label="Transition (Holding → OR)",
                hatch="\\\\\\",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                label="Cleanup (OR Exit — Op2 only)",
                hatch="|||",
            ),
        ]

        # Leyenda compacta debajo del eje (fuera del área útil del Gantt).
        # Moverla a "lower center" con bbox_to_anchor libera el cuadrante
        # superior-derecho del eje, ganando espacio visual real para las barras.
        legend = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, self.fonts["legend_anchor_y"]),
            ncol=min(len(handles), self.fonts["legend_ncol"]),
            fontsize=self.fonts["legend"],
            title="Segments:",
            framealpha=0.85,
            borderpad=0.5,
        )
    def _add_job_panel(self, fig, ax):
        """
        Añade un panel lateral compacto (fuera del eje Gantt) que lista
        todos los jobs/CIE10 con su color asignado.

        Se activa solo cuando hay un job_label_map real (modo PKL).
        En modo sintético o sin map, se omite silenciosamente.

        Diseño:
        - Posicionado a la derecha del eje principal usando figure-level coords.
        - Cada fila: [■ cuadro de color]  CIE10_label
        - Fuente pequeña, fondo semi-transparente.
        """
        # Solo añadir cuando tenemos CIE10 real y hay jobs que mostrar
        if not self.job_label_map:
            return

        # Construir lista de (job_id, label, color) ordenada por label
        entries = []
        for job_id, label in self.job_label_map.items():
            color = self.job_colors.get(job_id, "gray")
            entries.append((label, color))

        if not entries:
            return

        # Ordenar alfabéticamente por CIE10
        entries.sort(key=lambda x: x[0])

        # Construir el texto del panel usando un axes secundario invisible
        # posicionado en la franja derecha de la figura.
        # Usamos inset_axes para controlarlo sin romper el layout del axes principal.
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            # Panel más estrecho (12% del eje vs 18% anterior) y más pegado al borde
            # derecho del eje (1.005 vs 1.01). Como el eje ahora llega a right=0.88,
            # el panel cae en x≈0.89-0.97 de figura — cómodamente dentro del lienzo.
            panel_ax = inset_axes(
                ax,
                width="12%",
                height="100%",
                loc="center left",
                bbox_to_anchor=(1.005, 0.0, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
        except Exception:
            return  # Si inset_axes no está disponible, omitir silenciosamente

        panel_ax.axis("off")

        # Limitar a los primeros 30 para no desbordar el panel
        max_entries = 30
        shown = entries[:max_entries]
        truncated = len(entries) > max_entries

        # Calcular tamaño de fuente adaptativo
        font_size = max(7.5, min(9.5, int(70 / max(len(shown), 1))))

        panel_ax.set_xlim(0, 1)
        panel_ax.set_ylim(0, 1)

        title_y = 0.99
        panel_ax.text(
            0.0,
            title_y,
            "Jobs / CIE10",
            transform=panel_ax.transAxes,
            fontsize=font_size + 1,
            fontweight="bold",
            va="top",
            ha="left",
            color="#333333",
        )

        step = (0.95) / max(len(shown) + (1 if truncated else 0), 1)
        for i, (label, color) in enumerate(shown):
            y_frac = title_y - (i + 1) * step
            # Color swatch
            panel_ax.add_patch(
                Rectangle(
                    (0.0, y_frac - step * 0.35),
                    0.12,
                    step * 0.7,
                    transform=panel_ax.transAxes,
                    facecolor=color,
                    edgecolor="#555555",
                    linewidth=0.4,
                    clip_on=False,
                )
            )
            panel_ax.text(
                0.16,
                y_frac,
                label,
                transform=panel_ax.transAxes,
                fontsize=font_size,
                va="center",
                ha="left",
                color="#222222",
                clip_on=False,
            )

        if truncated:
            remaining = len(entries) - max_entries
            y_frac = title_y - (len(shown) + 1) * step
            panel_ax.text(
                0.0,
                y_frac,
                f"(+{remaining} more)",
                transform=panel_ax.transAxes,
                fontsize=font_size - 1,
                va="center",
                ha="left",
                color="gray",
                style="italic",
                clip_on=False,
            )

    def build(self, title: Optional[str], sim_num: Optional[int] = None) -> plt.Figure:
        """
        Builds and returns the complete Gantt chart figure.

        Args:
            title: Chart title, or None to omit the embedded title
            sim_num: Simulation number (optional)

        Returns:
            matplotlib Figure object
        """
        n_rooms = max(len(self.rooms_with_tasks), 1)
        has_panel = bool(self.job_label_map)

        # Adapt the canvas height to the number of active rooms. A side panel
        # gets additional width so labels and the legend remain readable.
        adaptive_height = max(PlotConfig.GANTT_FIGSIZE[1], n_rooms * 0.65 + 2.5)
        adaptive_width = 18 if has_panel else 15

        fig, ax = plt.subplots(figsize=(adaptive_width, adaptive_height))

        # Reset etiquetas externas para esta figura
        self._ext_label_positions = {}

        # Draw all components
        # Connectors first (zorder=1) so they render below bars (zorder=2+)
        self._draw_job_connectors(ax)

        for task in self.schedule_details:
            self._draw_task_bar(ax, task)

        self._draw_unused_rooms(ax)

        self._configure_axes(ax, title, sim_num)
        if self.show_legend:
            self._add_legend(ax)

        if has_panel:
            self._add_job_panel(fig, ax)
            # Layout con panel lateral:
            # - left=0.06   → margen izquierdo (etiquetas de salas)
            # - right=0.88  → el eje Gantt ocupa 82% del ancho de figura
            #                  (antes 0.82, ganancia neta +6 pp de ancho útil REAL)
            # - El panel inset (12% del eje relativo) queda aprox. en x≈0.89-0.98
            #   → cabe limpio sin recortar nada
            # - bottom=0.22 → espacio para leyenda horizontal debajo del eje
            #   (antes 0.10; la leyenda migró de "upper right" interno a debajo)
            fig.subplots_adjust(right=0.88, left=0.06, top=0.92, bottom=0.22)
            fig._skip_tight_layout = True
        else:
            # Sin panel: márgenes explícitos, leyenda también debajo
            fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.22)
            fig._skip_tight_layout = True

        return fig


# =============================================================================
# HIGH-LEVEL PLOTTING FUNCTIONS (PUBLIC API)
# =============================================================================


def plot_gantt_chart(
    schedule_details: List[Dict],
    rooms: List[str],
    title: str,
    algo_name: str,
    output_dir: str,
    sim_num: Optional[int] = None,
    job_label_map: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Render and save a Gantt chart for an elective schedule.

    Args:
        schedule_details: List of task dictionaries
        rooms: List of room names
        title: Chart title
        algo_name: Algorithm name (for filename)
        output_dir: Output directory
        sim_num: Simulation number (optional, for title)
        job_label_map: Optional dict {job_id -> display_label} for CIE10 labels.
                       Falls back to str(job_id) for missing entries.

    Returns:
        Tuple of (png_path, pdf_path)
    """
    if not schedule_details:
        logger.warning(f"No schedule details for {algo_name} Gantt chart")
        return None, None

    filename = f"best_gantt_{algo_name.lower()}"

    # Build chart
    builder = GanttChartBuilder(
        schedule_details, rooms, job_label_map=job_label_map
    )
    fig = builder.build(title, sim_num)

    # Save — when a CIE10 side panel is present, trim trailing right-side
    # whitespace in post-process (PIL pixel scan).  matplotlib's bbox_inches='tight'
    # does NOT capture inset_axes elements drawn outside the main axes with
    # clip_on=False, so we do a pixel-level right-edge crop instead.
    # trim_pad_px=18 keeps a ~0.12" clean border after the panel.
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "gantt", filename)
    has_panel = bool(job_label_map)
    PlotConfig.save_and_close(
        fig,
        png_path,
        pdf_path,
        trim_right=has_panel,
        trim_pad_px=18,
    )

    return png_path, pdf_path


def plot_boxplot(
    all_results: Dict,
    output_dir: str,
    show_title: bool = True,
    axis_label_size: float = 18.0,
    tick_label_size: float = 15.0,
) -> Tuple[str, str]:
    """Render a makespan comparison boxplot for elective simulations.

    Args:
        all_results: Dictionary with algorithm results
        output_dir: Output directory
    Returns:
        Tuple of (png_path, pdf_path)
    """
    if not all_results:
        logger.warning("No results for elective boxplot")
        return None, None

    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)

    data_to_plot = []
    labels = []

    for algo_name in ALGORITHM_ORDER:
        data_name = ALGORITHM_DATA_KEYS[algo_name]
        if data_name in all_results:
            makespans = [
                mk for mk in all_results[data_name]["makespan"] if mk != float("inf")
            ]
            if makespans:
                data_to_plot.append(makespans)
                labels.append(algo_name)

    if not data_to_plot:
        logger.warning("No valid data for elective boxplot")
        plt.close(fig)
        return None, None

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)

    for patch, label in zip(bp["boxes"], labels):
        color = ALGORITHM_COLORS.get(label, "mediumpurple")
        patch.set_facecolor(color)

    ax.set_ylabel("Makespan (minutes)", fontsize=18.0)
    ax.tick_params(axis="both", labelsize=16.0)
    ax.set_xticklabels(labels, fontsize=16.0)
    if show_title:
        title = "Elective Simulation: Makespan Comparison"
        ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    filename = "elective_makespan_comparison"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "boxplot", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)

    return png_path, pdf_path


def plot_execution_time_barplot(all_results: Dict, output_dir: str) -> str:
    """Render an execution-time barplot for elective simulations.

    Args:
        all_results: Dictionary with algorithm results
        output_dir: Output directory
    Returns:
        Path to saved PNG file
    """
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)

    avg_times = []
    algo_names = []
    colors = []

    for algo_name in ALGORITHM_ORDER:
        data_name = ALGORITHM_DATA_KEYS[algo_name]
        if data_name in all_results and all_results[data_name].get("time"):
            avg_times.append(np.mean(all_results[data_name]["time"]))
            algo_names.append(algo_name)
            colors.append(ALGORITHM_COLORS[algo_name])

    if not avg_times:
        plt.close(fig)
        return None

    bars = ax.bar(algo_names, avg_times, color=colors, edgecolor="black", linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(
        "Average Execution Time (seconds)",
        fontsize=PlotConfig.DEFAULT_AXIS_LABEL_SIZE,
    )
    title = "Elective Simulation: Average Execution Time per Algorithm"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    filename = "elective_execution_time"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "barplot", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)

    return png_path


def plot_makespan_histogram(
    data: List[float], algo_name: str, output_dir: str
) -> Tuple[str, str]:
    """Render a makespan histogram for one elective algorithm.

    Args:
        data: List of makespan values
        algo_name: Algorithm name
        output_dir: Output directory
    Returns:
        Tuple of (png_path, pdf_path)
    """
    if not data:
        logger.warning(f"No data for {algo_name} histogram")
        return None, None

    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)

    hist_color = ALGORITHM_COLORS.get(algo_name, "steelblue")

    ax.hist(
        data, bins=20, color=hist_color, edgecolor="black", alpha=0.8, linewidth=1.2
    )

    mean_val = np.mean(data)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )

    median_val = np.median(data)
    ax.axvline(
        median_val,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.2f}",
    )

    ax.set_xlabel("Makespan (minutes)", fontsize=PlotConfig.DEFAULT_AXIS_LABEL_SIZE)
    ax.set_ylabel("Frequency", fontsize=PlotConfig.DEFAULT_AXIS_LABEL_SIZE)

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    title = f"Elective Simulation: {algo_name} - Makespan Distribution"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=PlotConfig.DEFAULT_LEGEND_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    filename = f"histogram_{algo_name.lower()}"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "histograms", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)

    return png_path, pdf_path


def plot_convergence_history(
    best_history: List[float],
    avg_history: List[float],
    max_iters: int,
    algo_name: str,
    sim_num: int,
    output_dir: str,
    metric_name: str = "Fitness",
) -> Tuple[str, str]:
    """
    Plots the convergence curve.

    Args:
        best_history: Best fitness/makespan history
        avg_history: Average/Iteration fitness/makespan history
        max_iters: Maximum iterations
        algo_name: Algorithm name
        sim_num: Simulation number
        output_dir: Output directory
        metric_name: "Fitness" or "Makespan"

    Returns:
        Tuple of (png_path, pdf_path)
    """
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)

    valid_best = [(i, f) for i, f in enumerate(best_history) if f != float("inf")]
    valid_avg = [(i, f) for i, f in enumerate(avg_history) if f != float("inf")]

    if metric_name == "Makespan":
        best_label = f"Best Makespan ({algo_name})"
        avg_label = f"Iteration Makespan ({algo_name})"
        y_label = "Makespan (minutes)"
        title = f"{algo_name} Makespan Evolution (Simulation #{sim_num})"
        file_suffix = "_makespan"
    else:
        best_label = f"Best Fitness ({algo_name})"
        avg_label = f"Average Fitness ({algo_name})"
        y_label = "Objective Value"
        title = f"{algo_name} Evolution (Simulation #{sim_num})"
        file_suffix = ""

    if valid_best:
        iterations, values = zip(*valid_best)
        ax.plot(
            iterations,
            values,
            label=best_label,
            linestyle="-",
            drawstyle="steps-post",
        )

    if valid_avg:
        iterations, values = zip(*valid_avg)
        ax.plot(
            iterations, values, label=avg_label, linestyle="--"
        )

    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(0, max_iters)

    finite_vals = [f for f in (best_history + avg_history) if f != float("inf")]
    if finite_vals:
        min_val, max_val = min(finite_vals), max(finite_vals)
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 1
        ax.set_ylim(max(0, min_val - padding), max_val + padding)

    step = max(1, max_iters // 10)
    ax.set_xticks(np.arange(0, max_iters + 1, step))
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()

    filename = f"{algo_name.lower()}_convergence{file_suffix}_sim_{sim_num}"
    png_path, pdf_path = PlotConfig.get_output_paths(
        output_dir, "convergence", filename
    )
    PlotConfig.save_and_close(fig, png_path, pdf_path)

    return png_path, pdf_path


def plot_personnel_workload(
    personnel_stats: Dict,
    output_dir: str,
    algo_name: str,
    sim_num: Optional[int] = None,
) -> Tuple[str, str]:
    if not personnel_stats:
        return None, None

    anestesiologos = {k: v for k, v in personnel_stats.items() if k.startswith("A")}
    cirujanos = {k: v for k, v in personnel_stats.items() if k.startswith("S")}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

    suffix = f" (Sim #{sim_num})" if sim_num is not None else ""
    fig.suptitle(
        f"Personnel Workload Analysis - {algo_name}{suffix}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    anest_names = sorted(anestesiologos.keys())
    ciruj_names = sorted(cirujanos.keys())

    anest_util = [anestesiologos[n]["utilization"] for n in anest_names]
    ciruj_util = [cirujanos[n]["utilization"] for n in ciruj_names]

    x_anest = np.arange(len(anest_names))
    x_ciruj = np.arange(len(ciruj_names))
    width = 0.35

    # Ax1: Utilization
    bars1 = ax1.bar(
        x_anest,
        anest_util,
        width,
        color="#EF5350",
        alpha=0.8,
        edgecolor="black",
        label="Anesthesiologists",
    )
    bars2 = ax1.bar(
        x_ciruj + len(anest_names) + 0.5,
        ciruj_util,
        width,
        color="#FFA726",
        alpha=0.8,
        edgecolor="black",
        label="Surgeons",
    )

    ax1.axhline(
        y=80,
        color="green",
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
        label="Target: 80%",
    )
    ax1.set_ylabel("Utilization Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Utilization Rate per Personnel", fontsize=12, fontweight="bold")
    ax1.set_xticks(list(x_anest) + [len(anest_names) + 0.5 + i for i in x_ciruj])
    ax1.set_xticklabels(
        list(anest_names) + list(ciruj_names), rotation=45, ha="right", fontsize=9
    )
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Ax2: Operations
    anest_ops = [anestesiologos[n]["operations"] for n in anest_names]
    ciruj_ops = [cirujanos[n]["operations"] for n in ciruj_names]
    bars3 = ax2.bar(
        x_anest,
        anest_ops,
        width,
        color="#EF5350",
        alpha=0.8,
        edgecolor="black",
        label="Anesthesiologists",
    )
    bars4 = ax2.bar(
        x_ciruj + len(anest_names) + 0.5,
        ciruj_ops,
        width,
        color="#FFA726",
        alpha=0.8,
        edgecolor="black",
        label="Surgeons",
    )
    avg_anest = np.mean(anest_ops) if anest_ops else 0
    avg_ciruj = np.mean(ciruj_ops) if ciruj_ops else 0
    ax2.axhline(
        y=avg_anest,
        color="#B71C1C",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"Avg Anest: {avg_anest:.1f}",
    )
    ax2.axhline(
        y=avg_ciruj,
        color="#E65100",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"Avg Surg: {avg_ciruj:.1f}",
    )
    ax2.set_ylabel("Number of Operations", fontsize=11, fontweight="bold")
    ax2.set_title("Operations per Personnel", fontsize=12, fontweight="bold")
    ax2.set_xticks(list(x_anest) + [len(anest_names) + 0.5 + i for i in x_ciruj])
    ax2.set_xticklabels(
        list(anest_names) + list(ciruj_names), rotation=45, ha="right", fontsize=9
    )
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # Ax3: Work vs Idle (Anest)
    anest_work = [anestesiologos[n]["total_time"] for n in anest_names]
    anest_idle = [anestesiologos[n]["idle_time"] for n in anest_names]
    ax3.bar(
        x_anest,
        anest_work,
        width,
        label="Work Time",
        color="#EF5350",
        alpha=0.8,
        edgecolor="black",
    )
    ax3.bar(
        x_anest,
        anest_idle,
        width,
        bottom=anest_work,
        label="Idle Time",
        color="#BDBDBD",
        alpha=0.8,
        edgecolor="black",
    )
    ax3.set_ylabel("Time (minutes)", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Work vs Idle Time - Anesthesiologists", fontsize=12, fontweight="bold"
    )
    ax3.set_xticks(x_anest)
    ax3.set_xticklabels(list(anest_names), rotation=45, ha="right", fontsize=9)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(axis="y", linestyle=":", alpha=0.5)
    max_anest = (
        max([w + i for w, i in zip(anest_work, anest_idle)]) if anest_work else 0
    )
    ax3.set_ylim(0, max_anest * 1.10 if max_anest > 0 else 100)

    # Ax4: Work vs Idle (Surg)
    ciruj_work = [cirujanos[n]["total_time"] for n in ciruj_names]
    ciruj_idle = [cirujanos[n]["idle_time"] for n in ciruj_names]
    ax4.bar(
        x_ciruj,
        ciruj_work,
        width,
        label="Work Time",
        color="#FFA726",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.bar(
        x_ciruj,
        ciruj_idle,
        width,
        bottom=ciruj_work,
        label="Idle Time",
        color="#BDBDBD",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.set_ylabel("Time (minutes)", fontsize=11, fontweight="bold")
    ax4.set_title("Work vs Idle Time - Surgeons", fontsize=12, fontweight="bold")
    ax4.set_xticks(x_ciruj)
    ax4.set_xticklabels(list(ciruj_names), rotation=45, ha="right", fontsize=9)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(axis="y", linestyle=":", alpha=0.5)
    max_ciruj = (
        max([w + i for w, i in zip(ciruj_work, ciruj_idle)]) if ciruj_work else 0
    )
    ax4.set_ylim(0, max_ciruj * 1.10 if max_ciruj > 0 else 100)

    fig.subplots_adjust(top=0.88, hspace=0.40, wspace=0.3)
    fig._skip_tight_layout = True

    filename = f"personnel_workload_{algo_name.lower()}"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "personnel", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)
    return png_path, pdf_path


def plot_kpi_histogram(
    kpis_dict: Dict, output_dir: str, algo_name: str
) -> Tuple[str, str]:
    if not kpis_dict:
        return None, None

    avg_rate = kpis_dict.get("Average", {}).get("occupancy_rate", 0.0)
    pabellones = sorted([k for k in kpis_dict.keys() if k != "Average"])
    rates = [kpis_dict[k]["occupancy_rate"] for k in pabellones]

    # Sort by rate descending for visual impact
    paired = sorted(zip(pabellones, rates), key=lambda x: x[1], reverse=True)
    pabellones = [p for p, _ in paired]
    rates = [r for _, r in paired]

    # Gradient colormap: low=cool blue, mid=teal, high=warm green
    cmap = plt.get_cmap("RdYlGn")
    max_rate = max(rates) if rates else 100
    colors = [cmap(r / max(max_rate, 1)) for r in rates]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(
        range(len(pabellones)),
        rates,
        width=0.7,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.axhline(
        y=avg_rate,
        color="#1565C0",
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg_rate:.1f}%",
    )
    ax.axhline(
        y=80,
        color="#2E7D32",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label="Target: 80%",
    )

    for i, rate in enumerate(rates):
        ax.text(
            i,
            rate + (max_rate * 0.02),
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    # Clean labels
    short_labels = [p.replace("Pabellon_", "OR ") for p in pabellones]
    ax.set_xticks(range(len(pabellones)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Operating Room", fontsize=12, fontweight="bold")
    ax.set_ylabel("Occupancy Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Operating Room Occupancy Rate - {algo_name}", fontsize=14, fontweight="bold"
    )
    ax.set_ylim(0, max(max_rate * 1.18, 10))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    filename = f"kpi_occupancy_{algo_name.lower()}"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "histograms", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)
    return png_path, pdf_path


def plot_cie10_histogram(
    schedule_details: List[Dict], output_dir: str, algo_name: str, top_k: int = 40
) -> Tuple[str, str]:
    """Shows the percentage of total processing time consumed by each Job (CIE10)."""
    job_times = {}

    for t in schedule_details:
        job = t.get("Job")
        if job is None:
            continue
        start = float(t.get("Start", 0.0))
        finish = float(t.get("Finish", start))
        dur = max(0.0, finish - start)

        job_key = str(job)
        if job_key in job_times:
            job_times[job_key] += dur
        else:
            job_times[job_key] = dur

    if not job_times:
        return None, None

    # Sort by time descending, take top_k
    sorted_jobs = sorted(job_times.items(), key=lambda x: x[1], reverse=True)[:top_k]
    labels = [k for k, _ in sorted_jobs]
    times = [v for _, v in sorted_jobs]
    total_time = sum(times)

    if total_time <= 0:
        return None, None

    perc = [(t / total_time) * 100 for t in times]
    avg = sum(perc) / len(perc) if perc else 0.0
    max_perc = max(perc) if perc else 1

    # Gradient colormap matching KPI style
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(p / max(max_perc, 1)) for p in perc]

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.6), 7))
    bars = ax.bar(
        range(len(labels)),
        perc,
        width=0.7,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.axhline(
        y=avg,
        color="#1565C0",
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg:.2f}%",
    )

    for i, p in enumerate(perc):
        ax.text(
            i,
            p + (max_perc * 0.02),
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [f"Job {l}" for l in labels], rotation=45, ha="right", fontsize=10
    )
    ax.set_xlabel("Job ID (CIE10)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time Usage (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Processing Time by Job - {algo_name}", fontsize=14, fontweight="bold"
    )
    ax.set_ylim(0, max_perc * 1.20)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    filename = f"job_usage_{algo_name.lower()}"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "histograms", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)
    return png_path, pdf_path


def plot_personnel_usage_histogram(
    schedule_details: List[Dict], output_dir: str, algo_name: str
) -> Tuple[str, str]:
    usage = {}
    for t in schedule_details:
        person = t.get("Personnel")
        if not person:
            continue
        start = float(t.get("Start", 0.0))
        finish = float(t.get("Finish", start))
        dur = max(0.0, finish - start)
        if person in usage:
            usage[person] += dur
        else:
            usage[person] = dur

    if not usage:
        return None, None

    anest_people = sorted([p for p in usage if p.startswith("A")])
    ciruj_people = sorted([p for p in usage if p.startswith("S")])
    ordered_people = anest_people + ciruj_people

    times = [usage.get(p, 0.0) for p in ordered_people]
    total_time = sum(times)
    if total_time <= 0:
        return None, None

    perc = [(t / total_time) * 100 for t in times]
    avg = sum(perc) / len(perc) if perc else 0.0

    # Build x positions with a gap between groups
    gap = 0.8
    x_positions = []
    for i, p in enumerate(ordered_people):
        offset = gap if i >= len(anest_people) else 0
        x_positions.append(i + offset)

    # Assign colors by group
    colors = ["#EF5350" if p.startswith("A") else "#FFA726" for p in ordered_people]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(
        x_positions,
        perc,
        width=0.7,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
    )

    ax.axhline(
        y=avg,
        color="#1565C0",
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg:.2f}%",
    )

    for x, p in zip(x_positions, perc):
        ax.text(
            x,
            p + (max(perc) * 0.02 if max(perc) > 0 else 0.5),
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Separator line between groups
    if anest_people and ciruj_people:
        sep_x = len(anest_people) - 0.5 + gap / 2
        ax.axvline(x=sep_x, color="gray", linestyle=":", linewidth=1.5, alpha=0.6)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_people, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Personnel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time Usage (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Personnel Time Usage - {algo_name}", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(perc) * 1.18 if perc else 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Legend with group labels
    # Patch already imported at top-level
    legend_items = [
        Patch(facecolor="#EF5350", edgecolor="black", label="Anesthesiologists"),
        Patch(facecolor="#FFA726", edgecolor="black", label="Surgeons"),
        plt.Line2D(
            [0],
            [0],
            color="#1565C0",
            linestyle="--",
            linewidth=2,
            label=f"Average: {avg:.2f}%",
        ),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=10)

    filename = f"personnel_usage_{algo_name.lower()}"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "personnel", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)
    return png_path, pdf_path


def plot_personnel_gantt(
    personnel_stats: Dict, schedule_details: List[Dict], output_dir: str, algo_name: str
) -> Tuple[str, str]:
    if not personnel_stats or not schedule_details:
        return None, None

    anestesiologos = sorted([p for p in personnel_stats if p.startswith("A")])
    cirujanos = sorted([p for p in personnel_stats if p.startswith("S")])

    active_personnel = [
        p for p in (anestesiologos + cirujanos) if personnel_stats[p]["operations"] > 0
    ]
    if not active_personnel:
        return None, None

    unique_jobs = sorted(
        set(task["Job"] for task in schedule_details if "Job" in task), key=str
    )
    try:
        cmap = (
            plt.get_cmap("tab20") if len(unique_jobs) <= 20 else plt.get_cmap("turbo")
        )
    except:
        cmap = plt.get_cmap("tab10")

    job_colors = {
        job: cmap(i / max(1, len(unique_jobs) - 1)) for i, job in enumerate(unique_jobs)
    }
    y_index = {p: i for i, p in enumerate(active_personnel)}

    fig, ax = plt.subplots(figsize=(16, max(8, len(active_personnel) * 0.5)))
    bar_height = 0.6
    max_time = 0.0

    for task in schedule_details:
        pers = task.get("Personnel")
        if pers not in y_index:
            continue

        job = task["Job"]
        start = task["Start"]
        finish = task["Finish"]
        duration = finish - start
        op = task.get("Operation", 1)

        if finish > max_time:
            max_time = finish

        y = y_index[pers]
        color = job_colors.get(job, "gray")

        rect = Rectangle(
            (start, y - bar_height / 2),
            duration,
            bar_height,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_patch(rect)

        label = f"A-{job}" if op == 1 else f"S-{job}"
        if duration > 8:
            luminance = (
                np.dot(mcolors.to_rgb(color), [0.299, 0.587, 0.114])
                if isinstance(color, tuple)
                else 0.5
            )
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(
                start + duration / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=text_color,
            )

    # Set axis limits explicitly (Rectangle patches don't auto-scale axes)
    ax.set_xlim(0, max_time * 1.05 if max_time > 0 else 100)
    ax.set_ylim(-0.5, len(active_personnel) - 0.5)

    ax.set_yticks(range(len(active_personnel)))
    ax.set_yticklabels(active_personnel, fontsize=9)
    ax.set_xlabel("Time (min)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Personnel", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Personnel Schedule Gantt Chart - {algo_name}", fontsize=14, fontweight="bold"
    )

    num_anest = len([p for p in active_personnel if p.startswith("A")])
    if 0 < num_anest < len(active_personnel):
        ax.axhline(
            y=num_anest - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7
        )
        ax.text(
            max_time * 1.02,
            num_anest - 0.7,
            "Anesthesiologists ↑",
            ha="right",
            color="red",
            fontsize=9,
        )
        ax.text(
            max_time * 1.02,
            num_anest + 0.2,
            "Surgeons ↓",
            ha="right",
            color="red",
            fontsize=9,
        )

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.invert_yaxis()

    filename = f"personnel_gantt_{algo_name.lower()}"
    png_path, pdf_path = PlotConfig.get_output_paths(output_dir, "gantt", filename)
    PlotConfig.save_and_close(fig, png_path, pdf_path)
    return png_path, pdf_path


# =============================================================================
# SUMMARY PLOT GENERATION (ORCHESTRATION)
# =============================================================================


def generate_summary_plots(all_results: Dict, output_dir: str):
    """Generate all summary plots for elective simulations."""
    logger.info("  -> Generating elective summary plots...")

    # Boxplot
    png_path, _ = plot_boxplot(all_results, output_dir)
    if png_path:
        logger.info(f"    - Elective boxplot saved to: {png_path}")

    # Execution time barplot
    png_path = plot_execution_time_barplot(all_results, output_dir)
    if png_path:
        logger.info(f"    - Execution time barplot saved to: {png_path}")

    # Histograms per algorithm
    for algo_name in ALGORITHM_ORDER:
        data_name = ALGORITHM_DATA_KEYS[algo_name]
        if data_name not in all_results:
            continue

        makespans = [
            mk for mk in all_results[data_name]["makespan"] if mk != float("inf")
        ]

        if len(makespans) >= 2:
            png_path, _ = plot_makespan_histogram(
                makespans, algo_name, output_dir
            )
            if png_path and VERBOSE_MODE:
                logger.info(
                    f"    - Elective histogram for {algo_name} saved to: {png_path}"
                )


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


def plot_comparison_boxplot(all_results: Dict, output_dir: str) -> Tuple[str, str]:
    """Alias for backward compatibility."""
    return plot_boxplot(all_results, output_dir)
