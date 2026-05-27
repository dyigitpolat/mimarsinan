"""Architecture-search result plots and reports."""

from mimarsinan.visualization.search_viz.history import (
    plot_history_best_metrics,
    plot_history_metrics_separate,
)
from mimarsinan.visualization.search_viz.population import write_final_population_json
from mimarsinan.visualization.search_viz.html import create_interactive_search_report
from mimarsinan.visualization.search_viz.report_png import write_search_report_png
from mimarsinan.visualization.search_viz.scatter import plot_default_pareto_scatters, plot_scatter

__all__ = [
    "create_interactive_search_report",
    "plot_default_pareto_scatters",
    "plot_history_best_metrics",
    "plot_history_metrics_separate",
    "plot_scatter",
    "write_final_population_json",
    "write_search_report_png",
]
