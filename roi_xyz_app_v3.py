import sys
import os
import csv
import numpy as np

import matplotlib
# Force a GUI backend so the app is interactive (not PyCharm's static viewer)
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle

import tkinter as tk
from tkinter import filedialog


class XYZCircleROIApp:
    """
    ROI picker for XYZ (x, y, z) maps.

    Features
    --------
    - Peak vs Baseline selection (button toggle):
        * Peak: use a subset of extreme points (count or % of ROI).
        * Baseline: use all points in ROI.

    - Bright vs dark defects:
        * Compare ROI median z to global median z.
        * If ROI median < global median  -> treat as low-conductivity defect,
          pick lowest values in peak mode.
        * If ROI median >= global median -> treat as high-conductivity defect,
          pick highest values in peak mode.

    - Spatial weighting (button toggle):
        * Uniform: plain arithmetic mean of selected points.
        * Gaussian: 2D Gaussian weighting centered at ROI center, sigma ~ r/2.

    - Z-scale sliders:
        * Z min / Z max control color scaling.

    - Undo last ROI, CSV export.

    - Right panel:
        * Top: summary + page info.
        * Middle: paged list of ROIs (Prev/Next buttons).
        * Bottom: histogram of mean_z over all ROIs.
    """

    def __init__(
        self,
        filename: str,
        peak_selection_mode: str,
        peak_point_count: int | None,
        peak_percent: float | None,
    ):
        self.filename = filename

        # Peak selection config
        # peak_selection_mode: "count" or "percent"
        self.peak_selection_mode = peak_selection_mode
        self.peak_point_count = peak_point_count
        self.peak_percent = peak_percent  # fraction, e.g. 0.05 for 5%

        # Data containers
        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self.global_median_z: float | None = None
        self._load_xyz()

        # ROI state
        self.pending_center: tuple[float, float] | None = None
        self.temp_marker = None
        self.roi_results: list[dict] = []
        self.circle_patches: list[Circle] = []

        # Selection mode: "peak" or "baseline"
        self.selection_mode: str = "peak"

        # Weighting: False = uniform, True = gaussian
        self.use_gaussian_weight: bool = False

        # Paging for ROI list
        self.page_size: int = 15  # number of rows to show in middle list
        self.page_start: int = 0  # starting index for current page

        # Plot objects / widgets
        self.fig = None
        self.ax_img = None
        self.ax_info = None
        self.ax_list = None
        self.ax_hist = None
        self.im = None
        self.cbar = None

        self.btn_undo = None
        self.btn_export = None
        self.btn_mode = None
        self.btn_weight = None
        self.btn_prev = None
        self.btn_next = None

        self.slider_vmin = None
        self.slider_vmax = None

        self._build_figure()

    # ----------------------------
    # Data loading and reshaping
    # ----------------------------
    def _load_xyz(self):
        """Load XYZ data and reshape to 2D grid."""
        data = np.loadtxt(self.filename)
        if data.shape[1] < 3:
            raise ValueError("Input file must have at least 3 columns: x, y, z.")

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        x_unique = np.unique(x)
        y_unique = np.unique(y)

        nx = len(x_unique)
        ny = len(y_unique)

        if nx * ny != len(z):
            raise ValueError(
                "Data cannot be reshaped into a regular grid.\n"
                f"Unique x: {nx}, unique y: {ny}, total points: {len(z)}.\n"
                "This script assumes a uniform x-y grid."
            )

        # Map (x, y) to grid indices
        xi = np.searchsorted(x_unique, x)
        yi = np.searchsorted(y_unique, y)

        Z = np.empty((ny, nx))
        Z[yi, xi] = z

        X, Y = np.meshgrid(x_unique, y_unique)

        self.X = X
        self.Y = Y
        self.Z = Z
        self.global_median_z = float(np.median(self.Z))

    # ----------------------------
    # Figure and UI setup
    # ----------------------------
    def _build_figure(self):
        # Build figure with a left image and a right panel split into 3 rows
        self.fig = plt.figure(figsize=(11, 7))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        self.fig.subplots_adjust(bottom=0.28, wspace=0.35)

        # Left: image axis
        self.ax_img = self.fig.add_subplot(gs[0, 0])

        # Right: nested grid with 3 rows: info, list, histogram
        gs_right = gs[0, 1].subgridspec(3, 1, height_ratios=[0.7, 1.7, 1.3])
        self.ax_info = self.fig.add_subplot(gs_right[0])
        self.ax_list = self.fig.add_subplot(gs_right[1])
        self.ax_hist = self.fig.add_subplot(gs_right[2])

        self.ax_info.axis("off")
        self.ax_list.axis("off")
        # ax_hist will show histogram; keep axes visible

        # Plot the 2D map
        extent = [
            float(self.X.min()),
            float(self.X.max()),
            float(self.Y.min()),
            float(self.Y.max()),
        ]

        z_min = float(self.Z.min())
        z_max = float(self.Z.max())

        self.im = self.ax_img.imshow(
            self.Z,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="turbo",  # change to "nipy_spectral", "jet", etc if you like
            vmin=z_min,
            vmax=z_max,
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_img)
        self.cbar.set_label("Current (A)")

        self.ax_img.set_xlabel("x (arb. units)")
        self.ax_img.set_ylabel("y (arb. units)")
        self.ax_img.set_title(
            "Left-click: center, left-click again: radius\n"
            "Right-click: cancel current center"
        )

        # Buttons along the bottom: Undo, Mode, Weight, Prev, Next, Export
        ax_undo = self.fig.add_axes([0.03, 0.05, 0.13, 0.06])
        ax_mode = self.fig.add_axes([0.18, 0.05, 0.13, 0.06])
        ax_weight = self.fig.add_axes([0.33, 0.05, 0.13, 0.06])
        ax_prev = self.fig.add_axes([0.50, 0.05, 0.08, 0.06])
        ax_next = self.fig.add_axes([0.60, 0.05, 0.08, 0.06])
        ax_export = self.fig.add_axes([0.71, 0.05, 0.16, 0.06])

        self.btn_undo = Button(ax_undo, "Undo last ROI")
        self.btn_mode = Button(ax_mode, "Mode: Peak")  # starts in peak mode
        self.btn_weight = Button(ax_weight, "Weight: uniform")
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.btn_export = Button(ax_export, "Export CSV")

        self.btn_undo.on_clicked(self.on_undo)
        self.btn_mode.on_clicked(self.on_toggle_mode)
        self.btn_weight.on_clicked(self.on_toggle_weight)
        self.btn_prev.on_clicked(self.on_prev_page)
        self.btn_next.on_clicked(self.on_next_page)
        self.btn_export.on_clicked(self.on_export)

        # Sliders for Z-scale (under the plot)
        ax_vmin = self.fig.add_axes([0.10, 0.16, 0.60, 0.03])
        ax_vmax = self.fig.add_axes([0.10, 0.12, 0.60, 0.03])

        self.slider_vmin = Slider(
            ax=ax_vmin,
            label="Z min",
            valmin=z_min,
            valmax=z_max,
            valinit=z_min,
            valstep=(z_max - z_min) / 500 if z_max > z_min else None,
        )
        self.slider_vmax = Slider(
            ax=ax_vmax,
            label="Z max",
            valmin=z_min,
            valmax=z_max,
            valinit=z_max,
            valstep=(z_max - z_min) / 500 if z_max > z_min else None,
        )

        self.slider_vmin.on_changed(self.on_zscale_change)
        self.slider_vmax.on_changed(self.on_zscale_change)

        # Mouse event
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # Initialize empty panels
        self._update_table()

    # ----------------------------
    # Z-scale update
    # ----------------------------
    def on_zscale_change(self, _val):
        """Update image and colorbar when Z min/max sliders move."""
        vmin = self.slider_vmin.val
        vmax = self.slider_vmax.val

        # Simple guard: if user drags min past max, do nothing
        if vmin >= vmax:
            return

        self.im.set_clim(vmin=vmin, vmax=vmax)
        self.cbar.update_normal(self.im)
        self.fig.canvas.draw_idle()

    # ----------------------------
    # Mouse interaction
    # ----------------------------
    def on_click(self, event):
        """Handle mouse clicks on the image for circle ROI definition."""
        if event.inaxes != self.ax_img:
            return

        if event.xdata is None or event.ydata is None:
            return

        # If the toolbar is in zoom/pan mode, don't create ROIs
        toolbar = getattr(event.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            # toolbar.mode is non-empty when zoom/pan is active
            return

        # Right-click cancels pending center
        if event.button == 3:
            if self.pending_center is not None:
                self.pending_center = None
                if self.temp_marker is not None:
                    self.temp_marker.remove()
                    self.temp_marker = None
                    self.fig.canvas.draw_idle()
            return

        # Only use left-click for ROI creation
        if event.button != 1:
            return

        x_click, y_click = event.xdata, event.ydata

        if self.pending_center is None:
            # First click: set center
            self.pending_center = (x_click, y_click)
            # Draw temporary marker
            if self.temp_marker is not None:
                self.temp_marker.remove()
            (self.temp_marker,) = self.ax_img.plot(
                x_click, y_click, "r+", markersize=10, markeredgewidth=2
            )
            self.fig.canvas.draw_idle()
        else:
            # Second click: set radius and finalize circle
            xc, yc = self.pending_center
            dx = x_click - xc
            dy = y_click - yc
            r = np.hypot(dx, dy)

            # Reset pending state and marker
            self.pending_center = None
            if self.temp_marker is not None:
                self.temp_marker.remove()
                self.temp_marker = None

            if r <= 0:
                # Degenerate radius, ignore
                self.fig.canvas.draw_idle()
                return

            # Add ROI
            self._add_roi(xc, yc, r)

    # ----------------------------
    # ROI processing and panels
    # ----------------------------
    def _add_roi(self, xc: float, yc: float, r: float):
        """Compute ROI stats, draw circle, update table."""
        # Circle patch on the image
        circle = Circle(
            (xc, yc),
            r,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
        )
        self.ax_img.add_patch(circle)
        self.circle_patches.append(circle)

        # Compute mask for points inside circle
        mask = (self.X - xc) ** 2 + (self.Y - yc) ** 2 <= r**2
        values = self.Z[mask]
        x_roi = self.X[mask]
        y_roi = self.Y[mask]

        total_pts = values.size
        if total_pts == 0:
            # No data points inside circle. Remove patch and exit.
            circle.remove()
            self.circle_patches.pop()
            print("Warning: circle contains no data points; ROI ignored.")
            self.fig.canvas.draw_idle()
            return

        # Median-based polarity: compare ROI median to global median
        roi_median = np.median(values)
        if roi_median < self.global_median_z:
            polarity = "low"   # lower conductivity than background
            ascending = True   # interesting stuff at lower end
        else:
            polarity = "high"  # higher conductivity than background
            ascending = False  # interesting stuff at upper end

        # Decide which points are selected (baseline vs peak)
        if self.selection_mode == "baseline":
            sel_idx = np.arange(total_pts)
        else:
            # Peak mode: use count or percent
            if self.peak_selection_mode == "count":
                k = self.peak_point_count if self.peak_point_count is not None else 1
                k = max(1, min(k, total_pts))
            else:  # "percent"
                frac = self.peak_percent if self.peak_percent is not None else 0.05
                if frac <= 0:
                    frac = 0.05
                k = int(np.ceil(frac * total_pts))
                k = max(1, min(k, total_pts))

            # Sort indices according to ascending flag
            sort_idx = np.argsort(values)  # ascending
            if ascending:
                sel_idx = sort_idx[:k]      # lowest k
            else:
                sel_idx = sort_idx[-k:]     # highest k

        x_sel = x_roi[sel_idx]
        y_sel = y_roi[sel_idx]
        vals_sel = values[sel_idx]
        n_used = vals_sel.size

        # Weighted or unweighted mean
        if self.use_gaussian_weight:
            dx = x_sel - xc
            dy = y_sel - yc
            r2 = dx**2 + dy**2

            # sigma proportional to ROI radius
            if r > 0:
                sigma = r / 2.0
            else:
                sigma = np.sqrt(np.mean(r2)) if n_used > 0 else 1.0
                if sigma == 0:
                    sigma = 1.0

            w = np.exp(-0.5 * r2 / sigma**2)
            w_sum = w.sum()
            if w_sum > 0:
                w /= w_sum
                mean_z = float(np.sum(w * vals_sel))
            else:
                mean_z = float(vals_sel.mean())
            weighting = "gaussian"
        else:
            mean_z = float(vals_sel.mean())
            weighting = "uniform"

        roi_index = len(self.roi_results) + 1

        roi_info = {
            "index": roi_index,
            "selection_mode": self.selection_mode,  # "peak" or "baseline"
            "polarity": polarity,                   # "high" or "low"
            "weight": weighting,                    # "uniform" or "gaussian"
            "center_x": xc,
            "center_y": yc,
            "radius": r,
            "points_in_roi": int(total_pts),
            "points_used": int(n_used),
            "mean_z": mean_z,
        }

        self.roi_results.append(roi_info)

        # Automatically jump to last page when a new ROI is added
        n_total = len(self.roi_results)
        self.page_start = max(0, n_total - self.page_size)

        self._update_table()
        self.fig.canvas.draw_idle()

        print(
            f"ROI {roi_index} [{self.selection_mode}, {polarity}, {weighting}]: "
            f"center=({xc:.4g}, {yc:.4g}), r={r:.4g}, "
            f"points={total_pts}, used={n_used}, mean={mean_z:.6g}"
        )

    def _update_table(self):
        """Redraw the summary, list, and histogram panels."""
        # Info panel (top right)
        self.ax_info.clear()
        self.ax_info.axis("off")

        n_total = len(self.roi_results)

        if n_total == 0:
            self.ax_info.text(
                0.5,
                0.5,
                "No ROIs yet.\n\nDraw a circle on the left plot to add entries.",
                ha="center",
                va="center",
                fontsize=10,
                transform=self.ax_info.transAxes,
            )
            # Clear list & hist too
            self.ax_list.clear()
            self.ax_list.axis("off")
            self.ax_hist.clear()
            self.ax_hist.text(
                0.5,
                0.5,
                "No data for histogram.",
                ha="center",
                va="center",
                transform=self.ax_hist.transAxes,
            )
            self.ax_hist.set_xticks([])
            self.ax_hist.set_yticks([])
            self.fig.canvas.draw_idle()
            return

        # Stats for info panel
        n_peak = sum(1 for r in self.roi_results if r["selection_mode"] == "peak")
        n_base = n_total - n_peak
        n_high = sum(1 for r in self.roi_results if r["polarity"] == "high")
        n_low = n_total - n_high

        # Paging
        max_start = max(0, n_total - self.page_size)
        # Clamp page_start
        self.page_start = max(0, min(self.page_start, max_start))
        start = self.page_start
        end = min(start + self.page_size, n_total)
        visible = self.roi_results[start:end]

        n_pages = max(1, int(np.ceil(n_total / self.page_size)))
        current_page = (start // self.page_size) + 1

        # Draw text in info panel
        self.ax_info.text(
            0.01,
            0.80,
            f"Total ROIs: {n_total}",
            ha="left",
            va="center",
            fontsize=10,
            transform=self.ax_info.transAxes,
        )
        self.ax_info.text(
            0.01,
            0.55,
            f"Selection: peak={n_peak}, baseline={n_base}",
            ha="left",
            va="center",
            fontsize=9,
            transform=self.ax_info.transAxes,
        )
        self.ax_info.text(
            0.01,
            0.30,
            f"Polarity: high={n_high}, low={n_low}",
            ha="left",
            va="center",
            fontsize=9,
            transform=self.ax_info.transAxes,
        )
        self.ax_info.text(
            0.01,
            0.05,
            f"Showing ROIs {start + 1}–{end} of {n_total} "
            f"(page {current_page}/{n_pages})",
            ha="left",
            va="center",
            fontsize=9,
            transform=self.ax_info.transAxes,
        )

        # List panel (middle right) - monospace text, paged
        self.ax_list.clear()
        self.ax_list.axis("off")

        fontsize = 9
        n_rows = len(visible)
        n_lines = n_rows + 1  # header + rows
        line_step = 1.0 / (n_lines + 2)
        y = 1.0 - line_step

        # Header
        header = (
            "{:>4} {:>6} {:>4} {:>4} "
            "{:>11} {:>11} {:>9} {:>8} {:>9} {:>11}".format(
                "ROI",
                "sel",
                "pol",
                "wgt",
                "x_center",
                "y_center",
                "radius",
                "pts_in",
                "pts_used",
                "mean_z",
            )
        )
        self.ax_list.text(
            0.01,
            y,
            header,
            ha="left",
            va="center",
            fontsize=fontsize,
            family="monospace",
            fontweight="bold",
            transform=self.ax_list.transAxes,
        )

        # Rows
        for roi in visible:
            y -= line_step
            sel = "peak" if roi["selection_mode"] == "peak" else "base"
            pol = roi["polarity"]
            wgt = "gau" if roi["weight"] == "gaussian" else "uni"

            row_str = (
                "{:4d} {:>6} {:>4} {:>4} "
                "{:11.4g} {:11.4g} {:9.4g} {:8d} {:9d} {:11.3g}".format(
                    roi["index"],
                    sel,
                    pol,
                    wgt,
                    roi["center_x"],
                    roi["center_y"],
                    roi["radius"],
                    roi["points_in_roi"],
                    roi["points_used"],
                    roi["mean_z"],
                )
            )
            self.ax_list.text(
                0.01,
                y,
                row_str,
                ha="left",
                va="center",
                fontsize=fontsize,
                family="monospace",
                transform=self.ax_list.transAxes,
            )

        # Histogram panel (bottom right)
        self.ax_hist.clear()
        values = [r["mean_z"] for r in self.roi_results]
        self.ax_hist.hist(values, bins="auto", edgecolor="black")
        self.ax_hist.set_xlabel("mean_z")
        self.ax_hist.set_ylabel("count")
        self.ax_hist.set_title("Histogram of mean_z", fontsize=9)

        self.fig.canvas.draw_idle()

    # ----------------------------
    # Button callbacks
    # ----------------------------
    def on_undo(self, event):
        """Remove last ROI and its circle."""
        if not self.roi_results:
            print("No ROI to undo.")
            return

        self.roi_results.pop()
        last_circle = self.circle_patches.pop()
        last_circle.remove()

        # Re-index ROIs to keep numbering clean
        for i, roi in enumerate(self.roi_results, start=1):
            roi["index"] = i

        # After undo, keep showing last page
        n_total = len(self.roi_results)
        self.page_start = max(0, n_total - self.page_size)

        self._update_table()
        self.fig.canvas.draw_idle()
        print("Last ROI removed.")

    def on_toggle_mode(self, event):
        """Toggle between peak mode and baseline mode."""
        if self.selection_mode == "peak":
            self.selection_mode = "baseline"
            self.btn_mode.label.set_text("Mode: Baseline")
            print("Switched to BASELINE mode (use all points in ROI).")
        else:
            self.selection_mode = "peak"
            self.btn_mode.label.set_text("Mode: Peak")
            print(
                "Switched to PEAK mode (use extreme points "
                "(count or %) based on ROI polarity)."
            )
        self.fig.canvas.draw_idle()

    def on_toggle_weight(self, event):
        """Toggle between uniform and Gaussian weighting."""
        self.use_gaussian_weight = not self.use_gaussian_weight
        if self.use_gaussian_weight:
            self.btn_weight.label.set_text("Weight: gaussian")
            print(
                "Switched to GAUSSIAN weighting (2D Gaussian centered on ROI center)."
            )
        else:
            self.btn_weight.label.set_text("Weight: uniform")
            print("Switched to UNIFORM weighting (simple arithmetic mean).")
        self.fig.canvas.draw_idle()

    def on_prev_page(self, event):
        """Scroll the ROI list to the previous page."""
        if not self.roi_results:
            return
        self.page_start = max(0, self.page_start - self.page_size)
        self._update_table()

    def on_next_page(self, event):
        """Scroll the ROI list to the next page."""
        if not self.roi_results:
            return
        n_total = len(self.roi_results)
        max_start = max(0, n_total - self.page_size)
        self.page_start = min(max_start, self.page_start + self.page_size)
        self._update_table()

    def on_export(self, event):
        """Export ROI table to CSV."""
        if not self.roi_results:
            print("No ROIs to export.")
            return

        # Tkinter dialog for filename
        root = tk.Tk()
        root.withdraw()
        default_name = "roi_summary.csv"
        save_path = filedialog.asksaveasfilename(
            title="Save ROI CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()

        if not save_path:
            print("Export cancelled.")
            return

        fieldnames = [
            "ROI_index",
            "selection_mode",
            "polarity",
            "weight",
            "center_x",
            "center_y",
            "radius",
            "points_in_ROI",
            "points_used",
            "mean_z",
        ]

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for roi in self.roi_results:
                writer.writerow(
                    [
                        roi["index"],
                        roi["selection_mode"],
                        roi["polarity"],
                        roi["weight"],
                        roi["center_x"],
                        roi["center_y"],
                        roi["radius"],
                        roi["points_in_roi"],
                        roi["points_used"],
                        roi["mean_z"],
                    ]
                )

        print(f"Exported {len(self.roi_results)} ROIs to {save_path}")

    # ----------------------------
    # Public entry point
    # ----------------------------
    def run(self):
        plt.show()


# ----------------------------
# CLI helpers
# ----------------------------

def ask_peak_selection_config(
    default_mode: str = "count",
    default_n: int = 10,
    default_percent: float = 5.0,
) -> tuple[str, int | None, float | None]:
    """
    Ask the user how to define 'top' points in PEAK mode.

    Returns
    -------
    peak_selection_mode : "count" or "percent"
    peak_point_count    : int or None
    peak_percent        : float or None   # fraction, e.g. 0.05 for 5%
    """
    print("Peak selection metric:")
    print("  [1] Number of points")
    print("  [2] Percentage of points in each ROI")
    choice = input("Enter 1 or 2 (default 1): ").strip()

    if choice == "2":
        # Percentage mode
        try:
            s = input(
                f"Percentage of points in each ROI to use in PEAK mode "
                f"(0–100, default {default_percent}): "
            ).strip()
            if not s:
                pct = default_percent
            else:
                pct = float(s)
            if pct <= 0 or pct > 100:
                print(f"Invalid percentage, using default {default_percent}%.")
                pct = default_percent
        except Exception:
            print(f"Invalid input, using default {default_percent}%.")
            pct = default_percent

        return "percent", None, pct / 100.0

    # Default: count mode
    try:
        s = input(
            f"Number of top intensity points to average in PEAK mode "
            f"(default {default_n}): "
        ).strip()
        if not s:
            n = default_n
        else:
            n = int(s)
        if n <= 0:
            print(f"Non-positive number, using default {default_n}.")
            n = default_n
    except Exception:
        print(f"Invalid input, using default {default_n}.")
        n = default_n

    return "count", n, None


def choose_file_via_dialog() -> str:
    """Open a file dialog to choose the XYZ file."""
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select XYZ file (x y z columns)",
        filetypes=[("Text files", "*.txt *.dat *.xyz *.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not filename:
        raise SystemExit("No file selected.")
    return filename


def main():
    # Peak selection configuration
    peak_selection_mode, peak_point_count, peak_percent = ask_peak_selection_config()

    # File: either from command line or dialog
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if not os.path.isfile(filename):
            raise SystemExit(f"File not found: {filename}")
    else:
        filename = choose_file_via_dialog()

    print(f"Using file: {filename}")
    if peak_selection_mode == "count":
        print(
            f"PEAK mode: using top {peak_point_count} points in each ROI "
            f"(after polarity-based ordering)."
        )
    else:
        print(
            f"PEAK mode: using top {peak_percent * 100:.2f}% of points in each ROI "
            f"(after polarity-based ordering)."
        )
    print(
        "You can toggle between PEAK and BASELINE modes and between "
        "UNIFORM and GAUSSIAN weighting in the GUI."
    )

    app = XYZCircleROIApp(
        filename=filename,
        peak_selection_mode=peak_selection_mode,
        peak_point_count=peak_point_count,
        peak_percent=peak_percent,
    )
    app.run()


if __name__ == "__main__":
    main()
