# plot_runs.gp -- visualize the logged evaluation runs (logs/runs.csv).
#
# Usage (from the repo root):
#   python logs/make_plot_data.py   # regenerate .dat files after new runs
#   gnuplot logs/plot_runs.gp       # writes logs/runs_plot.png
#
# log_run.py refreshes the .dat files automatically, so normally you only
# need the gnuplot line. Panels:
#   1. Stacked recovery breakdown (exact / near-miss / off-target) per config
#   2. Mean best-candidate score per config
#   3. Constraint-tightness sweep: recovery vs how loose the box bounds are
#   4. Config landscape: exacts vs mean score (tag-labelled points)

# Give the 2x2 layout explicit outer margins and row/column breathing room.
# The default layout puts panel titles and lower x-labels right on the
# canvas/panel boundaries, which lets pngcairo clip the tops of glyphs.
set terminal pngcairo size 1400,1100 font "Arial,11"
set output "logs/runs_plot.png"

set datafile separator "\t"
set style fill solid 0.85 border -1
# Keep keys local to the panels that need them. A shared `above` key is
# rendered in the outer multiplot margin and overlaps the figure/panel titles.
unset key
set grid ytics lc rgb "#dddddd"
set border 3

set multiplot layout 2,2 title "Unredact — logged evaluation runs (21 boxes)" font ",14" \
    margins 0.08, 0.97, 0.12, 0.88 spacing 0.10, 0.17

# --- Panel 1: stacked recovery breakdown per config -----------------------
set title "Recovery breakdown per configuration" font ",12"
set key inside top left opaque font ",9"
set style data histograms
set style histogram rowstacked
set boxwidth 0.75 relative
set ylabel "boxes"
set xlabel ""
set yrange [0:*]
unset x2tics; unset y2tics; unset y2label
plot 'logs/runs_plot.dat' using 3:xtic(1) lc rgb "#2ca02c" title "exact", \
     '' using 4 lc rgb "#ffbb33" title "near-miss", \
     '' using 5 lc rgb "#d62728" title "off-target"

# --- Panel 2: mean best-candidate score per config ------------------------
set title "Mean best-candidate score (prior05 = blended final metric)" font ",12"
set key inside top left opaque font ",9"
set style data boxes
unset ylabel
# Leave headroom for the numeric labels above the bars.
set offsets 0, 0, graph 0.16, 0
set yrange [0:*]
plot 'logs/runs_plot.dat' using 6:xtic(1) lc rgb "#1f77b4" title "mean score", \
     '' using 6:($6 + 0.025):(sprintf("%.2f", $6)) with labels font ",8" notitle

# --- Panel 3: constraint-tightness sweep ----------------------------------
set title "Tightness sweep: recovery vs loosened box bounds" font ",12"
set key inside top left opaque font ",9"
set style data linespoints
set xlabel "chars added to each side of the box char range"
set ylabel "exact recoveries" tc rgb "#2ca02c"
set y2label "mean final score" tc rgb "#1f77b4"
set yrange [0:*]
set y2range [0:1.5]
set y2tics
set xrange [*:*]
# Restore panel-specific offsets after panel 2's bar-label headroom.
set offsets 0, 0, 0, 0
plot 'logs/sweep_plot.dat' using 1:2 lc rgb "#2ca02c" lw 2 pt 7 ps 1.5 title "exact", \
     '' using 1:5 axes x1y2 lc rgb "#1f77b4" lw 2 pt 9 ps 1.5 title "mean final score"

# --- Panel 4: config landscape --------------------------------------------
set title "Config landscape: exacts vs mean score" font ",12"
unset key
set style data points
unset xlabel; unset ylabel
unset y2tics; unset y2label
# Use explicit padded ranges: the point annotations and the outer x ticks
# otherwise sit on the panel border and get clipped by pngcairo.
set xrange [0.38:0.61]
set xtics 0.05
set yrange [2.5:5.8]
set offsets 0, 0, 0, 0
plot 'logs/runs_plot.dat' using 6:3 pt 7 ps 1.8 lc rgb "#d62728" notitle, \
     '' using 6:3:1 with labels offset char 0.35,0.45 font ",8" notitle

unset multiplot
