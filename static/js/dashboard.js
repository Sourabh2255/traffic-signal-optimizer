/**
 * Data Science Dashboard charts (Phase 13), built from the current
 * cycle's analysis data using Plotly.
 */

if (typeof ANALYSIS !== "undefined" && ANALYSIS && ANALYSIS.results && typeof Plotly !== "undefined") {
    const roads = Object.keys(ANALYSIS.results);
    const plotBg = "#1e293b";
    const paperBg = "#1e293b";
    const fontColor = "#e2e8f0";

    const baseLayout = {
        paper_bgcolor: paperBg,
        plot_bgcolor: plotBg,
        font: { color: fontColor },
        margin: { t: 40, l: 50, r: 20, b: 40 },
    };

// --- Vehicle count by road ---
Plotly.newPlot("chart-vehicle-by-road", [{
    x: roads,
    y: roads.map(r => ANALYSIS.results[r].counts.total),
    type: "bar",
    marker: { color: "#4ade80" },
}], { ...baseLayout, title: "Vehicle Count by Road" });

// --- Vehicle type distribution (aggregated across all roads) ---
const typeTotals = { car: 0, motorcycle: 0, bus: 0, truck: 0 };
roads.forEach(r => {
    const c = ANALYSIS.results[r].counts;
    typeTotals.car += c.car;
    typeTotals.motorcycle += c.motorcycle;
    typeTotals.bus += c.bus;
    typeTotals.truck += c.truck;
});
Plotly.newPlot("chart-type-distribution", [{
    labels: Object.keys(typeTotals),
    values: Object.values(typeTotals),
    type: "pie",
    marker: { colors: ["#4ade80", "#60a5fa", "#facc15", "#f87171"] },
}], { ...baseLayout, title: "Vehicle Type Distribution (All Roads)" });

// --- Predicted green time by road ---
Plotly.newPlot("chart-green-time", [{
    x: roads,
    y: roads.map(r => ANALYSIS.results[r].prediction.final_green_time),
    type: "bar",
    marker: { color: "#60a5fa" },
}], { ...baseLayout, title: "Predicted Green Time by Road (sec)" });

// --- Density % vs Demand score by road ---
Plotly.newPlot("chart-density-demand", [
    {
        x: roads,
        y: roads.map(r => ANALYSIS.results[r].density.percentage),
        name: "Density %",
        type: "bar",
        marker: { color: "#facc15" },
    },
    {
        x: roads,
        y: roads.map(r => ANALYSIS.results[r].demand_score),
        name: "Demand Score",
        type: "bar",
        marker: { color: "#f87171" },
    },
], { ...baseLayout, title: "Density % vs Demand Score by Road", barmode: "group" });
}
