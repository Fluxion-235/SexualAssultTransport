"""
Main entry point for Transit Harassment Risk Modeling and Visualization.
Run this file directly to generate all high-resolution figures and view tabular summaries.
"""

from visualize_transit_harassment import (
    plot_transit_harassment_heatmap,
    plot_mathematical_models_dashboard,
    plot_complete_dashboard,
    plot_western_province_capacity_deficits,
    generate_commuter_profiles,
    get_global_prevalence_data,
    get_route_risk_data
)

def main():
    print("=" * 70)
    print(" TRANSIT SEXUAL HARASSMENT RISK & ECONOMETRIC VISUALIZATION SUITE")
    print("=" * 70)

    print("\n[1/4] Generating Global Prevalence & Corridor Heatmap...")
    plot_transit_harassment_heatmap("transit_harassment_heatmap.png")

    print("\n[2/4] Generating 4-Panel Mathematical Models Dashboard...")
    plot_mathematical_models_dashboard("transit_mathematical_models.png")

    print("\n[3/4] Generating Executive Master Dashboard...")
    plot_complete_dashboard("transit_harassment_complete_dashboard.png")

    print("\n[4/4] Generating Western Province Systemic Capacity & Deficit Analysis...")
    plot_western_province_capacity_deficits("western_province_capacity_analysis.png")


    print("\n" + "=" * 70)
    print("SUMMARY OF EMPIRICAL MODEL ESTIMATES")
    print("=" * 70)
    
    print("\n• Binary Logit Predicted Vulnerability by Commuter Profile:")
    df_prof = generate_commuter_profiles()
    for _, row in df_prof.iterrows():
        print(f"  - {row['Profile']:<42} : P = {row['Probability']*100:5.1f}% (U = {row['Utility_Uh']:+.2f})")

    print("\n• Global Prevalence vs Reporting Gap:")
    df_glob = get_global_prevalence_data()
    for _, row in df_glob.iterrows():
        print(f"  - {row['Country']:<14} : Lifetime Prev: {row['Prevalence']:4.1f}% | Reported: {row['Reporting']:4.1f}% | Silence: {row['Silence']:4.1f}%")

    print("\n• Route Risk Determinant Matrix (Overall Score out of 10):")
    df_routes = get_route_risk_data()
    for _, row in df_routes.iterrows():
        print(f"  - {row['Route']:<36} [{row['Mode']:<18}] : Score = {row['OverallRisk']:3.1f} ({row['Classification']})")

    print("\n[✓] All figures generated successfully in the project directory.")
    print("\n[Tip] To run real-time telemetry synced with live Colombo weather & clock:")
    print("      • Terminal: .venv/bin/python live_telemetry.py")
    print("      • Browser : Open index.html (or run 'python -m http.server 8000')")

if __name__ == "__main__":
    main()
