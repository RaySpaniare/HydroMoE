"""Recover the four executed HydroPy/CMA-ES process fluxes without changing evidence.

The simulation loop is the loop in hydrocsv(PBM)/optimization.py::_run_hydro_simulation.
It is run from 1980 so the test-period snow, soil, and groundwater states have the
same spin-up/history as the archived standalone PBM runoff.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PBM_DIR = ROOT / "hydrocsv(PBM)"
OUT_ROOT = ROOT / "audit_results" / "four_module_physical_outputs"
sys.path.insert(0, str(PBM_DIR))
os.chdir(PBM_DIR)

from data_processing import (  # noqa: E402
    find_common_stations,
    load_all_csv_data,
    prepare_model_data,
    validate_time_consistency,
)
from hydro_core import HydroPyCSVCore  # noqa: E402


def simulate_four(precip, temp, pet, static_params, lat):
    n_days = len(precip)
    model = HydroPyCSVCore(static_params=static_params)
    wcap = static_params["wcap"]
    beta = static_params["beta"]
    wmin = static_params["wmin"]
    wmax = static_params["wmax"]
    lai = static_params["lai_annual"]
    fveg = static_params.get("fveg", 0.2)
    fbare = static_params.get("fbare", 0.7)
    pet_correction = static_params.get("pet_correction", 1.0)

    swe = 0.0
    wliq = 0.0
    rootmoist = wcap * 0.5
    groundwstor = model.get_initial_groundwater()
    snow = np.zeros(n_days, np.float32)
    quick = np.zeros(n_days, np.float32)
    et = np.zeros(n_days, np.float32)
    slow = np.zeros(n_days, np.float32)
    total = np.zeros(n_days, np.float32)
    swe_state = np.zeros(n_days, np.float32)
    soil_state = np.zeros(n_days, np.float32)
    groundwater_state = np.zeros(n_days, np.float32)
    balance_residual = np.zeros(n_days, np.float32)

    previous_storage = swe + wliq + rootmoist + groundwstor
    for day in range(n_days):
        daily_precip = precip[day]
        daily_temp = temp[day]
        daily_pet = pet[day] * pet_correction
        day_of_year = day % 365 + 1

        snowf, rainf, _ = model.get_rain_and_snow(daily_precip, daily_temp)
        smelt_pot, _ = model.get_potential_snowmelt(daily_temp, lat, day_of_year)
        swe, wliq, _, rainmelt = model.update_snow(swe, wliq, snowf, smelt_pot)
        frozen = model.diagnose_frozen_ground(daily_temp)
        throughfall = rainmelt + rainf
        qs = model.get_surface_runoff(throughfall, rootmoist, wcap, beta, wmin, wmax, frozen)
        qsb = model.get_drainage(rootmoist, wcap, dt=86400, frozen=frozen)
        transp = model.get_transpiration(daily_pet, rootmoist, wcap, fveg, lai, static_params)
        sevap = model.get_soilevap(daily_pet, rootmoist, wcap, fbare, static_params)
        rootmoist, qs_add = model.update_soil(rootmoist, throughfall, qs, transp, sevap, qsb, wcap)
        qs += qs_add
        groundwstor, qg = model.update_groundwater(groundwstor, qsb, static_params=static_params)

        # These definitions preserve the executed total-runoff identity exactly:
        # total runoff = surface/overflow quickflow + soil drainage + baseflow.
        snow[day] = rainmelt
        quick[day] = qs
        et[day] = transp + sevap
        slow[day] = qsb + qg
        total[day] = qs + qsb + qg
        swe_state[day] = swe + wliq
        soil_state[day] = rootmoist
        groundwater_state[day] = groundwstor
        storage = swe + wliq + rootmoist + groundwstor
        balance_residual[day] = daily_precip - (transp + sevap) - total[day] - (storage - previous_storage)
        previous_storage = storage

    return snow, quick, et, slow, total, swe_state, soil_state, groundwater_state, balance_residual


def main(params_file: Path, label: str) -> None:
    files = load_all_csv_data()
    if not files or not validate_time_consistency(files):
        raise RuntimeError("Forcing files missing or time-inconsistent")
    common = find_common_stations(files)
    forcing, _ = prepare_model_data(files, common)
    dates = pd.to_datetime(forcing["dates"])
    test_mask = np.asarray(dates >= pd.Timestamp("2008-01-01"))
    test_dates = dates[test_mask]
    stations = forcing["stations"]
    station_index = {stations.iloc[i]["station_name"]: i for i in range(len(stations))}
    params_doc = json.loads(params_file.read_text(encoding="utf-8"))
    params = params_doc.get("station_results", params_doc)

    frames = []
    for number, station in enumerate(sorted(params), 1):
        j = station_index[station]
        p = dict(params[station]["best_params"])
        lat = float(stations.iloc[j]["latitude"])
        p["lat"] = lat
        outputs = simulate_four(
            forcing["precip"][:, j], forcing["temp"][:, j], forcing["pet"][:, j], p, lat
        )
        snow, quick, et, slow, total, swe, soil, groundwater, residual = [x[test_mask] for x in outputs]
        frames.append(pd.DataFrame({
            "station_id": station,
            "date": test_dates,
            "snow_output": snow,
            "runoff_output": quick,
            "et_output": et,
            "groundwater_output": slow,
            "pbm_total_runoff": total,
            "snow_storage": swe,
            "soil_storage": soil,
            "groundwater_storage": groundwater,
            "water_balance_residual": residual,
        }))
        if number % 50 == 0:
            print(f"{number}/550 stations", flush=True)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["station_id", "date"]).reset_index(drop=True)
    archived = pq.read_table(ROOT / "hydropy1.0MoE" / "outputs" / "cache" / "pbm_test_long.parquet").to_pandas()
    archived["date"] = pd.to_datetime(archived["date"])
    joined = result.merge(
        archived[["station_id", "date", "pbm_runoff"]],
        on=["station_id", "date"], how="left", validate="one_to_one",
    )
    delta = joined["pbm_total_runoff"].to_numpy(np.float64) - joined["pbm_runoff"].to_numpy(np.float64)
    identity = result["pbm_total_runoff"].to_numpy(np.float64) - (
        result["runoff_output"].to_numpy(np.float64)
        + result["groundwater_output"].to_numpy(np.float64)
    )

    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(result, preserve_index=False),
        out_dir / "pbm_four_module_test_long.parquet",
        compression="zstd",
    )
    summary = {
        "stations": int(result.station_id.nunique()),
        "dates": int(result.date.nunique()),
        "rows": int(len(result)),
        "params_file": str(params_file),
        "max_abs_total_runoff_error_vs_archived": float(np.nanmax(np.abs(delta))),
        "median_abs_total_runoff_error_vs_archived": float(np.nanmedian(np.abs(delta))),
        "max_abs_quick_plus_slow_identity_error": float(np.max(np.abs(identity))),
        "module_statistics": {
            c: {
                "mean": float(result[c].mean()),
                "median": float(result[c].median()),
                "max": float(result[c].max()),
                "nonzero_fraction": float((result[c] != 0).mean()),
            }
            for c in ["snow_output", "runoff_output", "et_output", "groundwater_output"]
        },
        "water_balance_residual": {
            "mean": float(result.water_balance_residual.mean()),
            "median": float(result.water_balance_residual.median()),
            "mean_absolute": float(result.water_balance_residual.abs().mean()),
            "max_absolute": float(result.water_balance_residual.abs().max()),
        },
        "definitions": {
            "snow_output": "executed rainmelt/overflow released from the stateful snowpack",
            "runoff_output": "executed surface runoff plus soil-capacity overflow (quickflow)",
            "et_output": "executed transpiration plus soil evaporation",
            "groundwater_output": "executed soil drainage qsb plus groundwater baseflow qg (slowflow)",
            "pbm_total_runoff": "runoff_output + groundwater_output; exactly the archived standalone PBM target",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=PBM_DIR / "cmaes_optimal_params.json")
    parser.add_argument("--label", default="primary_benchmark")
    args = parser.parse_args()
    main(args.params.resolve(), args.label)
