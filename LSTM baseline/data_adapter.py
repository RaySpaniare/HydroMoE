from __future__ import annotations

import argparse
import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import xarray as xr

from project_config import (
    DEFAULT_LONG_TABLE_PARQUET,
    DYNAMIC_INPUTS,
    STATIC_ATTRIBUTES,
    TARGET_VARIABLES,
    SplitConfig,
    get_paths,
)


@dataclass
class AdapterReport:
    input_parquet: str
    output_data_dir: str
    n_stations: int
    start_date: str
    end_date: str
    dynamic_inputs: List[str]
    target_variables: List[str]
    static_attributes: List[str]
    train_basin_file: str
    validation_basin_file: str
    test_basin_file: str


def _require_duckdb():
    try:
        import duckdb

        return duckdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "未安装 duckdb。请在 GIS3.9 环境执行: pip install duckdb"
        ) from exc


def _validate_columns(columns: Sequence[str], required: Sequence[str]) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError("输入 parquet 缺少关键列: {0}".format(missing))


def _load_station_ids(parquet_path: Path) -> List[str]:
    duckdb = _require_duckdb()
    query = """
    SELECT DISTINCT CAST(station_id AS VARCHAR) AS station_id
    FROM read_parquet(?)
    ORDER BY station_id
    """
    with duckdb.connect(database=":memory:") as con:
        rows = con.execute(query, [str(parquet_path)]).fetchall()
    return [str(r[0]) for r in rows]


def _load_schema_columns(parquet_path: Path) -> List[str]:
    duckdb = _require_duckdb()
    with duckdb.connect(database=":memory:") as con:
        relation = con.execute("SELECT * FROM read_parquet(?) LIMIT 1", [str(parquet_path)])
        return [c[0] for c in relation.description]


def _load_station_frame(parquet_path: Path, station_id: str) -> pd.DataFrame:
    duckdb = _require_duckdb()
    query = """
    SELECT
      CAST(station_id AS VARCHAR) AS station_id,
      CAST(date AS DATE) AS date,
      CAST(lon AS DOUBLE) AS lon,
      CAST(lat AS DOUBLE) AS lat,
      CAST(pet AS DOUBLE) AS pet,
      CAST(precip AS DOUBLE) AS precip,
      CAST(temp AS DOUBLE) AS temp,
      CAST(runoff AS DOUBLE) AS runoff
    FROM read_parquet(?)
    WHERE CAST(station_id AS VARCHAR) = ?
    ORDER BY date
    """
    with duckdb.connect(database=":memory:") as con:
        return con.execute(query, [str(parquet_path), station_id]).df()


def _load_full_frame(parquet_path: Path) -> pd.DataFrame:
    duckdb = _require_duckdb()
    query = """
    SELECT
        CAST(station_id AS VARCHAR) AS station_id,
        CAST(date AS DATE) AS date,
        CAST(lon AS DOUBLE) AS lon,
        CAST(lat AS DOUBLE) AS lat,
        CAST(pet AS DOUBLE) AS pet,
        CAST(precip AS DOUBLE) AS precip,
        CAST(temp AS DOUBLE) AS temp,
        CAST(runoff AS DOUBLE) AS runoff
    FROM read_parquet(?)
    ORDER BY station_id, date
    """
    with duckdb.connect(database=":memory:") as con:
        return con.execute(query, [str(parquet_path)]).df()


def _write_basin_list(path: Path, basin_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(basin_ids) + "\n", encoding="utf-8")


def _build_daily_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def _write_dataset_with_retry(ds: xr.Dataset, output_nc: Path, max_retries: int = 3) -> None:
    """Write NetCDF via temp file + atomic replace, retrying transient lock errors on Windows."""
    for attempt in range(1, max_retries + 1):
        uid = str(time.time_ns())[-8:]
        temp_nc = output_nc.with_suffix(f".tmp.{uid}.nc")
        trash_nc = output_nc.with_suffix(f".trash.{uid}.nc")
        
        try:
            # 使用 scipy engine 能极大程度绕过 netCDF4(C底层HDF5) 在Windows平台上的奇怪句柄锁和路径错误
            ds.to_netcdf(temp_nc, mode="w", engine="scipy")
            ds.close()

            if output_nc.exists():
                try:
                    os.chmod(output_nc, stat.S_IWRITE)
                except OSError:
                    pass
                
                try:
                    output_nc.unlink()
                except PermissionError:
                    # Windows trick: sometimes you can rename a locked file when you can't delete it
                    try:
                        output_nc.rename(trash_nc)
                    except OSError:
                        pass

            temp_nc.replace(output_nc)
            return
        except Exception as exc:
            import gc
            gc.collect()

            try:
                if temp_nc.exists():
                    temp_nc.unlink()
            except OSError:
                pass

            if attempt >= max_retries:
                if isinstance(exc, PermissionError):
                    raise PermissionError(
                        f"写入失败：{output_nc}。该文件可能被其他进程长期占用，系统无法释放。建议关闭 Python 并重启重试。"
                    ) from exc
                raise
            
            time.sleep(0.6 * attempt)


def convert_long_table_to_generic_dataset(
    parquet_path: Path,
    output_data_dir: Path,
    basin_list_root: Path,
    split_cfg: Optional[SplitConfig] = None,
    overwrite: bool = False,
) -> AdapterReport:
    """Convert long-table parquet into NeuralHydrology GenericDataset layout.

    Output structure:
    - output_data_dir/time_series/<station_id>.nc
    - output_data_dir/attributes/station_attributes.csv
    - basin_list_root/train_basins.txt
    - basin_list_root/validation_basins.txt
    - basin_list_root/test_basins.txt
    """
    if split_cfg is None:
        split_cfg = SplitConfig()

    if not parquet_path.exists():
        raise FileNotFoundError("输入 parquet 不存在: {0}".format(parquet_path))

    output_ts_dir = output_data_dir / "time_series"
    output_attr_dir = output_data_dir / "attributes"

    if output_data_dir.exists() and any(output_data_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "输出目录已存在内容。若需覆盖请加 --overwrite: {0}".format(output_data_dir)
            )
        try:
            # 覆盖模式下先清空旧目录的内容，尽量避免直接 rmtree 整个文件夹导致 Windows 幽灵目录或权限锁
            for item in output_data_dir.rglob("*"):
                if item.is_file():
                    try:
                        item.unlink()
                    except PermissionError:
                        try:
                            # 尝试魔法换名再删
                            trash = item.with_name(item.name + f".{time.time_ns()}trash")
                            item.rename(trash)
                            trash.unlink()
                        except OSError:
                            pass
            time.sleep(0.5)  # 稍微给 Windows NTFS 缓冲时间
        except Exception as exc:
            raise PermissionError(
                f"清空输出目录失败，可能有 nc 文件被其他进程占用（例如 Python、可视化软件或资源管理器预览）。"
                f"请先关闭占用后重试，目录: {output_data_dir}"
            ) from exc

    output_ts_dir.mkdir(parents=True, exist_ok=True)
    output_attr_dir.mkdir(parents=True, exist_ok=True)

    required_columns = ["station_id", "date", "lon", "lat"] + DYNAMIC_INPUTS + TARGET_VARIABLES
    schema_columns = _load_schema_columns(parquet_path)
    _validate_columns(schema_columns, required_columns)

    print("正在一次性加载 Parquet 全表到内存，以减少重复扫描...")
    full_df = _load_full_frame(parquet_path)
    if full_df.empty:
        raise ValueError("输入 parquet 中没有可用数据")

    full_df["date"] = pd.to_datetime(full_df["date"])
    basin_ids = full_df["station_id"].dropna().astype(str).unique().tolist()
    if not basin_ids:
        raise ValueError("未在 parquet 中找到任何 station_id")

    attributes_rows: List[Dict[str, object]] = []
    global_start = pd.Timestamp(split_cfg.train_start)
    global_end = pd.Timestamp(split_cfg.test_end)
    full_index = _build_daily_index(global_start, global_end)

    def process_station(basin_id: str, station_df: pd.DataFrame) -> Dict[str, object]:
        basin_id = str(basin_id)
        station_df = station_df.drop_duplicates(subset=["date"]).sort_values("date")
        station_df = station_df.set_index("date")

        # Reindex to daily continuity so NH sequence slicing stays deterministic.
        station_df = station_df.reindex(full_index)
        station_df.index.name = "date"

        # Persist station-level lon/lat as attributes for traceability.
        lon = float(station_df["lon"].dropna().iloc[0]) if station_df["lon"].notna().any() else float("nan")
        lat = float(station_df["lat"].dropna().iloc[0]) if station_df["lat"].notna().any() else float("nan")

        variables = DYNAMIC_INPUTS + TARGET_VARIABLES
        ds = xr.Dataset(
            data_vars={
                col: ("date", station_df[col].astype("float32").to_numpy())
                for col in variables
            },
            coords={"date": station_df.index.to_numpy()},
        )

        output_nc = output_ts_dir / "{0}.nc".format(basin_id)
        try:
            _write_dataset_with_retry(ds=ds, output_nc=output_nc, max_retries=3)
        except PermissionError as exc:
            raise PermissionError(
                "写入失败：{0}。该文件可能被其他进程占用，或被系统防护进程短暂锁定。"
                "请关闭占用后重试；若你只想继续训练且数据已准备完，可在 main.py 使用 --resume 跳过适配阶段。"
                .format(output_nc)
            ) from exc
        return {"gauge_id": basin_id, "lon": lon, "lat": lat}

    total_stations = len(basin_ids)
    max_workers = min(32, max(1, (os.cpu_count() or 4) * 2))
    max_workers = min(max_workers, max(1, total_stations))
    print("开始并行生成 NetCDF 文件，总计 {0} 个站点...".format(total_stations))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_station, basin_id, station_df): basin_id
            for basin_id, station_df in full_df.groupby("station_id", sort=False)
            if not station_df.empty
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            basin_id = futures[future]
            try:
                attributes_rows.append(future.result())
            except Exception as exc:
                print("站点 {0} 写入失败: {1}".format(basin_id, exc))
                raise

            if idx % 50 == 0 or idx == total_stations:
                print("已转换 {0}/{1} 个站点...".format(idx, total_stations))

    attr_df = pd.DataFrame(attributes_rows).set_index("gauge_id").sort_index()
    attr_df.to_csv(output_attr_dir / "station_attributes.csv", encoding="utf-8")

    train_basin_file = basin_list_root / "train_basins.txt"
    val_basin_file = basin_list_root / "validation_basins.txt"
    test_basin_file = basin_list_root / "test_basins.txt"

    _write_basin_list(train_basin_file, basin_ids)
    _write_basin_list(val_basin_file, basin_ids)
    _write_basin_list(test_basin_file, basin_ids)

    report = AdapterReport(
        input_parquet=str(parquet_path),
        output_data_dir=str(output_data_dir),
        n_stations=len(basin_ids),
        start_date=str(global_start.date()),
        end_date=str(global_end.date()),
        dynamic_inputs=DYNAMIC_INPUTS,
        target_variables=TARGET_VARIABLES,
        static_attributes=STATIC_ATTRIBUTES,
        train_basin_file=str(train_basin_file),
        validation_basin_file=str(val_basin_file),
        test_basin_file=str(test_basin_file),
    )

    (output_data_dir / "adapter_report.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将长表 parquet 转换为 NH GenericDataset")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=DEFAULT_LONG_TABLE_PARQUET,
        help="长表 parquet 路径",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=get_paths()["prepared_data_root"],
        help="GenericDataset 根目录",
    )
    parser.add_argument(
        "--basin-list-root",
        type=Path,
        default=get_paths()["basin_list_root"],
        help="train/val/test basin 列表输出目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标目录已有内容则覆盖",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_long_table_to_generic_dataset(
        parquet_path=args.input_parquet,
        output_data_dir=args.output_data_dir,
        basin_list_root=args.basin_list_root,
        split_cfg=SplitConfig(),
        overwrite=args.overwrite,
    )
    print("数据适配完成:")
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
