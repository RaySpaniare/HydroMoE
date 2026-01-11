import logging
import os
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def has_parquet_support() -> bool:
    """Return True if optional parquet dependencies are available."""
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def read_table_auto(path: str, usecols=None, parse_dates=None) -> pd.DataFrame:
    """Read CSV/Parquet/Feather with format auto-detection and light optimisation."""
    try:
        if 'ARROW_NUM_THREADS' not in os.environ:
            os.environ['ARROW_NUM_THREADS'] = str(max(1, os.cpu_count() or 1))
    except Exception:
        pass

    lower = path.lower()
    try:
        if lower.endswith(('.parquet', '.pq')):
            return pd.read_parquet(path, columns=usecols)
        if lower.endswith('.feather'):
            return pd.read_feather(path, columns=usecols)
    except Exception as exc:
        logger.warning("按列式格式读取失败，回退CSV。原因: %s", exc)

    try:
        sample_df = pd.read_csv(path, nrows=1000, usecols=usecols)
        dtype_map = {}
        for col in sample_df.columns:
            if col == 'station_id':
                dtype_map[col] = 'category'
            elif col == 'date':
                continue
            elif pd.api.types.is_numeric_dtype(sample_df[col]):
                dtype_map[col] = 'float32'
        return pd.read_csv(
            path,
            usecols=usecols,
            parse_dates=parse_dates,
            dtype=dtype_map,
            engine='c',
            memory_map=True,
            low_memory=False,
            na_filter=True,
            float_precision='high'
        )
    except Exception as exc:
        logger.warning("优化CSV读取失败，使用基础方法。原因: %s", exc)
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, engine='c')


def build_cache_path(src_path: str) -> Optional[Path]:
    """Return a deterministic cache file path adjacent to the source file."""
    try:
        src = Path(src_path)
        if not src.exists():
            return None
        stat = src.stat()
        sig_raw = f"{str(src.resolve())}::{stat.st_size}::{int(stat.st_mtime)}"
        digest = hashlib.md5(sig_raw.encode('utf-8')).hexdigest()
        cache_dir = src.parent / '.hydropy_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = '.parquet' if has_parquet_support() else '.feather'
        return cache_dir / f"{src.stem}_{digest}{suffix}"
    except Exception:
        return None


def read_from_cache(cache_path: Path, columns=None) -> Optional[pd.DataFrame]:
    """Read dataframe from cache path if possible."""
    try:
        if cache_path.suffix.lower() == '.parquet':
            import pyarrow.parquet as pq
            table = pq.read_table(cache_path, columns=columns)
            return table.to_pandas()
        if cache_path.suffix.lower() == '.feather':
            return pd.read_feather(cache_path, columns=columns)
    except Exception as exc:
        logger.warning("⚠️ 读取缓存失败，将回退原始文件: %s", exc)
    return None


def write_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Persist dataframe to cache. Failures are logged but ignored."""
    try:
        if cache_path.suffix.lower() == '.parquet':
            df.to_parquet(cache_path, index=False)
        elif cache_path.suffix.lower() == '.feather':
            df.reset_index(drop=True).to_feather(cache_path)
        logger.info("🧭 已创建加速缓存: %s", cache_path)
    except Exception as exc:
        logger.warning("⚠️ 写入缓存失败: %s", exc)
