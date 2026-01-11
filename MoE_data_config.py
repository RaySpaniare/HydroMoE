from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FixedDataConfig:
    """数据加载配置，原先定义在 MoE_data_loader 中。"""

    csv_path: str = r"D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理\特征合并长表.csv"
    sequence_length: int = 96
    sequence_stride: int = 16
    feature_cols: Optional[List[str]] = None
    target_col: str = "runoff"

    train_start: str = '1980-01-01'
    train_end: str = '1999-12-31'
    val_start: str = '2000-01-01'
    val_end: str = '2007-12-31'
    test_start: str = '2008-01-01'
    test_end: str = '2014-09-30'

    use_date_split: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    normalize_features: bool = True
    normalize_targets: bool = True

    use_all_stations: bool = True
    quick_test: bool = False
    quick_test_stations: int = 10

    station_batch_size: int = 100
    use_sequence_cache: bool = True
    parallel_sequence_creation: bool = False
    max_sequence_workers: int = 4

    filter_station_ids: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.feature_cols is None:
            self.feature_cols = ["pet", "precip", "temp"]
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6
