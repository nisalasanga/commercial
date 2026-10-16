from __future__ import annotations

import pandas as pd

from src.master_table.one import create_run_date_master_table, load_config


def test_load_config_reads_train_flag(tmp_path):
    config_path = tmp_path / "conf.yaml"
    config_path.write_text("train_mt: False\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["train_mt"] is False


def test_create_run_date_master_table_filters_to_run_date():
    rows = []
    for day in range(1, 92):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day - 1)
        rows.append(
            {
                "seller_id": "S1",
                "date": date,
                "daily_gmv": 100.0 + day,
                "refund_amount": 5.0,
                "platform_fee": 15.0,
                "promo_flag": 0,
                "is_default": 0,
            }
        )

    raw = pd.DataFrame(rows)
    run_table = create_run_date_master_table(raw, run_date="2024-03-31")

    assert not run_table.empty
    assert set(run_table["date"]) == {pd.Timestamp("2024-03-31")}
