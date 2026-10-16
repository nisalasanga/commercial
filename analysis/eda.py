"""Modular EDA script for ecommerce lending data."""

from pathlib import Path

import pandas as pd

from src.utils import load_data

try:
	import matplotlib.dates as mdates
	import matplotlib.pyplot as plt
except ImportError as exc:
	raise ImportError(
		"matplotlib is required for plotting. Install it with: pip install matplotlib"
	) from exc


def _print_section(title: str) -> None:
	"""Print a section title for easier terminal readability."""
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


def add_business_ratios(data: pd.DataFrame) -> pd.DataFrame:
	"""Create business ratio features used in analysis and plots."""
	result = data.copy()
	safe_gmv = result["daily_gmv"].replace(0, pd.NA)
	result["refund_ratio"] = (result["refund_amount"] / safe_gmv).fillna(0)
	result["fee_ratio"] = (result["platform_fee"] / safe_gmv).fillna(0)
	return result


def descriptive_overview(data: pd.DataFrame) -> None:
	"""Print dataset shape, columns, dtypes, missing values, and duplicates."""
	_print_section("DESCRIPTIVE ANALYSIS: DATA OVERVIEW")
	print(f"Rows: {data.shape[0]:,}")
	print(f"Columns: {data.shape[1]}")
	print("\nColumns:")
	print(data.columns.tolist())
	print("\nDtypes:")
	print(data.dtypes)

	missing_counts = data.isnull().sum()
	missing_pct = (missing_counts / len(data) * 100).round(2)
	missing_summary = pd.DataFrame(
		{
			"null_count": missing_counts,
			"null_percentage": missing_pct,
		}
	)
	print("\nMissing values summary:")
	print(missing_summary)
	print("\nDuplicate rows:", data.duplicated().sum())

	numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
	if numeric_cols:
		numeric_summary = data[numeric_cols].agg(
			["count", "mean", "median", "std", "min", "max"]
		).T
		numeric_summary["q1_25pct"] = data[numeric_cols].quantile(0.25)
		numeric_summary["q3_75pct"] = data[numeric_cols].quantile(0.75)
		print("\nNumeric descriptive summary (mean, median, spread):")
		print(numeric_summary)


def descriptive_statistics(data: pd.DataFrame) -> None:
	"""Print descriptive stats for numeric fields."""
	_print_section("DESCRIPTIVE ANALYSIS: NUMERIC SUMMARY")
	numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
	print(data[numeric_cols].describe().T)


def descriptive_target_distribution(data: pd.DataFrame) -> None:
	"""Print class distribution for default target."""
	_print_section("DESCRIPTIVE ANALYSIS: TARGET DISTRIBUTION")
	target_counts = data["is_default"].value_counts().sort_index()
	target_pct = data["is_default"].value_counts(normalize=True).sort_index() * 100
	target_report = pd.DataFrame(
		{
			"count": target_counts,
			"percentage": target_pct.round(2),
		}
	)
	print(target_report)


def descriptive_seller_checks(data: pd.DataFrame) -> None:
	"""Print seller-level consistency checks."""
	_print_section("DESCRIPTIVE ANALYSIS: SELLER-LEVEL CHECKS")
	sellers = data["seller_id"].nunique()
	min_days = data.groupby("seller_id").size().min()
	max_days = data.groupby("seller_id").size().max()
	mixed_label_sellers = (
		data.groupby("seller_id")["is_default"].nunique().gt(1).sum()
	)
	print(f"Unique sellers: {sellers}")
	print(f"Days per seller (min, max): ({min_days}, {max_days})")
	print(f"Sellers with mixed default labels: {mixed_label_sellers}")


def descriptive_ratio_summary(data: pd.DataFrame) -> None:
	"""Print summary for business ratios."""
	_print_section("DESCRIPTIVE ANALYSIS: BUSINESS RATIOS")
	print(data[["refund_ratio", "fee_ratio"]].describe().T)


def plot_numeric_variable(data: pd.DataFrame, column: str, output_dir: Path) -> None:
	"""Save histogram plot for a numeric variable."""
	plt.figure(figsize=(9, 5))
	data[column].dropna().hist(bins=40)
	plt.title(f"Distribution of {column}")
	plt.xlabel(column)
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(output_dir / f"{column}_hist.png")
	plt.close()


def plot_binary_or_category_variable(
	data: pd.DataFrame,
	column: str,
	output_dir: Path,
) -> None:
	"""Save bar plot for a low-cardinality categorical variable."""
	counts = data[column].value_counts(dropna=False).sort_index()
	plt.figure(figsize=(8, 4))
	counts.plot(kind="bar")
	plt.title(f"Count plot of {column}")
	plt.xlabel(column)
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(output_dir / f"{column}_count.png")
	plt.close()


def plot_date_variable(data: pd.DataFrame, output_dir: Path) -> None:
	"""Plot daily GMV trend across dates."""
	if "date" not in data.columns:
		return

	date_series = pd.to_datetime(data["date"], errors="coerce")
	date_gmv = (
		data.assign(date=date_series)
		.dropna(subset=["date"])
		.groupby("date", as_index=False)["daily_gmv"]
		.sum()
		.sort_values("date")
	)

	plt.figure(figsize=(14, 6))
	plt.plot(date_gmv["date"], date_gmv["daily_gmv"], linewidth=1.2)
	plt.title("Daily Total GMV Over Time")
	plt.xlabel("Date")
	plt.ylabel("Total daily_gmv")
	ax = plt.gca()
	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	plt.savefig(output_dir / "date_daily_gmv_trend.png")
	plt.close()


def plot_high_cardinality_seller(data: pd.DataFrame, output_dir: Path) -> None:
	"""Plot top sellers by row count for high-cardinality seller_id variable."""
	seller_counts = data["seller_id"].value_counts().head(20)
	plt.figure(figsize=(12, 6))
	seller_counts.plot(kind="bar")
	plt.title("Top 20 Sellers by Number of Records")
	plt.xlabel("seller_id")
	plt.ylabel("Record count")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	plt.savefig(output_dir / "seller_id_top20_count.png")
	plt.close()


def plot_each_variable(data: pd.DataFrame, output_dir: Path) -> None:
	"""Generate and save a plot for each variable in the dataset."""
	_print_section("PLOTTING ANALYSIS")
	output_dir.mkdir(parents=True, exist_ok=True)

	plot_date_variable(data, output_dir)
	plot_high_cardinality_seller(data, output_dir)

	numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
	for column in numeric_cols:
		plot_numeric_variable(data, column, output_dir)

	for column in ["promo_flag", "is_default"]:
		if column in data.columns:
			plot_binary_or_category_variable(data, column, output_dir)

	print(f"Saved plots in: {output_dir}")


def run_eda(data_path: str, plot_output_dir: str) -> None:
	"""Run EDA in two phases: descriptive analysis, then plotting."""
	data = load_data(data_path)
	data = add_business_ratios(data)

	# Phase 1: descriptive analysis.
	descriptive_overview(data)
	descriptive_statistics(data)
	descriptive_target_distribution(data)
	descriptive_seller_checks(data)
	descriptive_ratio_summary(data)

	# Phase 2: variable-wise plotting.
	plot_each_variable(data, Path(plot_output_dir))

	print("\nEDA completed.")


if __name__ == "__main__":
	project_root = Path(__file__).resolve().parents[1]
	csv_path = project_root / "ecommerce_data.csv"
	plots_dir = project_root / "analysis" / "plots"
	run_eda(str(csv_path), str(plots_dir))