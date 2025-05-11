import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("all_data.csv")

    # 문자열 제외
    exclude_columns = ['file_path', 'file_name']
    numeric_columns = [col for col in data.columns if col not in exclude_columns]

    # 결과 저장용 DataFrame 생성
    iqr_stats = []

    for col in numeric_columns:
        series = pd.to_numeric(data[col], errors='coerce')
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        iqr_low = q1 - 1.5 * iqr
        iqr_high = q3 + 1.5 * iqr
        iqr_stats.append({
            "field": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "IQR_low": iqr_low,
            "IQR_high": iqr_high,
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "std": series.std(),
        })

    # DataFrame으로 변환 후 저장 or 출력
    stats_df = pd.DataFrame(iqr_stats).set_index("field")
    print(stats_df)

    # CSV로 저장하고 싶으면 아래 사용
    stats_df.to_excel("field_statistics.xlsx")
