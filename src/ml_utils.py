import pandas as pd
import numpy as np


def universal_styler(
    df,
    metric_type="min",
    axis=0,  # Для основных таблиц по умолчанию поиск идет по вертикали
    subset=None,
    highlight_diff=False,
    col_train=None,
    col_test=None,
    min_r2_threshold=0.40,
    return_metrics=False,
    top_n=1,
):
    """
    Универсальный стилизатор ВЕРТИКАЛЬНЫХ таблиц результатов (axis=0).
    Ищет лучших и худших лидеров, а также рассчитывает стабильность (Absolute_Diff).
    """
    style_bad = "background-color: pink; color: darkred; font-weight: bold;"
    style_good = "background-color: lightgreen; color: darkgreen; font-weight: bold;"

    working_df = df.copy()

    # --- ЛОГИКА СТАБИЛЬНОСТИ (Поиск минимального разрыва Train/Test) ---
    if (
        highlight_diff
        and col_train in working_df.columns
        and col_test in working_df.columns
    ):
        working_df["Absolute_Diff"] = (
            working_df[col_train] - working_df[col_test]
        ).abs()

        row_names_str = working_df.index.astype(str)
        is_naive = row_names_str.str.contains("Naive|Mean|Median", case=False, na=False)
        if "model" in working_df.columns:
            is_naive = is_naive | working_df["model"].astype(str).str.contains(
                "Naive|Mean|Median", case=False, na=False
            )

        not_naive = working_df[~is_naive]
        max_test_r2 = not_naive[col_test].max()

        epsilon = 0.01
        top_quality_models = not_naive[not_naive[col_test] >= (max_test_r2 - epsilon)]

        no_overfit_models = top_quality_models[
            top_quality_models[col_test] >= top_quality_models[col_train]
        ]
        final_pool = (
            no_overfit_models if not no_overfit_models.empty else top_quality_models
        )

        stable_indices = final_pool["Absolute_Diff"].nsmallest(top_n).index

        if return_metrics:
            return (
                list(stable_indices)
                if "model" not in working_df.columns
                else working_df.loc[stable_indices, "model"].tolist()
            )

        styler = working_df.style.format(
            lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x,
            subset=["Absolute_Diff"],
        ).format(
            lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x,
            subset=df.columns,
        )
        return styler.apply(
            lambda r: (
                [style_good] * len(r) if r.name in stable_indices else [""] * len(r)
            ),
            axis=1,
        )

    # --- СТАНДАРТНЫЙ ВЕРТИКАЛЬНЫЙ ПОИСК (axis=0) БЕЗ НАИВНЫХ МОДЕЛЕЙ ---
    styler = working_df.style.format(
        lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x
    )

    active_rows = working_df[
        ~working_df["model"]
        .astype(str)
        .str.contains("Naive|Mean|Median", case=False, na=False)
    ]

    target_col = (
        subset
        if subset is not None
        else working_df.select_dtypes(include=[int, float]).columns[0]
    )

    if metric_type == "max":
        best_indices = active_rows[target_col].nlargest(top_n).index
        worst_idx = active_rows[target_col].idxmin()
    else:
        best_indices = active_rows[target_col].nsmallest(top_n).index
        worst_idx = active_rows[target_col].idxmax()

    def style_vertical(col):
        if col.name == target_col:
            return [
                style_good
                if idx in best_indices
                else (style_bad if idx == worst_idx else "")
                for idx in col.index
            ]
        return [""] * len(col)

    return styler.apply(style_vertical, axis=0)


def highlight_results(row, top_n=1, r2=True, mae=True, rmse=True):
    """
    Стилизатор строк DataFrame. По умолчанию подсвечивает все тестовые метрики.
    Позволяет отключить подсветку передачей параметров r2=False, mae=False или rmse=False.
    """
    styles = [""] * len(row)
    style_bad = "background-color: pink; color: darkred; font-weight: bold;"
    style_good = "background-color: lightgreen; color: darkgreen; font-weight: bold;"

    # Подсвечиваем только если в названии строки есть "Test"
    if "Test" in str(row.name):
        vals = row.to_numpy(dtype=float)
        sorted_indices = vals.argsort()

        # Проверка для R2
        if "R2" in str(row.name) and r2:
            styles[sorted_indices[0]] = style_bad
            for i in range(1, top_n + 1):
                styles[sorted_indices[-i]] = style_good

        # Проверка для MAE
        elif "MAE" in str(row.name) and mae:
            styles[sorted_indices[-1]] = style_bad
            for i in range(top_n):
                styles[sorted_indices[i]] = style_good

        # Проверка для RMSE
        elif "RMSE" in str(row.name) and rmse:
            styles[sorted_indices[-1]] = style_bad
            for i in range(top_n):
                styles[sorted_indices[i]] = style_good

    return styles


import pickle
import os


def save_metrics_checkpoint(
    mae_df, rmse_df, r2_df, sk_dict=None, my_dict=None, folder="checkpoints"
):
    """
    Сохраняет датафреймы метрик в CSV, а словари моделей — в бинарный файл pickle.
    Создает папку checkpoints автоматически, если её нет.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    mae_df.to_csv(os.path.join(folder, "result_MAE.csv"), index=False)
    rmse_df.to_csv(os.path.join(folder, "result_RMSE.csv"), index=False)
    r2_df.to_csv(os.path.join(folder, "result_R2.csv"), index=False)

    if sk_dict is not None:
        with open(os.path.join(folder, "sk_models.pkl"), "wb") as f:
            pickle.dump(sk_dict, f)

    if my_dict is not None:
        with open(os.path.join(folder, "my_models.pkl"), "wb") as f:
            pickle.dump(my_dict, f)

    print(f"Все метрики и словари успешно сохранены в папку '{folder}'")


def load_metrics_checkpoint(folder="checkpoints"):
    """
    Загружает сохраненные датафреймы и словари из папки checkpoints.
    Возвращает кортеж: (mae_df, rmse_df, r2_df, sk_dict, my_dict)
    Если файлов нет, возвращает None для словарей.
    """

    mae_df = (
        pd.read_csv(os.path.join(folder, "result_MAE.csv"))
        if os.path.exists(os.path.join(folder, "result_MAE.csv"))
        else pd.DataFrame(columns=["model", "train", "test"])
    )
    rmse_df = (
        pd.read_csv(os.path.join(folder, "result_RMSE.csv"))
        if os.path.exists(os.path.join(folder, "result_RMSE.csv"))
        else pd.DataFrame(columns=["model", "train", "test"])
    )
    r2_df = (
        pd.read_csv(os.path.join(folder, "result_R2.csv"))
        if os.path.exists(os.path.join(folder, "result_R2.csv"))
        else pd.DataFrame(columns=["model", "train", "test"])
    )

    sk_dict = None
    if os.path.exists(os.path.join(folder, "sk_models.pkl")):
        with open(os.path.join(folder, "sk_models.pkl"), "rb") as f:
            sk_dict = pickle.load(f)

    my_dict = None
    if os.path.exists(os.path.join(folder, "my_models.pkl")):
        with open(os.path.join(folder, "my_models.pkl"), "rb") as f:
            my_dict = pickle.load(f)

    print(f"Метрики успешно восстановлены из папки '{folder}'")
    return mae_df, rmse_df, r2_df, sk_dict, my_dict
