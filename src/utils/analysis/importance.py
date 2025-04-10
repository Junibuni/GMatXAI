import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def analyze_param_importance(df, target_col="val_mae", output_dir=None):
    df = df.dropna()

    X = df.drop(columns=[target_col, "tag", "error"], errors="ignore")
    y = df[target_col]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    correlations = X.corrwith(y)

    importance_df = pd.DataFrame({
        "hyperparameter": X.columns,
        "importance": importances,
        "correlation": correlations
    }).sort_values("importance", ascending=False)

    # if output_dir is None:
    #     output_dir = os.path.dirname(csv_path)
    # os.makedirs(output_dir, exist_ok=True)

    # plt.figure(figsize=(8, 5))
    # ax1 = plt.gca()
    # importance_df.plot.barh(
    #     x="hyperparameter", y="importance", ax=ax1,
    #     color="steelblue", legend=False
    # )
    # ax1.set_xlabel("Random Forest Importance")
    # ax1.set_title("Hyperparameter Importance")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "hparam_importance.png"))

    # out_csv = os.path.join(output_dir, "hparam_importance.csv")
    # importance_df.to_csv(out_csv, index=False)

    return importance_df

# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Data
# data = {
#     'Hyperparameter': ['optimizer_weight_decay', 'hidden_dim', 'num_layers', 'optimizer_name'],
#     'Importance': [0.930, 0.038, 0.028, 0.004],
#     'Correlation': [0.968, -0.127, 0.151, -0.527]
# }
# df = pd.DataFrame(data)

# # Settings
# bar_height = 0.1
# bar_color_imp = 'green'
# bar_color_corr_pos = 'blue'
# bar_color_corr_neg = 'red'
# max_imp = df['Importance'].max()
# max_corr = df['Correlation'].abs().max()

# # Plot
# fig, ax = plt.subplots(figsize=(10, 2 + len(df)))

# # Plot each row
# for i, row in df.iterrows():
#     y = len(df) - i - 1

#     ax.text(0.01, y, row['Hyperparameter'], va='center', fontsize=11)

#     # Importance bar
#     imp_bar_len = row['Importance'] / max_imp * 0.1
#     ax.add_patch(patches.Rectangle((0.35, y - bar_height / 2), imp_bar_len, bar_height,
#                                    color=bar_color_imp))
#     ax.text(0.35 + imp_bar_len + 0.01, y, f"{row['Importance']:.3f}", va='center', fontsize=10)

#     # Correlation bar
#     corr_val = row['Correlation']
#     corr_bar_len = abs(corr_val) / max_corr * 0.3
#     corr_color = bar_color_corr_pos if corr_val >= 0 else bar_color_corr_neg
#     ax.add_patch(patches.Rectangle((0.70, y - bar_height / 2), corr_bar_len, bar_height,
#                                    color=corr_color))
#     ax.text(0.70 + corr_bar_len + 0.01, y, f"{corr_val:+.3f}", va='center', fontsize=10)

# ax.set_xlim(0, 1.1)
# ax.set_ylim(-0.5, len(df) - 0.5)
# ax.axis('off')

# ax.text(0.01, len(df) + 0.2, "Hyperparameter", fontsize=12, fontweight='bold')
# ax.text(0.35, len(df) + 0.2, "Importance", fontsize=12, fontweight='bold')
# ax.text(0.70, len(df) + 0.2, "Correlation", fontsize=12, fontweight='bold')

# plt.tight_layout()
# # plt.savefig(".png", dpi=150)
# plt.show()
