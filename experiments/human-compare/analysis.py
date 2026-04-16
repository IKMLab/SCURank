import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
df = pd.read_csv('experiments/human-compare/data/human_evaluation_results.csv')

print("=" * 80)
print("HUMAN ANNOTATION ANALYSIS: SCU vs GPT Summaries")
print("=" * 80)
print(f"\nDataset Overview:")
print(f"Total annotations: {len(df)}")
print(f"Data sources: {df['Input.data_label'].unique().tolist()}")
print(f"Summary methods: {df['Input.label'].unique().tolist()}")


# ── Label explanation ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("UNDERSTANDING THE LABELS")
print("=" * 80)
print("\nInput.label = scu-gpt means:")
print("  - Summary 1 (S1): SCU-based summary")
print("  - Summary 2 (S2): GPT-based summary")
print("\nInput.label = gpt-scu means:")
print("  - Summary 1 (S1): GPT-based summary")
print("  - Summary 2 (S2): SCU-based summary")
print(f"\nDistribution:")
print(df['Input.label'].value_counts())


# ── Normalize preferences ─────────────────────────────────────────────────────
def normalize_preference(row):
    """Convert raw preference to: 'scu_better', 'gpt_better', or 'equal'."""
    pref = row['Answer.overall_preference']
    label = row['Input.label']

    if pref == 'equal':
        return 'equal'

    if label == 'scu-gpt':
        if pref in ['s1_much_better', 's1_better']:
            return 'scu_better'
        elif pref in ['s2_much_better', 's2_better']:
            return 'gpt_better'
    elif label == 'gpt-scu':
        if pref in ['s1_much_better', 's1_better']:
            return 'gpt_better'
        elif pref in ['s2_much_better', 's2_better']:
            return 'scu_better'

    return None


df['normalized_preference'] = df.apply(normalize_preference, axis=1)

print("\n" + "=" * 80)
print("CORE INSIGHT: SCU vs GPT PREFERENCE")
print("=" * 80)

pref_counts = df['normalized_preference'].value_counts()
total = len(df)

print("\nOverall Results:")
for pref, count in pref_counts.items():
    pct = count / total * 100
    print(f"  {pref.upper()}: {count} ({pct:.1f}%)")

scu_wins = pref_counts.get('scu_better', 0)
gpt_wins = pref_counts.get('gpt_better', 0)
equal = pref_counts.get('equal', 0)

print(f"\nSummary:")
print(f"  SCU Wins: {scu_wins} ({scu_wins/total*100:.1f}%)")
print(f"  GPT Wins: {gpt_wins} ({gpt_wins/total*100:.1f}%)")
print(f"  Equal: {equal} ({equal/total*100:.1f}%)")
print(f"  Advantage: {'SCU' if scu_wins > gpt_wins else 'GPT'} by {abs(scu_wins - gpt_wins)} votes")


# ── Article-level analysis ────────────────────────────────────────────────────
cnn_article_preferences = {}
xsum_article_preferences = {}

for _, row in df.iterrows():
    data_label = row['Input.data_label']
    bucket = cnn_article_preferences if data_label == 'cnn' else xsum_article_preferences
    bucket.setdefault(row['Input.article'], []).append(row['normalized_preference'])


def article_level_counts(article_preferences):
    result_counts = {"scu win": 0, "gpt win": 0, "equal": 0}
    for v in article_preferences.values():
        scu_better = v.count('scu_better')
        gpt_better = v.count('gpt_better')
        tie = v.count('equal')

        if scu_better >= 2:
            result = 'scu win'
        elif gpt_better >= 2:
            result = 'gpt win'
        elif tie >= 2:
            result = 'equal'
        elif scu_better == 1 and gpt_better == 1 and tie == 1:
            result = 'equal'
        else:
            result = 'uncategorized'

        if result in result_counts:
            result_counts[result] += 1
    return result_counts


cnn_counts = article_level_counts(cnn_article_preferences)
xsum_counts = article_level_counts(xsum_article_preferences)

print("\nCNN Article-level Results:")
for result, count in cnn_counts.items():
    print(f"  {result}: {count}")

print("\nXSum Article-level Results:")
for result, count in xsum_counts.items():
    print(f"  {result}: {count}")


# ── Figure 1: Combined bar chart (XSum + CNN) ─────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

comparison_data = {
    "XSum": [xsum_counts["scu win"], xsum_counts["equal"], xsum_counts["gpt win"]],
    "CNN":  [cnn_counts["scu win"],  cnn_counts["equal"],  cnn_counts["gpt win"]],
}


def process_data(data):
    data_percentage = {}
    for model, values in data.items():
        total = sum(values)
        data_percentage[model] = [v / total * 100 for v in values]
    return pd.DataFrame(data_percentage, index=['SCURank', 'Tie', 'GPTRank'])


def create_bar_chart(ax, df_plot, show_ylabel=True):
    colors = ['#87CEEB', '#DDA0DD', '#FFE4B5']
    df_plot.T.plot(
        kind='barh', stacked=True,
        color=colors, alpha=0.75, ax=ax,
        width=0.8, edgecolor='white', linewidth=1, legend=False,
    )
    ax.set_xticklabels([f'{i}%' for i in range(0, 101, 20)], fontsize=16)
    ax.set_ylabel('')

    for i, model in enumerate(df_plot.columns):
        cumulative = 0
        for category in df_plot.index:
            value = df_plot.loc[category, model]
            if value > 5:
                ax.text(cumulative + value / 2, i, f'{value:.1f}%',
                        ha='center', va='center', fontweight='normal',
                        fontsize=16, color='black')
            cumulative += value

    ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('white')


df_combined = process_data(comparison_data)

fig, ax = plt.subplots(figsize=(8, 3))
create_bar_chart(ax=ax, df_plot=df_combined, show_ylabel=True)
plt.legend(['SCURank', 'Tie', 'GPTRank'],
           loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=16)
plt.tight_layout()
plt.savefig('human_preference_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()


# ── Figure 2: Side-by-side CNN / XSum ────────────────────────────────────────
cnn_scu_wins = df[(df['Input.data_label'] == 'cnn') & (df['normalized_preference'] == 'scu_better')]
cnn_gpt_wins = df[(df['Input.data_label'] == 'cnn') & (df['normalized_preference'] == 'gpt_better')]
cnn_equal    = df[(df['Input.data_label'] == 'cnn') & (df['normalized_preference'] == 'equal')]

xsum_scu_wins = df[(df['Input.data_label'] == 'xsum') & (df['normalized_preference'] == 'scu_better')]
xsum_gpt_wins = df[(df['Input.data_label'] == 'xsum') & (df['normalized_preference'] == 'gpt_better')]
xsum_equal    = df[(df['Input.data_label'] == 'xsum') & (df['normalized_preference'] == 'equal')]

print("\n" + "=" * 80)
print("DETAILED ANALYSIS BY DATA SOURCE")
print("=" * 80)

for label, scu_w, gpt_w, eq_df in [
    ("CNN/DailyMail", cnn_scu_wins, cnn_gpt_wins, cnn_equal),
    ("XSum",          xsum_scu_wins, xsum_gpt_wins, xsum_equal),
]:
    src_total = len(scu_w) + len(gpt_w) + len(eq_df)
    print(f"\n{label} Results:")
    print(f"  SCU Wins: {len(scu_w)} ({len(scu_w)/src_total*100:.1f}%)")
    print(f"  GPT Wins: {len(gpt_w)} ({len(gpt_w)/src_total*100:.1f}%)")
    print(f"  Equal: {len(eq_df)} ({len(eq_df)/src_total*100:.1f}%)")
    advantage = len(scu_w) - len(gpt_w)
    if advantage > 0:
        print(f"  Advantage: SCU by +{advantage}")
    elif advantage < 0:
        print(f"  Advantage: GPT by +{-advantage}")
    else:
        print("  Advantage: Equal")

data1 = {'scu': [len(cnn_scu_wins),  len(cnn_equal),  len(cnn_gpt_wins)]}
data2 = {'scu': [len(xsum_scu_wins), len(xsum_equal), len(xsum_gpt_wins)]}

df1 = process_data(data1)
df2 = process_data(data2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4), sharey=True)
create_bar_chart(ax1, df1, show_ylabel=True)
create_bar_chart(ax2, df2, show_ylabel=False)
ax1.set_title("CNN/DM", fontsize=20)
ax2.set_title("XSum", fontsize=20)
handles, labels_legend = ax1.get_legend_handles_labels()
fig.legend(handles, labels_legend, bbox_to_anchor=(0.5, 0.0),
           loc='center', frameon=False, ncol=3, fontsize=16)
fig.patch.set_facecolor('white')
plt.subplots_adjust(wspace=0.1)
plt.tight_layout()
plt.show()


# ── Per-metric scores ─────────────────────────────────────────────────────────
metric_data = {
    "cnn":  {'scu': {m: [] for m in ['completeness', 'conciseness', 'faithfulness', 'article', 'summary']},
             'gpt': {m: [] for m in ['completeness', 'conciseness', 'faithfulness', 'article', 'summary']}},
    "xsum": {'scu': {m: [] for m in ['completeness', 'conciseness', 'faithfulness', 'article', 'summary']},
             'gpt': {m: [] for m in ['completeness', 'conciseness', 'faithfulness', 'article', 'summary']}},
}

for _, row in df.iterrows():
    ds = row['Input.data_label']
    label = row['Input.label']

    if label == 'scu-gpt':
        scu_s, gpt_s = 's1', 's2'
        scu_sum, gpt_sum = row['Input.summary1'], row['Input.summary2']
    else:  # gpt-scu
        scu_s, gpt_s = 's2', 's1'
        scu_sum, gpt_sum = row['Input.summary2'], row['Input.summary1']

    for metric in ['completeness', 'conciseness', 'faithfulness']:
        metric_data[ds]['scu'][metric].append(row[f'Answer.{scu_s}_{metric}'])
        metric_data[ds]['gpt'][metric].append(row[f'Answer.{gpt_s}_{metric}'])

    metric_data[ds]['scu']['article'].append(row['Input.article'])
    metric_data[ds]['scu']['summary'].append(scu_sum)
    metric_data[ds]['gpt']['article'].append(row['Input.article'])
    metric_data[ds]['gpt']['summary'].append(gpt_sum)

for dataset in metric_data:
    print(f"\n{dataset.upper()} METRIC AVERAGES:")
    for model in metric_data[dataset]:
        print(f"  {model.upper()}:")
        for metric in ['completeness', 'conciseness', 'faithfulness']:
            avg = np.mean(metric_data[dataset][model][metric])
            print(f"    {metric.capitalize()}: {avg:.2f}")


# ── Gap analysis ──────────────────────────────────────────────────────────────
def gap_analysis(data):
    for ds in data:
        articles      = data[ds]['scu']['article']
        scu_summaries = data[ds]['scu']['summary']
        gpt_summaries = data[ds]['gpt']['summary']
        art2idx       = {article: idx for idx, article in enumerate(articles)}
        overall_gap   = {article: 0 for article in articles}

        for metric in ['completeness', 'conciseness', 'faithfulness']:
            scu_scores = data[ds]['scu'][metric]
            gpt_scores = data[ds]['gpt'][metric]
            for article, gap in zip(articles, [s - g for s, g in zip(scu_scores, gpt_scores)]):
                overall_gap[article] += gap

        print(f"\n{ds.upper()} GAP ANALYSIS:")
        print("Top 5 articles where SCU > GPT:")
        for article in sorted(overall_gap, key=overall_gap.get, reverse=True)[:5]:
            idx = art2idx[article]
            print(f"  Gap: {overall_gap[article]:.2f}, Idx: {idx}")
            print(f"  SCU Scores: "
                  f"{data[ds]['scu']['completeness'][idx]}, "
                  f"{data[ds]['scu']['conciseness'][idx]}, "
                  f"{data[ds]['scu']['faithfulness'][idx]}")
            print(f"  GPT Scores: "
                  f"{data[ds]['gpt']['completeness'][idx]}, "
                  f"{data[ds]['gpt']['conciseness'][idx]}, "
                  f"{data[ds]['gpt']['faithfulness'][idx]}")
            print(f"  SCU Summary: {scu_summaries[idx]}")
            print(f"  GPT Summary: {gpt_summaries[idx]}")
            print("  -----")


gap_analysis(metric_data)


# ── Figure 3: Metric bar chart with CI and significance ───────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12


def plot_metric_scores(data):
    fields = ['completeness', 'conciseness', 'faithfulness']
    c_scu_cnn  = '#87CEEB'
    c_scu_xsum = '#A9A9F5'
    c_gpt_cnn  = '#FFE4B5'
    c_gpt_xsum = '#F4A460'

    plt.figure(figsize=(10, 6))

    for i, field in enumerate(fields):
        scu_cnn_vals  = data['cnn']['scu'][field]
        scu_xsum_vals = data['xsum']['scu'][field]
        gpt_cnn_vals  = data['cnn']['gpt'][field]
        gpt_xsum_vals = data['xsum']['gpt'][field]

        n = 30
        means = [np.mean(v) for v in [scu_cnn_vals, scu_xsum_vals, gpt_cnn_vals, gpt_xsum_vals]]
        cis   = [1.96 * (np.std(v, ddof=1) / np.sqrt(n))
                 for v in [scu_cnn_vals, scu_xsum_vals, gpt_cnn_vals, gpt_xsum_vals]]

        rects_scu_cnn  = plt.barh(i + 0.3,  means[0], 0.2, xerr=cis[0], label='SCURank-CNN',  color=c_scu_cnn,  capsize=5)
        rects_gpt_cnn  = plt.barh(i + 0.1,  means[2], 0.2, xerr=cis[2], label='GPTRank-CNN',  color=c_gpt_cnn,  capsize=5)
        rects_scu_xsum = plt.barh(i - 0.1,  means[1], 0.2, xerr=cis[1], label='SCURank-XSum', color=c_scu_xsum, capsize=5)
        rects_gpt_xsum = plt.barh(i - 0.3,  means[3], 0.2, xerr=cis[3], label='GPTRank-XSum', color=c_gpt_xsum, capsize=5)

        if i == 0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)

        _, p_cnn  = ttest_ind(scu_cnn_vals,  gpt_cnn_vals)
        _, p_xsum = ttest_ind(scu_xsum_vals, gpt_xsum_vals)
        print(f"\n{field.capitalize()} — CNN p={p_cnn:.4f}, XSum p={p_xsum:.4f}")

        def annotate_bars(rects, ci, p_value, r_type):
            for r in rects:
                v     = r.get_width()
                x_loc = v + ci + 0.1
                sig   = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                text  = f'{v:.2f} ({ci:.2f}){sig}' if (r_type == 'scu' and sig) else f'{v:.2f} ({ci:.2f})'
                weight = 'bold' if (r_type == 'scu' and sig) else 'normal'
                plt.text(x_loc, r.get_y() + r.get_height() / 2, text,
                         va='center', ha='left', fontsize=12, fontweight=weight, color='#444')

        for rects, ci, p_value, r_type in zip(
            [rects_scu_cnn, rects_scu_xsum, rects_gpt_cnn, rects_gpt_xsum],
            cis,
            [p_cnn, p_xsum, p_cnn, p_xsum],
            ['scu', 'scu', 'gpt', 'gpt'],
        ):
            annotate_bars(rects, ci, p_value, r_type)

    plt.yticks(range(len(fields)), [f.capitalize() for f in fields])
    plt.xlabel('Average Score (with 95% CI)', fontsize=16)
    plt.xlim(0, 5.5)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig("human_evaluation_scores_by_metric_and_model.pdf", bbox_inches='tight')
    plt.show()


plot_metric_scores(metric_data)


# ── Statistical significance table ───────────────────────────────────────────
def calculate_pvalues(data):
    fields   = ['completeness', 'conciseness', 'faithfulness']
    datasets = ['cnn', 'xsum']
    results  = {}

    for ds in datasets:
        results[ds] = {}
        for field in fields:
            t_stat, p_value = stats.ttest_ind(data[ds]['scu'][field], data[ds]['gpt'][field])
            results[ds][field] = {
                'p_value':     p_value,
                't_statistic': t_stat,
                'significant': p_value < 0.05,
            }
    return results


pvalues = calculate_pvalues(metric_data)

print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE")
print("=" * 80)
for ds in ['cnn', 'xsum']:
    print(f"\n{ds.upper()}:")
    for field in ['completeness', 'conciseness', 'faithfulness']:
        p   = pvalues[ds][field]['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {field}: p = {p:.4f} {sig}")
