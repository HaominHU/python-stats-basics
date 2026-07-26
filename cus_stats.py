import statistics
import math
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
# smf.ols, smf.glm
import pandas as pd

def unique_total(series):
    return len(set(series))

def total(df):
    return len(df)

def count_by_condition(df, col,condition):
    return len(df[df[col] == condition])

def get_mean(series, cus_string):
    print(f"Mean {cus_string}: {statistics.mean(series): .2f}")

def get_sd(series, cus_string):
    print(f"SD {cus_string}: {statistics.stdev(series): .2f}")

def get_mean_sd(series, cus_string):
    return f"{cus_string} Mean (SD): {statistics.mean(series): .2f} ({statistics.stdev(series): .2f})"

def get_range(series, cus_string):
    print(f"Range {cus_string}: {min(series): .2f}, {max(series): .2f}")

def get_subset(df, col, subset_label):
    tmp_df = df.copy()
    return tmp_df[tmp_df[col] == subset_label]

def get_subset_excluding(df, col, subset_label):
    tmp_df = df.copy()
    return tmp_df[tmp_df[col] != subset_label]

# Categorical Data
def get_freq_table_with_percentage(series, total):
    for index, count in series.value_counts().sort_index().items():
        print(f"{index}: {count}/{total} ({count/total*100:.2f}%)")


def cronbach_alpha_analysis(df, start_col=0, end_col=None, drop_incomplete_rows=True, report_label=None):
    """
    Compute Cronbach's alpha and item diagnostics for a set of scale items.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe that contains item columns.
    start_col : int, default 0
        Start column index (0-based, inclusive) for item selection.
    end_col : int or None, default None
        End column index (0-based, exclusive). None means select to the last column.
    drop_incomplete_rows : bool, default True
        If True, keep only complete cases across selected items.
    report_label : str or None, default None
        Optional label used in printed summary.

    Returns
    -------
    dict with:
        alpha: float
        n_rows: int
        n_items: int
        items_complete: pd.DataFrame
        item_total_corr: pd.Series
        alpha_if_dropped: pd.Series
    """
    items = df.iloc[:, start_col:end_col].apply(pd.to_numeric, errors='coerce')
    items_complete = items.dropna(axis=0, how='any') if drop_incomplete_rows else items

    k = items_complete.shape[1]
    n = items_complete.shape[0]

    if k < 2:
        raise ValueError("Need at least 2 items (columns) to compute Cronbach's alpha.")
    if n < 2:
        raise ValueError("Need at least 2 rows to compute Cronbach's alpha.")

    item_variances = items_complete.var(axis=0, ddof=1)
    total_scores = items_complete.sum(axis=1)
    total_variance = total_scores.var(ddof=1)

    if total_variance == 0:
        raise ValueError("Total score variance is 0; Cronbach's alpha is undefined.")

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

    item_total_corr = {}
    for col in items_complete.columns:
        rest_score = total_scores - items_complete[col]
        item_total_corr[col] = items_complete[col].corr(rest_score)

    alpha_if_dropped = {}
    for col in items_complete.columns:
        subset = items_complete.drop(columns=[col])
        k_sub = subset.shape[1]

        if k_sub < 2:
            alpha_if_dropped[col] = math.nan
            continue

        total_sub = subset.sum(axis=1)
        total_var_sub = total_sub.var(ddof=1)
        if total_var_sub == 0:
            alpha_if_dropped[col] = math.nan
            continue

        var_sub = subset.var(axis=0, ddof=1)
        alpha_if_dropped[col] = (k_sub / (k_sub - 1)) * (1 - var_sub.sum() / total_var_sub)

    item_total_corr_s = pd.Series(item_total_corr).round(3)
    alpha_if_dropped_s = pd.Series(alpha_if_dropped).round(4)

    label = report_label if report_label is not None else f"columns {start_col + 1}+"
    print(f"Cronbach's alpha ({label}): {alpha:.4f}")
    print(f"Using {n} rows and {k} items.")

    print("\nCorrected item-total correlations:")
    for col, corr in item_total_corr_s.items():
        print(f"  {col}: {corr}")

    print("\nAlpha if item dropped:")
    for col, a in alpha_if_dropped_s.items():
        print(f"  {col}: {a}")

    return {
        'alpha': alpha,
        'n_rows': n,
        'n_items': k,
        'items_complete': items_complete,
        'item_total_corr': item_total_corr_s,
        'alpha_if_dropped': alpha_if_dropped_s,
    }

# Correlation
def get_correlation_matrix(df, cols, method='spearman'):
    """Pairwise correlation matrix using scipy. Returns a square r-value DataFrame."""
    from itertools import combinations
    import numpy as np
    result = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c in cols:
        result.loc[c, c] = 1.0
    for x, y in combinations(cols, 2):
        pair = df[[x, y]].dropna()
        if method == 'pearson':
            r, _ = stats.pearsonr(pair[x], pair[y])
        else:
            r, _ = stats.spearmanr(pair[x], pair[y])
        result.loc[x, y] = round(r, 4)
        result.loc[y, x] = round(r, 4)
    print(f"Correlation Matrix ({method}):\n")
    print(result)
    return result

def get_correlation(df, cols, method='spearman'):
    """Pairwise correlations using scipy. method = 'pearson' or 'spearman'."""
    from itertools import combinations
    rows = []
    for x, y in combinations(cols, 2):
        pair = df[[x, y]].dropna()
        n = len(pair)
        if method == 'pearson':
            r, p = stats.pearsonr(pair[x], pair[y])
        else:
            r, p = stats.spearmanr(pair[x], pair[y])
        rows.append({
            'X': x,
            'Y': y,
            'method': method,
            'n': n,
            'r': round(r, 4),
            'p-val': '<0.001' if p < 0.001 else f'{p:.4f}',
        })
    result = pd.DataFrame(rows)
    display(result)
    return result

# Assumptions Basic

def levene_test(series1, series2):
    print(f"Levene test: {stats.levene(series1, series2)}")
    levene_stats, levene_p = stats.levene(series1, series2)
    return levene_stats, levene_p

def brown_forsythe_test(series_arr):
    stat, p = stats.levene(*series_arr, center='median')
    print(f'Brown-Forsythe test: W={stat}, p-value={p}')

def normality_test(series, series_name, new_color=False):
    # Histogram
    if (not new_color):
        series.plot(kind='hist', title=f"{series_name} Histogram")
    else:
        series.plot(kind='hist', title=f"{series_name} Histogram", color='green')
    plt.xlabel('Value')
    plt.show()
    # Q-Q plot
    stats.probplot(series, dist="norm", plot= plt)
    plt.title(f"{series_name} Q-Q Plot")
    plt.show()
    # Shapiro-Wilk test
    print(f"Shapiro-Wilk test for {series_name}: {stats.shapiro(series)}")
    shapiro_stats, shapiro_p = stats.shapiro(series)
    return shapiro_p

# Effect Size
# Cohen's d
def cohen_d(group1, group2):
    return (statistics.mean(group1) - statistics.mean(group2)) / (math.sqrt((statistics.stdev(group1) ** 2 + statistics.stdev(group2) ** 2) / 2))

# Cohen's f
def cohen_f(rqaured):
    return math.sqrt(rqaured / (1 - rqaured))

# T tests
def ttest_assumption(series1, series2, cus_string1, cus_string2):
    shapiro_p1 = normality_test(series1, cus_string1)
    shapiro_p2 = normality_test(series2, cus_string2)
    levene_stats, levene_p = levene_test(series1, series2)
    return shapiro_p1, shapiro_p2, levene_p

# independent t-test
def ttest(series1, series2, cus_string1, cus_string2, label=None):
    t_stat, p_val = stats.ttest_ind(series1, series2)
    row = {"comparison": f"{cus_string1} vs {cus_string2}", "t_statistic": t_stat, "p_value": p_val}
    if label is not None:
        row["label"] = label
    return pd.DataFrame([row])

# paired t-test
def paired_ttest(before, after, cus_string_before, cus_string_after, label=None):
    t_stat, p_val = stats.ttest_rel(before, after)
    row = {"comparison": f"paired: {cus_string_before} vs {cus_string_after}", "t_statistic": t_stat, "p_value": p_val}
    if label is not None:
        row["label"] = label
    return pd.DataFrame([row])

# ANOVA
def anova_assumption(series_arr, group_names):
    for idx in range(len(series_arr)):
        normality_test(series_arr[idx], group_names[idx])
    brown_forsythe_test(series_arr)

def one_way_anova(series_arr, group_names):
    f_statistic, p_value = stats.f_oneway(*series_arr)
    print(f"One-Way ANOVA in groups {group_names}: F={f_statistic:.3f}, p-value={p_value:.3f}")

# Regression
# Linear Regression Forward
def forward_selection(X, y, covariate_columns=None, significance_level=.05):
    included_features = covariate_columns.copy() if covariate_columns else []

    while True:
        adding_features = [col for col in X.columns if col not in included_features]
        if not adding_features:
            break;
        pvalues = []

        # Fit the model with the current set of included features
        model = sm.OLS(y, sm.add_constant(X[included_features])).fit()
        print("\n ----------Init Model fit controlling covars %s: ---------- \n %s:" %(included_features, model.summary()))

        # Calculate p-values for adding features
        for feature in adding_features:
            model_with_iv = sm.OLS(y, sm.add_constant(X[included_features + [feature]])).fit()
            print("\n ---------- Evaluation Model fit adding IV %s: ---------- \n %s:" %(feature, model_with_iv.summary()))
            p_value = model_with_iv.pvalues[feature]
            pvalues.append((feature, p_value))

        pvalues.sort(key=lambda x: x[1])
        best_pvalue = pvalues[0][1]

        # Check if the best p-value is below the significance level
        if pvalues and best_pvalue < significance_level:
            included_features.append(pvalues[0][0])
        else:
            break

    return included_features

def linear_regression(df, iv_name, dv_name, covar_names=None, significance_level=None, withSelection=False):
    if (covar_names):
        print(f"-----Linear Regression between {iv_name} and {dv_name} controlling for {''.join(covar_names) if len(covar_names) == 1 else ','.join(covar_names)}-----")
        if withSelection:
            # Forward selection
            selected_features = forward_selection(df[[iv_name] + covar_names], df[dv_name], covariate_columns=covar_names, significance_level=significance_level if significance_level else .05)
            print("----------Feature Selection Done, Finalizing the model----------")
            final_model = sm.OLS(df[dv_name], sm.add_constant(df[selected_features])).fit()
            print(f"---------- Final Model Fit: ----------")
            print(final_model.summary())
        else:
            reg_model = sm.OLS(df[dv_name], sm.add_constant(df[[iv_name] + covar_names])).fit()
            print(f"---------- Model Fit: ----------".center(100))
            print(reg_model.summary())
    else :
        print(f"-----Linear Regression between {iv_name} and {dv_name}-----")
        # if iv_name is a string
        if isinstance(iv_name, str):
            reg_model = sm.OLS(df[dv_name], sm.add_constant(df[[iv_name]])).fit()
        else:
            reg_model = sm.OLS(df[dv_name], sm.add_constant(df[iv_name])).fit()
        print(f"---------- Model Fit: ----------".center(100))
        print(reg_model.summary())

# Report-ready OLS helpers ────────────────────────────────────────────────────
def run_ols_simple(df, dv, iv, label=None, total_n=None):
    """
    Simple OLS regression (no covariates). Returns a clean results table.

    Parameters
    ----------
    df       : DataFrame with at least [dv, iv] columns
    dv       : str, dependent variable
    iv       : str, independent variable (main predictor)
    label    : str, optional display label
    total_n  : int, optional reference N for missing count; defaults to len(df)
    """
    df_model = df[[dv, iv]].dropna().reset_index(drop=True)
    N       = len(df_model)
    ref_n   = total_n if total_n is not None else len(df)
    missing = ref_n - N

    X = sm.add_constant(df_model[[iv]])
    y = df_model[dv]
    result = sm.OLS(y, X).fit()

    conf = result.conf_int()
    conf.columns = ['CI_lower', 'CI_upper']

    table = pd.DataFrame({
        'Predictor': result.params.index,
        'B':         result.params.values.round(4),
        'SE':        result.bse.values.round(4),
        'p-value':   result.pvalues.values.round(4),
        'CI_lower':  conf['CI_lower'].values.round(4),
        'CI_upper':  conf['CI_upper'].values.round(4),
    })
    table = table[table['Predictor'] != 'const'].reset_index(drop=True)
    table['p-value'] = table['p-value'].apply(lambda p: '<0.001' if p < 0.001 else f'{p:.4f}')

    hdr = label if label else f'OLS: {iv} \u2192 {dv}'
    print(f"\n{'\u2500'*60}")
    print(f"  {hdr}")
    print(f"  N = {N} | Missing = {missing}")
    print(f"{'\u2500'*60}")
    try:
        display(table)
    except NameError:
        print(table.to_string(index=False))
    return table


def run_ols_cov(df, dv, iv, covariates, label=None, total_n=None, covariate_df=None):
    """
    OLS regression with covariates. Returns a clean results table.

    Parameters
    ----------
    df           : DataFrame with at least [SubjectID, dv, iv] columns
    dv           : str, dependent variable
    iv           : str, independent variable (main predictor)
    covariates   : list of str, covariate column names
    label        : str, optional display label
    total_n      : int, optional reference N; defaults to len(df)
    covariate_df : optional DataFrame with [SubjectID] + covariates columns.
                   If provided, left-merged onto df by SubjectID before fitting.
    """
    if covariate_df is not None:
        df = df.merge(covariate_df[['SubjectID'] + covariates], on='SubjectID', how='left')
    model_cols = [dv, iv] + covariates
    df_model = df[model_cols].dropna().reset_index(drop=True)
    N       = len(df_model)
    ref_n   = total_n if total_n is not None else len(df)
    missing = ref_n - N

    X = sm.add_constant(df_model[[iv] + covariates])
    y = df_model[dv]
    result = sm.OLS(y, X).fit()

    conf = result.conf_int()
    conf.columns = ['CI_lower', 'CI_upper']

    table = pd.DataFrame({
        'Predictor': result.params.index,
        'B':         result.params.values.round(4),
        'SE':        result.bse.values.round(4),
        'p-value':   result.pvalues.values.round(4),
        'CI_lower':  conf['CI_lower'].values.round(4),
        'CI_upper':  conf['CI_upper'].values.round(4),
    })
    table = table[table['Predictor'] != 'const'].reset_index(drop=True)
    table['p-value'] = table['p-value'].apply(lambda p: '<0.001' if p < 0.001 else f'{p:.4f}')

    hdr = label if label else f'OLS+cov: {iv} \u2192 {dv}'
    print(f"\n{'\u2500'*60}")
    print(f"  {hdr}")
    print(f"  N = {N} | Missing = {missing}")
    print(f"{'\u2500'*60}")
    try:
        display(table)
    except NameError:
        print(table.to_string(index=False))
    return table


# Logistic Regress
def logit_forward(X, y, covariate_columns=None, significance_level=.05):
    included_features = covariate_columns.copy() if covariate_columns else []

    while True:
        adding_features = [col for col in X.columns if col not in included_features]
        if not adding_features:
            break;
        pvalues = []

        # Fit the model with the current set of included features
        model = sm.Logit(y, sm.add_constant(X[included_features])).fit(disp=0)
        print("\n ----------Init Model fit controlling covars %s: ---------- \n %s:" %(included_features, model.summary()))

        # Calculate p-values for adding features
        for feature in adding_features:
            model_with_iv = sm.Logit(y, sm.add_constant(X[included_features + [feature]])).fit(disp=0)
            print("\n ---------- Evaluation Model fit adding IV %s: ---------- \n %s:" %(feature, model_with_iv.summary()))
            p_value = model_with_iv.pvalues[feature]
            pvalues.append((feature, p_value))

        pvalues.sort(key=lambda x: x[1])
        best_pvalue = pvalues[0][1]

        # Check if the best p-value is below the significance level
        if pvalues and best_pvalue < significance_level:
            included_features.append(pvalues[0][0])
        else:
            break

    return included_features


def run_logit_cov(df, dv, iv, covariates, label=None, total_n=None, covariate_df=None):
    """
    Logistic regression with covariates. DV must be binary (0/1).
    Tries newton \u2192 bfgs \u2192 lbfgs; skips if N \u2264 n_params + 5.

    Parameters
    ----------
    df           : DataFrame with at least [SubjectID, dv, iv] columns
    dv           : str, binary dependent variable (0/1)
    iv           : str, independent variable (main predictor)
    covariates   : list of str, covariate column names
    label        : str, optional display label
    total_n      : int, optional reference N; defaults to len(df)
    covariate_df : optional DataFrame with [SubjectID] + covariates columns.
                   If provided, left-merged onto df by SubjectID before fitting.

    Returns
    -------
    pd.DataFrame with Predictor, B (log-OR), OR, SE, p-value, OR_CI_lower,
    OR_CI_upper — or None if skipped/failed.
    """
    import numpy as np
    if covariate_df is not None:
        df = df.merge(covariate_df[['SubjectID'] + covariates], on='SubjectID', how='left')
    model_cols = [dv, iv] + covariates
    df_model = df[model_cols].dropna().reset_index(drop=True)
    N       = len(df_model)
    ref_n   = total_n if total_n is not None else len(df)
    missing = ref_n - N
    n_params = 1 + len([iv] + covariates)

    hdr = label if label else f'Logit+cov: {iv} \u2192 {dv}'
    print(f"\n{'\u2500'*60}")
    print(f"  {hdr}")
    print(f"  N = {N} | Missing = {missing}")
    if N <= n_params + 5:
        print(f"  \u26a0 Skipped: N={N} too small for {n_params} parameters.")
        print(f"{'\u2500'*60}")
        return None

    X = sm.add_constant(df_model[[iv] + covariates])
    y = df_model[dv].astype(int)

    result = None
    for method in ('newton', 'bfgs', 'lbfgs'):
        try:
            result = sm.Logit(y, X).fit(method=method, disp=False, maxiter=200)
            break
        except Exception:
            continue

    if result is None:
        print(f"  \u26a0 Model failed to converge with all methods.")
        print(f"{'\u2500'*60}")
        return None

    conf = result.conf_int()
    conf.columns = ['CI_lower', 'CI_upper']

    table = pd.DataFrame({
        'Predictor':   result.params.index,
        'B (log-OR)':  result.params.values.round(4),
        'OR':          np.exp(result.params.values).round(4),
        'SE':          result.bse.values.round(4),
        'p-value':     result.pvalues.values.round(4),
        'OR_CI_lower': np.exp(conf['CI_lower'].values).round(4),
        'OR_CI_upper': np.exp(conf['CI_upper'].values).round(4),
    })
    table = table[table['Predictor'] != 'const'].reset_index(drop=True)
    table['p-value'] = table['p-value'].apply(lambda p: '<0.001' if p < 0.001 else f'{p:.4f}')

    print(f"{'\u2500'*60}")
    try:
        display(table)
    except NameError:
        print(table.to_string(index=False))
    return table


# Non parametric tests
# Chi-square contigeny table
def chi_square_cont(data, group_col=None, outcome_col=None, label=None):
    if isinstance(data, pd.DataFrame) and group_col is not None and outcome_col is not None:
        observed = pd.crosstab(data[group_col], data[outcome_col])
    else:
        # Fallback for list of lists input
        observed_lists = data
        cus_strings = group_col
        group = []
        outcome = []
        for idx in range(len(observed_lists)):
            group += [cus_strings[idx]] * len(observed_lists[idx])
            outcome += observed_lists[idx]
        df_observed = pd.DataFrame({'group': group, 'outcome': outcome})
        observed = pd.crosstab(df_observed['group'], df_observed['outcome'])

    chi2, p, dof, expected = stats.chi2_contingency(observed)
    rates = observed.div(observed.sum(axis=0), axis=1) * 100
    results_row = {"chi2": chi2, "p_value": p, "dof": dof}
    if label is not None:
        results_row["label"] = label
    return {
        "results": pd.DataFrame([results_row]),
        "contingency_table": observed,
        "rates": rates.round(2),
        "expected": pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    }

# Fisher's Exact Test 2x2
def fisher_exact_test(data, group_col=None, outcome_col=None):
    if isinstance(data, pd.DataFrame) and group_col is not None and outcome_col is not None:
        observed = pd.crosstab(data[group_col], data[outcome_col])
    else:
        # Fallback for list of lists input
        observed_lists = data
        cus_strings = group_col
        group = []
        outcome = []
        for idx in range(len(observed_lists)):
            group += [cus_strings[idx]] * len(observed_lists[idx])
            outcome += observed_lists[idx]
        df_observed = pd.DataFrame({'group': group, 'outcome': outcome})
        observed = pd.crosstab(df_observed['group'], df_observed['outcome'])

    odds_ratio, p = stats.fisher_exact(observed)
    print(f"Contingency Table:\n{observed}")
    print(f"Odds Ratio: {odds_ratio}")
    print(f"P-Value: {p}")

# McNemar's Test
def mcnemar_test(df, before_str, after_str):
    b = ((df[before_str] == 0) & (df[after_str] == 1)).sum()
    c = ((df[before_str] == 1) & (df[after_str] == 0)).sum()
    table = [
        [0, b],
        [c, 0]
    ]
    result = mcnemar(table, exact=True)
    print(f"McNemar’s test statistic: {result.statistic}")
    print(f"p-value: {result.pvalue:.4f}")

# Mann-Whitney U test
def mann_whitney_u(control, treat, ctrl_string, treat_string, label=None):
    u_statistic, p_val = stats.mannwhitneyu(control, treat)
    row = {"comparison": f"{ctrl_string} vs {treat_string}", "u_statistic": u_statistic, "p_value": p_val}
    if label is not None:
        row["label"] = label
    return pd.DataFrame([row])

# Kruskal-Wallis H test
def kruskal_wallis_h(series_arr, group_names):
    h_statistic, p = stats.kruskal(*series_arr)
    print(f'Kruskal-Wallis H test between {group_names}: H={h_statistic:.3f}, p-value={p:.3f}')
