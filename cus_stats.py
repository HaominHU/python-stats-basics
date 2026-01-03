import statistics
import math
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
# smf.ols, smf.glm
import pingouin as pg
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

# Correlation
def get_correlation_matrix(df, cols, method):
    print(f"Correlation Matrix:\n")
    print(df[cols].rcorr(method=method))
    return df[cols].rcorr(method=method)

def get_correlation(df, cols, method):
    print(f"Correlation value:\n")
    print(df[cols].pairwise_corr(method=method))

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
def ttest(series1, series2, cus_string1, cus_string2):
    print(f"t-test between {cus_string1} and {cus_string2}: {stats.ttest_ind(series1, series2)}")

# paired t-test
def paired_ttest(before, after, cus_string_before, cus_string_after):
    print(f"Paired t-test between {cus_string_before} and {cus_string_after}: {stats.ttest_rel(before, after)}")

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

# Non parametric tests
# Chi-square contigeny table
def chi_square_cont(observed_lists, cus_strings):
    group = []
    outcome = []
    for idx in range(len(observed_lists)):
        group += [cus_strings[idx]] * len(observed_lists[idx])
        outcome += observed_lists[idx]
    df_observed = pd.DataFrame({'group': group, 'outcome': outcome})
    observed = pd.crosstab(df_observed['group'], df_observed['outcome'])
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f"Contingency Table:\n{observed}")
    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-Value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies: \n{expected}")

# Fisher's Exact Test 2x2
def fisher_exact_test(observed_lists, cus_strings):
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
def mann_whitney_u(control, treat, ctrl_string, treat_string):
    u_statistic, p = stats.mannwhitneyu(control, treat)
    print(f'U-Statistic between {ctrl_string} and {treat_string}: {u_statistic}')
    print(f'P-Value: {p}')

# Kruskal-Wallis H test
def kruskal_wallis_h(series_arr, group_names):
    h_statistic, p = stats.kruskal(*series_arr)
    print(f'Kruskal-Wallis H test between {group_names}: H={h_statistic:.3f}, p-value={p:.3f}')
