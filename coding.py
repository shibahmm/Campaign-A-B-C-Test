import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

#import csv dataset
df = pd.read_csv("E:/Side Projects/Marketing-Campaign-A-B-Test/WA_Marketing-Campaign.csv")
#print("First 5 records:", df.head())

'''
Null Hypothesis (Ho): There is no difference in mean sales between the 3 promotions; no promotion has a statistically significant impact on sales compared to others.
aka μ₁ = μ₂ = μ₃
Alternative Hypothesis (Ha): At least one promotion has a different mean sales.
aka μᵢ ≠ μⱼ for some i, j
'''

#aggregate sales by LocationID (average of 4 weeks)
aggregated = df.groupby(['LocationID', 'Promotion', 'MarketSize', 'AgeOfStore'])['SalesInThousands'].mean().reset_index()

#split into promotion groups
promo_1 = aggregated[aggregated['Promotion'] == 1]['SalesInThousands']
promo_2 = aggregated[aggregated['Promotion'] == 2]['SalesInThousands']
promo_3 = aggregated[aggregated['Promotion'] == 3]['SalesInThousands']

#descriptive statistics
print("Promotion 1 Mean Sales:", promo_1.mean())
print("Promotion 2 Mean Sales:", promo_2.mean())
print("Promotion 3 Mean Sales:", promo_3.mean())

#check ANOVA assumptions
#1. Normality (Shapiro-Wilk test)
_, p1 = stats.shapiro(promo_1)
_, p2 = stats.shapiro(promo_2)
_, p3 = stats.shapiro(promo_3)
print(f"\nNormality p-values: Promotion 1 ({p1:.3f}), Promotion 2 ({p2:.3f}), Promotion 3 ({p3:.3f})")

#2. Homogeneity of variances (Levene's test)
_, p_levene = stats.levene(promo_1, promo_2, promo_3)
print(f"\nLevene's Test p-value: {p_levene:.3f}")

#run ANOVA or Kruskal-Wallis
if p_levene > 0.05:
    # ANOVA
    _, p_anova = stats.f_oneway(promo_1, promo_2, promo_3)
    print(f"\nANOVA p-value: {p_anova:.4f}")
else:
    # Kruskal-Wallis (non-parametric)
    _, p_kruskal = stats.kruskal(promo_1, promo_2, promo_3)
    print(f"\nKruskal-Wallis p-value: {p_kruskal:.4f}")

#post-hoc analysis if overall test is significant
if p_anova < 0.05 or p_kruskal < 0.05:
    tukey = pairwise_tukeyhsd(
        endog=aggregated['SalesInThousands'],
        groups=aggregated['Promotion'],
        alpha=0.05
    )
    print("\nTukey HSD Results:")
    print(tukey.summary())

#visualize results
plt.figure(figsize=(10, 6))
sns.boxplot(x='Promotion', y='SalesInThousands', data=aggregated)
plt.title("Sales Distribution by Promotion")
plt.show()