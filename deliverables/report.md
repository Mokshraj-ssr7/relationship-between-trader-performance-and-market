# Bitcoin Trading Sentiment Analysis: Final Report

**Date:** November 6, 2025  
**Project:** Hyperliquid Trader Performance vs Bitcoin Market Sentiment  
**Analyst:** Data Science Team

---

## Executive Summary

This report presents a comprehensive analysis of the relationship between Bitcoin market sentiment (Fear/Greed Index) and historical trader performance on the Hyperliquid exchange. The goal is to uncover actionable patterns that can inform smarter trading strategies in the Web3 ecosystem.

### Key Findings

> **Note:** This is a template. Actual findings will be populated after running the analysis with real data.

1. **Sentiment-Performance Correlation**
   - [To be filled: Correlation coefficient and significance]
   - [To be filled: Direction of relationship - positive/negative/neutral]

2. **Performance by Sentiment Regime**
   - **Fear Periods:** [Average PnL, win rate, sample size]
   - **Greed Periods:** [Average PnL, win rate, sample size]
   - **Statistical Significance:** [p-value from t-test]

3. **Lag Effects**
   - [To be filled: Optimal lag period (e.g., 1-3 days)]
   - [To be filled: Predictive power of lagged sentiment]

4. **Leverage Behavior**
   - [To be filled: Leverage differences across sentiment regimes]
   - [To be filled: Risk-taking patterns]

5. **Sentiment Transitions**
   - [To be filled: Most profitable/unprofitable transition patterns]
   - [To be filled: Trend-following vs counter-trend opportunities]

---

## Methodology

### Data Sources

1. **Bitcoin Fear/Greed Index**
   - Source: Google Drive (provided link)
   - Columns: Date, Classification
   - Period: [Date range after loading data]
   - Frequency: Daily

2. **Hyperliquid Historical Trader Data**
   - Source: Google Drive (provided link)
   - Columns: account, symbol, execution price, size, side, time, closedPnL, leverage, etc.
   - Period: [Date range after loading data]
   - Records: [Total count after loading]

### Analysis Pipeline

1. **Data Preprocessing**
   - Standardized timestamps and column names
   - Handled missing values and outliers
   - Merged datasets on date keys

2. **Feature Engineering**
   - Created lagged sentiment features (0, 1, 3, 7 days)
   - Aggregated trades to daily account-level metrics
   - Computed rolling statistics and streaks

3. **Statistical Analysis**
   - Correlation analysis (Pearson)
   - Hypothesis testing (t-tests, ANOVA)
   - Time-lag analysis

4. **Visualization**
   - Distribution plots (histograms, box plots, violin plots)
   - Time series overlays (PnL vs sentiment)
   - Correlation heatmaps
   - Interactive Plotly dashboards

---

## Detailed Findings

### 1. Performance Metrics Overview

| Metric | Value |
|--------|-------|
| Total PnL | [To be filled] |
| Overall Win Rate | [To be filled] |
| Average Win | [To be filled] |
| Average Loss | [To be filled] |
| Profit Factor | [To be filled] |
| Number of Traders | [To be filled] |
| Trading Days Analyzed | [To be filled] |

### 2. Sentiment Distribution

[Insert sentiment frequency table and pie chart findings]

### 3. Performance by Sentiment

| Sentiment | Count | Mean PnL | Median PnL | Win Rate | Std Dev |
|-----------|-------|----------|------------|----------|---------|
| Extreme Fear | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Fear | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Neutral | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Greed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Extreme Greed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Key Insights:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

### 4. Correlation Analysis

**Sentiment vs PnL:** [Correlation coefficient, p-value]

**Sentiment vs Leverage:** [Correlation coefficient, p-value]

**Sentiment vs Trade Volume:** [Correlation coefficient, p-value]

[Insert correlation heatmap interpretation]

### 5. Lag Analysis Results

| Lag (Days) | Correlation | P-Value | Significance |
|------------|-------------|---------|--------------|
| 0 (Same day) | [TBD] | [TBD] | [TBD] |
| 1 | [TBD] | [TBD] | [TBD] |
| 3 | [TBD] | [TBD] | [TBD] |
| 7 | [TBD] | [TBD] | [TBD] |

**Interpretation:**
[Describe which lag periods show strongest predictive power]

### 6. Trader Behavior Patterns

#### Leverage Usage
- **Fear Periods:** [Average leverage]
- **Greed Periods:** [Average leverage]
- **Interpretation:** [Risk appetite changes]

#### Trading Volume
- **Fear Periods:** [Average volume]
- **Greed Periods:** [Average volume]
- **Interpretation:** [Activity level changes]

### 7. Statistical Validation

#### Hypothesis Test: Fear vs Greed

**Null Hypothesis (H₀):** No difference in PnL between Fear and Greed periods  
**Alternative Hypothesis (H₁):** Significant difference exists

**T-Test Results:**
- T-statistic: [TBD]
- P-value: [TBD]
- Conclusion: [Reject/Fail to reject H₀]

**Mann-Whitney U Test:**
- U-statistic: [TBD]
- P-value: [TBD]
- Conclusion: [Reject/Fail to reject H₀]

#### ANOVA: All Sentiment Categories

**F-statistic:** [TBD]  
**P-value:** [TBD]  
**Conclusion:** [TBD]

---

## Actionable Insights

### For Traders

1. **Counter-Trend Strategy**
   - [Recommendation based on fear/greed reversal patterns]
   - [Optimal entry/exit conditions]

2. **Trend-Following Strategy**
   - [Recommendation based on sentiment momentum]
   - [Optimal holding periods]

3. **Risk Management**
   - [Leverage recommendations by sentiment]
   - [Position sizing guidelines]

### For Trading Teams

1. **Signal Integration**
   - Incorporate sentiment as a feature in trading algorithms
   - Weight: [TBD based on correlation strength]

2. **Monitoring Dashboard**
   - Real-time sentiment tracking
   - Alert system for sentiment regime changes

3. **Backtesting Priorities**
   - Test strategies identified in transitions analysis
   - Validate lag-based entry signals

---

## Limitations & Caveats

1. **Data Limitations**
   - Historical analysis only - not predictive guarantee
   - Potential survivorship bias in trader data
   - Limited time period [specify range]

2. **Market Context**
   - Crypto markets are highly volatile and non-stationary
   - Sentiment index may have measurement limitations
   - Regulatory and macro factors not accounted for

3. **Causation vs Correlation**
   - Observed correlations do not imply causation
   - Third variables may influence both sentiment and performance

---

## Next Steps & Recommendations

### Immediate Actions (1-2 weeks)

1. **Deploy Monitoring Dashboard**
   - Build Streamlit/Dash app for real-time sentiment tracking
   - Integrate with existing trading infrastructure

2. **Strategy Prototyping**
   - Develop rule-based strategies from identified patterns
   - Paper trade for 2-4 weeks

### Short-Term (1-3 months)

3. **Predictive Modeling**
   - Train ML models (XGBoost, LightGBM) to predict next-day PnL
   - Feature importance analysis
   - Cross-validation on time-series splits

4. **Trader Segmentation**
   - Cluster analysis to identify trader archetypes
   - Personalized recommendations per cluster

5. **Expanded Features**
   - Integrate on-chain metrics (wallet flows, exchange reserves)
   - Social media sentiment (Twitter/X, Reddit)
   - Macroeconomic indicators (Fed rates, inflation)

### Long-Term (3-6 months)

6. **Production Deployment**
   - Real-time API integration
   - Automated signal generation
   - Performance monitoring and A/B testing

7. **Research Extensions**
   - Multi-asset analysis (ETH, SOL, etc.)
   - Intraday sentiment variations
   - Sentiment momentum strategies

---

## Technical Appendix

### Tools & Technologies

- **Language:** Python 3.10+
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Statistics:** scipy, statsmodels
- **Machine Learning:** scikit-learn (for future modeling)
- **Environment:** Jupyter Lab, VS Code

### Reproducibility

All analysis code is available in:
- `notebooks/eda.ipynb` - Main exploratory analysis
- `src/` - Reusable modules (data_loader, preprocessing, analysis, features)
- `tests/` - Unit tests for validation

To reproduce:
```bash
pip install -r requirements.txt
jupyter notebook notebooks/eda.ipynb
```

### Data Dictionary

See `README.md` for complete column descriptions.

---

## Contact & Support

For questions or to discuss findings:
- **Project Lead:** [Name]
- **Data Science Team:** [Email]
- **GitHub Repository:** [Link if applicable]

---

**Document Version:** 1.0  
**Last Updated:** November 6, 2025  
**Status:** Draft - Pending data analysis completion
