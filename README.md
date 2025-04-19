# qf-chitrea
This is a github place for Alan and Dominic to work on the quantitative finance project in Chinese Treasury (qf-chitrea) for model prediction and training. 

# Current Progress
- Run PCA in R

# Objective
1. Quantitative model for bond
2. Insight into Chinese bond and interest rate and their prediction model in long-term
3. Time-series analysis

# Messages
- Dom: I might just pop up messages in here in case you have anything want to say related to the project/code when I am not online/available. 
- Dom: I may also update on anything when I get to some points or errors so we can discuss it in advance. 
- Dom: You can also leave messages/note/reminders at th bottom so we can get them highlighted. 


# Datafiles
1. 中國_中債國債到期收益率_10年 - Can be used as the target variable in the predictive model, and also for calculating yield changes, yield spreads, duration, etc.
3. 中債國債總淨價指數 - Serves as a benchmark for backtesting and for verifying whether a strategy generates alpha.
4. 回測結果（兩個） - Used to evaluate the effectiveness of the strategy under different market conditions, and to calculate performance metrics such as Sharpe ratio and maximum drawdown.
5. 因子signal匯總 - Used to generate portfolio allocation signals; signals can also be combined into a composite indicator using factor weighting.
6. 因子匯總 - Used for model training, such as PCA dimensionality reduction, factor selection, or scoring model development.
7. 估值因子signal - Helps determine whether interest rate bonds are “undervalued or overvalued”; considered a valuation-type signal.
8. 宏觀因子signal / 宏觀因子篩選 - Provides macro trend signals (e.g., economic expansion is negative for bonds, signal = -1).
9. 宏觀择時因子庫 - Can be used to generate timing signals based on z-scores, percentiles, or other emission mechanisms.

      


# Note
1. Original data have 20,206 na and 2 columns of  constant variance which has been removed
2. There is 2036 na remained amd 143 numerical columns that are not imputed by pmm for unknown reason (Maybe due to incomplete process). (R)
4. Tried running the pmm with 20 interations, but still ended up with 2036 na remained with unknown reason. (R)
5. The remaining na are replaced by 0. (R) 
6. PCA ran successfully (R)
7. Visualised the data in PCA (R)


# Reminder
1. Feel free to change anything if you have more details.
