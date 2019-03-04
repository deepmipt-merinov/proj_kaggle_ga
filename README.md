# proj_kaggle_ga
88th place solution

**MODEL I**

The 1st model is based on time series forecasting.

1. Firstly, we select all users with only positive revenue history. Here we assume that users who didn't pay will not pay in the future. 
2. To predict the sum of revenue from these users we use common pipeline for time series forecasting: 
  - outlier rejection 
  - exponential smoothing 
  - Box Cox transformation 
  - forecasting algorithm (we used FBProphet algorithm) 
  - inverse transformation
3. The next step is to distribute this value among all users. Here we simply guess that payment distribution in the future will be the same as in the train period.

**MODEL II**

The 2nd model operates on probabilities. For each user we determine a transition from one state to another depending on whether there was a transaction or not.

1. Using parameters `[T0,T1,T2,T3]`, where `T0` and `T1` are start time and end time of train window and `T2` and `T3` are start time and end time of test window, we train a transition matrix `P(i,j)` from state `i` to state `j`, where `I = {0,1,2}` and `J = {1,2}`. We take into account transitions as follows:

  `(0,1) -&gt; [-] trans before T0, [-] trans in [T0,T1] and [-] trans in [T2,T3]`
  `(0,2) -&gt; [-] trans before T0, [-] trans in [T0,T1] and [+] trans in [T2,T3]`
  `(1,1) -&gt; [+] trans before T0, [-] trans in [T0,T1] and [-] trans in [T2,T3]`
  `(1,2) -&gt; [+] trans before T0, [-] trans in [T0,T1] and [+] trans in [T2,T3]`
  `(2,1) -&gt; [+] trans before T0, [+] trans in [T0,T1] and [-] trans in [T2,T3]`
  `(2,2) -&gt; [+] trans before T0, [+] trans in [T0,T1] and [+] trans in [T2,T3]`

where `[-]` means there was no transaction and `[+]` means there was at least one transaction. 
Of course, sum `P(i,j) == 1` over all `i` and `j`. 
At the same time we build matrix `C(i,j)` of average cost per user being in position `(i,j)`.

2. We train model for 2016 year (`T0 = 2016.07.31`, `T3 = 2017.01.31`) and for 2017 year (`T0 = 2017.07.31`, `T3 = 2018.01.31`) independently and forecast target for each user as follows:
  `forecast_2016(user|state=i) = P(i,2) / (P(i,1) + P(i,2)) * C(i,2)`, 
  where matrix `P` and `C` estimated on 2016 year, and state (0, 1 or 2) estimated on `test_v2` history
  `forecast_2017(user|state=i) = P(i,2) / (P(i,1) + P(i,2)) * C(i,2)`, 
  where matrix `P` and `C` estimated on 2017 year, and state (0, 1 or 2) estimated on `test_v2` history

3. We make average predictions with different weights
  `forecast = 0.3 * forecast_2016 + 0.7 * forecast_2017`

Final model makes simple average of Model 1 and Model 2. 
