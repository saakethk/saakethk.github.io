---
title: "Implementing Fundamental Stock Factors for Improved Stock Forecasting"
date: 2025-05-05 00:00:00 -0500
categories: [Research]
tags: [Finance, Research]
---

This post was to commemorate my first official research paper as part of my high school's capstone research program. I have pasted the content of the paper below:

**Background:**

One of the earliest forms of stock was issued by the Dutch East India Company in the 1600s (Mohan, 2019). These initial stock exchanges were primitive and lacked many luxuries like real-time price tracking and the regulation of stocks via government institutions like the Securities Exchange Commision. As time progressed, stock exchanges started to modernize with the London stock exchange and New York stock exchange becoming dominant trading markets in the 19th century (Mohan, 2019). Despite these developments, stock markets were still limited due to the lack of infrastructure they enjoy today. However, they underwent drastic improvements in the late 20th century which greatly increased accessibility of markets to the general public. Technological improvements such as the development of the smartphone and creation of cloud architecture coincided with the development of the stock market and played a key role in it. The integration of markets with these modern technologies allowed for advancements like stock orders and near-instant execution times.

Despite this development of stock markets over time, stock market investing still has innate risks and can cause people to possibly lose millions of dollars if they sell in times of recession or correction. This innate risk can be mitigated via algorithmic trading which aims to buy and sell stocks according to a predetermined set of criteria in order to maximize profit and minimize loss (Afua et al., 2024). Algorithmic trading first started in the 1970s and 1980s but didn’t fully mature until the 2000s (Afua et al., 2024). Initial algorithms focused on leveraging market inefficiencies in order to make a profit but as the accessibility of algorithmic trading increased, novel technologies like artificial intelligence were integrated into algorithms to help deal with the dynamic nature of the stock market. 

Algorithms which implement artificial intelligence can vary in complexity from relatively easy to understand machine learning algorithms, like polynomial regression, to complex neural networks which take a vast array of data and resources to train and deploy. Most conventional algorithms in the past focused on forecasting future stock prices via past stock prices. As models have improved, creating more accurate models is a priority in order for institutions to maintain a competitive edge. According to Sable et al. (2023), models such as Support Vector Machine’s (SVMs) and Long Short-Term Models (LSTMs), variations of conventional neural networks, can potentially analyze a large number of inputs and find associations between them in a way that conventional algorithms would struggle to accomplish. The benefits of these models mean they can potentially create solid predictive models which don’t merely make money by exploiting minute price fluctuations and actually predict prices based on substantive world and fiscal events. 

Despite the research being done into LSTMs and SVMs, there is a large amount of skepticism regarding the feasibility of substantive price forecasting in general. The paper  **“**Origins of Stock Market Fluctuations clearly investigates this and concludes that the two main factors driving price are discount rate shock and cash-flow innovation. These two factors are largely independent of the actual performance of the stock’s associated company and instead a result of human psychology. Because of this human intervention in stocks, many professional stock market investors such as Warren Buffet focus on the intrinsic value of the stock opposed to trying to predict the price on a set time basis. Though this methodology of investing has proven successful to Buffet, his decisions are largely based on instinct and personal memories making this methodology hard to replicate. Overall, no study thus far has definitively proved that the stock market can be accurately predicted and, likewise, no one has proved definitively that the stock market cannot be accurately predicted.

Though the feasibility of models has been called into question, there are some case studies of companies which utilize complex algorithmic trading algorithms currently to help drive their decisions. Afua et al. (2024) actually reveals that Renaissance Technologies, Virtu Financial, and Citadel Securities have all shown great success when using these models. However, their success has been limited in times of financial crisis. Many of their models allow for a 2-3% improvement in returns but don’t significantly decrease risk of loss during a financial crisis.

Mihova et al. (2024) details how statistical methods like Autoregressive Integrated Moving Average (ARIMA) and modified Ordinary Differential Equations (ODE) were actually able to reduce loss when back tested under conditions of financial crisis. However, the study also accentuated the fact that these models benefited from constant validation and training due to their mostly stationary nature. Models such as LSTMs show promising results which can exceed the performance of these existing statistical models (Moghar & Hamiche, 2020). However, the performance of these LSTM models is very reliant on the input factors. The number of input factors and the types of input factors possible for these models range in the thousands. Feeding every possible factor into the model may seem to solve the problem in theory but is not actually possible in the real world due to computational limits associated with current technology. Because of this, the input factors need to be selectively implemented so as to retain the practicality of LSTMs while improving their performance. The lack of depth in current research clearly warrants a need for this paper’s investigation into whether the effective implementation of LSTMs can improve stock forecasting performance.

This paper's goal will be to focus on creating a program capable of retrieving a mixture of fundamental and quantitative stock factors, training the LSTM on the most relevant data, and then evaluating its performance relative to existing statistical models. The overall objective will be to discern whether LSTMs can truly yield performance better than existing models.

**Literature Review:**

Before investigating the performance of LSTMs, it is important to understand where their proposed performance increase originates from. LSTMs are a subset of neural networks called recurrent neural networks (RNNs). LSTM networks were proposed by Hochreiter and Schmidhuber (1997) in order to solve a problem associated with RNNs known as the vanishing gradient problem. The way the LSTM network functions is that it can bridge time intervals in excess of 1000 steps. Essentially, a normal neural network will cause the weights closest to the inputs to be changed less because the way weights are updated is from the output nodes to the input nodes. Hence, if you have a very complex model, the correction which occurs after every training interval will have minimal impact on the nodes closest to the inputs. The LSTM solves this problem by having three gates: the input gate, the forget gate, and the output gate. These gates allow for the model to encode inputs with certain importance levels. LSTMs are able to classify what time steps are relevant and which are not making them far better than recurrent neural networks which give equal emphasis to all inputs (Hochreiter & Schmidhuber, 1997). 

Despite LSTMs theoretically having better performance, this has not been definitively proceeded given the small amount of research in the field. Most research currently published centers around the efficacy of statistical based methods like Autoregressive Integrated Moving Average (ARIMA), modified Ordered Differential Equations (ODEs), and Prophet. 

The ARIMA model consists of three major components: an autoregressive (AR) component, a differencing for stationarity (I) component, and a moving average (MA) component (Wagner & Cleland, 2023). The autoregressive component references the fact that the model attempts to predict future values based on past values. The integrated component references the need for stationarity, a term describing no change over time in the distribution. Lastly the moving average component indicates that the prediction takes into account previous errors and current errors to calculate the error term. These components are then combined to create an overall forecast (Wagner & Cleland, 2023). 

The modified ODEs work by attempting to fit a polynomial of a certain order to data in order to forecast future values (Xue & Lai, 2018). They have been shown to have benefits such as reduced need for data and are able to be modified easily (Xue & Lai, 2018). ODEs have a vast amount of research which indicates the idea is still being investigated opposed to ARIMA which has stagnated in its performance potential.

Prophet is a seasonality based prediction model which was made by Meta. According to Taylor & Letham (2017), the Prophet model focuses on seasonality and non-linear trends while the ARIMA model eliminates seasonality and searches for general trends. Though the Prophet model does well with predictable stocks that are primarily influenced by seasonality, it lacks the ability to predict prices of more volatile stocks since this wasn’t its intended purpose.

Both models predict future prices based purely on past prices, making the models unable to take into account fundamental factors, factors which don’t directly originate from the observed stock. According to Khanpuri et al. (2024), LSTMs which take into account fundamental stock factors are able to better capture short-term trends and seasonality. Because this research is quite new and not reproducible, the results of this study cannot be conclusive, warranting the need for this research.

Though research regarding LSTMs has shown considerable potential, their performance is largely dependent on the stocks tested and the factors implemented into the model. For instance, a detailed metaanalysis of over 80 papers clearly shows that over 80% of papers only use historical stock data as the primary factor. Only 4% of papers use other fundamental stock factors in their model (Verma et al., 2023). This distribution of factors clearly shows that very few papers take into account fundamental stock factors meaning their performance is not actually representative of the LSTMs actual potential performance. Additionally, the scope of prediction differs between papers. 58% of papers predict stock index price while only 32% take into account individual forecasts (Verma et al., 2023). Because of this additional variability in the scope of inference, the true performance of LSTMs could also not be inferred. One key metric in the metaanalysis was that 46% of the papers in this field utilized deep learning.

Given the limited research in the specific field this study is investigating, the factors that are investigated can vary. Some factors that can be implemented include: industry gdp, national gdp, stock price, industry index price, news sentiment, social sentiment, and company financials (Verma et al., 2023).

These factors can be broken down into two categories: Fundamental analysis and technical analysis. These categories can then be broken down further as fundamental analysis consists of economic analysis, industry analysis, and company analysis. Technical Analysis consists of charting and quantitative analysis (Verma et al., 2023)..

Technical analysis usually consists of the stock price and derivatives of the stock price like RSI (Relative Strength Index). These indicators are what are used primarily now. There is undercoverage in the fundamental analysis indicating that factors like industry gdp, national gdp, industry index price, news sentiment, and social sentiment need to be implemented.

Gathering data associated with these factors is straightforward for some factors like industry gdp and national gdp which can be easily accessed via government APIs. However, factors like social sentiment are harder to get data from as they greatly vary. According to Mao et al. (2024), the types of sentiment that can be found are usually categorized into positive, neutral and negative, and can be further divided into surprise, trust, anticipation, anger, fear, sadness, disgust, joy, ect. Because of this large amount of variability in the range of sentiment that can be analyzed, certain classification models can label text as merely positive or negative while others can go more specific. LLMs have been shown to provide the most accurate sentiment when evaluated according to a premade dataset but have high associated costs and a lack of reliability (Mao et al., 2024). There are tokenized language libraries which can provide sentiment scores based on the words in the sentence but they are unable to account for the order in which words are organized (Mao et al. 2024). There are hybrid models which use less resource intensive machine learning based methods to classify sentences into categories. These models can be run locally allowing for low costs and take into account the order of the words and context of the sentence.

Overall, there is an abundance of research into the field of stock forecasting, but there is a clear gap present in the implementation of fundamental stock factors in an LSTM. The implementation of these factors shows great promise as they allow for the model to find more complex relationships between input variables.

**Methods:**

In order to effectively carry out my research, the project needed an experimental methodology which would allow for conclusions to be drawn given a small sample size. According to Leedy & Ormrod (2015), a within-subjects design yields high statistical power, reduced variability, and smaller sample sizes. The actual methodology behind a within-subjects design is when all treatments are applied to all possible groups. High statistical power indicates a low likelihood of a Type II error which means a lower likelihood of the null hypothesis being accepted given that the null hypothesis is actually true. Reduced variability indicates that the results are more conclusive given comparison between groups. A smaller sample size is possible because all treatments are applied to everyone allowing for a matched pair design.

Though the within-subjects design is good for comparing the already created models, the design of experiments methodology is necessary for optimizing the factors fed to the model. According to Barad (2014), design of experiments is where numerous factors and levels are tested in an order to minimize the number of comparisons while finding the widest array of responses. In this project, the design of experiments methodology was specifically where the LSTM model was tested with each factor being incrementally added and then evaluated. This allows us to view the effect of each factor. Epochs were also tested in this manner where the number of epochs with which the model was tested was evaluated to find the average epoch which returned the lowest loss. Sequence length, the number of days of data to feed the model for inference, was also evaluated in a similar manner. After these three parameters of the model were tested with the design of experiments methodology, the optimized model was evaluated using the matched pair design.

Due to computational limitations, the initial subset of factors evaluated by the model was selected by the researcher and included volume weighted price, related value index, insider trades, aggregate news sentiment, personal consumption expenditure, and industry gdp. The data was gathered from the following services due to low cost and reliability of data: Alpaca, Finnhub, and Bureau of Economic Analysis. Once this data was gathered, it was processed via a python script. The python script utilized libraries like Pandas which allow for efficient handling of large datasets. The Pandas library allowed for the script to filter and sort the dataset and fill in missing values.

After processing all the values in the dataset, the filtered dataset was fed into 4 variants of the model. These 4 variants included the ARIMA, Prophet, LSTM wo/Fundamental Factors, and LSTM w/Fundamental Factors. This design setup was based on an already established experiment by Pilla and Mekonen (2025) which compared the performance of an LSTM to an ARIMA model. The paper concluded that the LSTM outperformed the ARIMA by a significant margin. This is why the Prophet model and LSTM w/Fundamental Factors were incorporated in order to provide additional reference and to fill the gap of this project. Because a within-subjects design has the groups, which are in this case the individual stocks evaluated, have all treatments applied to each of them, each stock can be compared through a matched pair design as well as through two sample designs. This versatility allows for further analysis during the data analysis phase of my project.

The program was then allowed to run on a small subset of stock from the NASDAQ 100. These stocks were filtered based on the data available. The ending sample was 32 individual stocks in the testing group. These 32 chosen stocks then went through the program which completed the following in order: gather data from all sources, aggregate data using pandas, clean aggregated data for parsing, initialize and train model, and evaluate trained model, record mean squared error statistic.

The primary metric recorded throughout this project was mean squared error (MSE). The reason for this was that Silva et al. (2022) clearly detailed how RMSE (Root Mean Squared Error), Mean Absolute Error (MAE), and MSE are all generally similar in their ability to convey the performance of models. Many of the error metrics generally capture the difference between predicted and actual.

After the MSE of all the models is recorded, the actual comparison of data required a blocked design where the blocks were the models. Every block was averaged and then the averages were compared via repeated two sample t-tests.

**Results:**

During optimization, the researcher found crucial findings related to three essential influencing factors: number/ types of inputs, number of epochs of training, and sequence length. The researcher found all of these influencing factors to be crucial to the efficiency of the model.

To begin with, the number/types of inputs to the model were initially six inputs which covered volume weighted price, insider trades, personal consumption expenditure, industry gdp, and related value index. The testing found that the influence of each factor on the mean squared error was very volatile regardless of the number of factors. In Table A1 (See Appendix A), factor 2 which was representative of insider trades has an aggregate MSE increase of 90.47% and a standard deviation of 170.11.This increase in MSE was significantly higher than any of the other factors. This makes sense given that the data was very limited and extrapolated off of small sample sizes according to Finnhub. When comparing the increase in MSE to the other factors via a statistical test, there is a statistically significant p-value of 0.03 which indicated that this was unlikely to happened due to chance and should be removed in order to give the model the best chance.

To add on, the number of epochs of training was recorded (See Table A2) for a small sample of the 32 stocks studied. The mean number of epochs for convergence of the model which only took in historical price was found to converge at a mean epoch duration of 19 epochs. The mean number of epochs for convergence of the model which took in fundamental stock factors was 22 epochs. The standard deviation of the model that accounted for all factors was less than the standard deviation for the price only model indicating more reliability in performance.

Thirdly, the sequence length was recorded for the small sample of stocks (See Table A3) in order to find the optimal sequence. Sequence lengths of 3 through 10 were tested and a sequence length of 9 yielded the largest statistically significant decrease in MSE. At a sequence length of 9, there was a reduction by 27.723 in MSE at a p-value of 0.0585. 

The optimization results indicated optimal factors excluded insider trades, the optimal epochs for training is 19/22, and the optimized sequence length is 9. The LSTM model was then run with these optimized parameters for each stock.

In Table B2 (See Appendix B), it is clear that the aggregate score of the LSTM with only historical stock price as a factor was marginally lower than the model which accounted for all factors. This difference in MSE of 0.0000519 was not statistically significant. The aggregate MSE of the Prophet and ARIMA model were .9225 and .3352 respectively. These were far higher than the MSE of both LSTM models which was around \~.0385.

Despite the relative indifference in performance, when the results are observed on a stock-by-stock basis, it is clear that stocks such as Apple (AAPL), Adobe (ADBE), and Analog Devices Inc. (ADI) actually had better stock forecasting performance with the modified LSTM than the one factor LSTM (See Table B1).

**Discussion:**

The results of the experiment indicated that a forecasting model which takes into account fundamental external factors did not significantly perform better than a price only model. However, there were specific type individual stocks in which the modified model performed better indicating a potential for further research.

The fact that insider trades worsened model performance by a significant margin is likely due to the small sample size and the lack of reliable data. The researcher noticed that the insider data was missing several months of data for each stock. The other factors also had high volatility in terms of its influence on mean squared error. This indicates that the inputs greatly vary in their influence from stock to stock.

During epoch testing, the model which accounted for fundamental external factors tended to converge later than the model using only price data. This makes sense given the fact that more factors should increase the number of epochs needed for training. This testing also indicated that AAPL converged far later than the other stocks in the sample in both model variants.

During sequence length testing, the standard deviation decreasing from a sequence length of 9 to a sequence length of 10 indicates more steady performance at a sequence length of 10 but better mean change in mean squared error at sequence length of 9. The standard deviation for sequence length was also much larger for all sequence lengths tested with the exception of a sequence length of 10 which seemed to be an outlier to the group. A sequence length of 10 indicated a mean squared error reduction of \~5 percent and had little variance with standard deviation of the sample at \~1.8 (See Table A3). However, this stability in performance is not as significant to performance as a sequence length of 9 which reduces mean squared error by \~27.7 percent (See Table A3).

Overall, the optimization of the LSTM model and its comparison to statistical and inference based methods like ARIMA and Prophet yielded statistically significant improvements. However, the difference in performance between the LSTM with only a price input and the LSTM with fundamental factors taken into account was not statistically significant despite there being a noticeable difference in their respective mean squared error.

**Future Research:**

The results of this study clearly showed a slight difference between the performance of a Long Short-Term Model (LSTM) with only historical stock data as an input opposed to the performance of a LSTM with fundamental data as an input as well. If this project were to be repeated, the researcher would attempt to increase the complexity of the LSTM. This increased complexity would require increased computing power which the researcher currently lacks.

Additionally, the researcher encountered a unique type of model known as a Kolmogorov Arnold Network (KAN) during his preliminary research. This model would allow for more interpretability of the model opposed to current LSTMs which have hidden layers that lack the interpretability of KANs. Given the novel nature of KANs they could potentially present a better alternative to LSTMs. The project could also be improved by integrating the initial stages of a KAN with the final stages of a LSTM.

KANs are based on the Kolmogorov-Arnold representation theorem  which states that every multivariable continuous function can be represented as a superposition of continuous single-variable functions (Xu et al., 2024). The study by Xu et al. (2024) explains how KANs differ from Multi-Layer Perceptrons (MLPs) in terms of their weights. MLPs tend to have fixed activation functions like the sigmoid function. KANs have learnable activation functions on the edge of nodes and they can be thought of as variable weights. Essentially, every weight parameter is a variable function. Because each weight is a variable function, you can view the impact of a node on the next node at different values. The paper specifies that KANs are characterized by mathematical, accurate, and interpretable.

There are variants of KANs such as T-KAN and MT-KAN. T-KAN is designed to focus on univariate time series and a term known as concept drift while MT-KAN focuses on improving predictive performance through multivariate functions (Xu et al., 2024). The results of the study by Xu et al. (2024) show that MLPs perform the best with 21221 parameters but that MT-KANs are nearly as performant with 2132 parameters. This shows that MT-KANs are clearly very efficient and good at predicting stock prices as the predicted values align closely with the price of six NASDAQ stocks while keeping the number of parameters low.

Overall, there is clearly potential for more research in this field as the model architecture, model optimization, and model factors can all be researched further so as to truly determine whether stock forecasting in the near term with a reasonable accuracy is possible under any financial conditions.

**References**

Afua, W., Adeola, O.A., Bello, G., Tubokirifuruar, S., Olubusola 

O., & Titilola F. (2024). Algorithmic trading and AI: A review of strategies and market impact. _World Journal of Advanced Engineering Technology and Sciences_, _11_(1), 258–267. [**https://doi.org/10.30574/wjaets.2024.11.1.0054**](https://doi.org/10.30574/wjaets.2024.11.1.0054) ****

Barad, M. (2014). Design of experiments (DOE): A valuable multi-purpose methodology. 

_Applied Mathematics_, _05_(14), 2120–2129. [**https://doi.org/10.4236/am.2014.514206**](https://doi.org/10.4236/am.2014.514206) 

Greenwald, D. L., Lettau, M., & Ludvigson, S. C. (2014, January 1). _Origins of Stock_ 

_Market Fluctuations._ National Bureau of Economic Research. [**https://www.nber.org/papers/w19818**](https://www.nber.org/papers/w19818) ****

Hochreiter, S., & Schmidhuber, J. (1997). _Long Short-Term Memory._ ResearchGate. 

[**https://www.researchgate.net/publication/13853244\_Long\_Short-term\_Memory**](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory) ****

Khanpuri, A., Darapaneni, N., & Paduri, A. (2024). Utilizing Fundamental Analysis to 

Predict Stock Prices. EAI Endorsed Transactions on Artificial Intelligence and Robotics, 3. [**https://doi.org/10.4108/airo.5140**](https://doi.org/10.4108/airo.5140) ****

Leedy, P. D., & Ormrod, J. E. (2015). Practical research: Planning and 

design (11th ed.). Pearson Education Inc.

Mao, Y., Liu, Q., & Zhang, Y. (2024). Sentiment analysis methods, applications, and 

challenges: A systematic literature review. _Journal of King Saud University. Computer and Information Sciences/Maǧalaẗ Ǧamʼaẗ Al-Malīk Saud : Ùlm Al-Ḥasib Wa Al-Maʼlumat_, _36_(4), 102048–102048. ****[**https://doi.org/10.1016/j.jksuci.2024.102048**](https://doi.org/10.1016/j.jksuci.2024.102048) ****

Mihova, V., Georgiev, I., Raeva, E., Georgiev, S., & Pavlov, V. (2024). Validation of Stock 

Price Prediction Models in the Conditions of Financial Crisis. _Mathematics, 12_(1), 33. [**https://doi.org/10.3390/math12010033**](https://doi.org/10.3390/math12010033) ****

Moghar, A., & Hamiche, M. (2020). Stock Market Prediction Using LSTM Recurrent 

Neural Network. _Procedia Computer Science, 170_(1), 1168–1173. [**https://doi.org/10.1016/j.procs.2020.03.049**](https://doi.org/10.1016/j.procs.2020.03.049) ****

Mohan, R. (2019, April). _(PDF) Stock Markets: An Overview and A Literature Review_. 

ResearchGate. [**https://www.researchgate.net/publication/342991622\_Stock**](https://www.researchgate.net/publication/342991622_Stock_Markets_An_Overview_and_A_Literature_Review)

[**\_Markets\_An\_Overview\_and\_A\_Literature\_Review**](https://www.researchgate.net/publication/342991622_Stock_Markets_An_Overview_and_A_Literature_Review) ****

Pilla, P., & Mekonen, R. (2025). Forecasting S\&P 500 Using LSTM Models. _ArXiv (Cornell_ 

_University)_. **https\://doi.org/10.5281/zenodo.14759118**

Sable, R., Goel, S., & Chatterjee, P. (2023). Techniques for Stock Market Prediction: A 

Review. _International Journal on Recent and Innovation Trends in Computing and Communication_, _11_(5s), 381–402. ****[**https://doi.org/10.17762/ijritcc.v11i5s.7056**](https://doi.org/10.17762/ijritcc.v11i5s.7056) ****

Silva, Teresinha, M., Sérgio, M., & Alvarenga, A. (2022). Performance evaluation of LSTM 

neural networks for consumption prediction. _E-Prime_, _2_, 100030–100030. [**https://doi.org/10.1016/j.prime.2022.100030**](https://doi.org/10.1016/j.prime.2022.100030) ****

Taylor, S. J., & Letham, B. (2017). Forecasting at scale. _Forecasting at Scale_. 

[**https://doi.org/10.7287/peerj.preprints.3190v2**](https://doi.org/10.7287/peerj.preprints.3190v2)

Verma, S., Sahu, S. P., & Sahu, T. P. (2023). Stock Market Forecasting with Different Input Indicators using Machine Learning and Deep Learning Techniques: A Review. _Engineering Letters_, _31_(1), 213–229. [**https://www.researchgate.net/publication/369971166\_Stock\_Market\_Forecasting\_with\_Different\_Input\_Indicators\_using\_Machine\_Learning\_and\_Deep\_Learning\_Techniques\_A\_Review**](https://www.researchgate.net/publication/369971166_Stock_Market_Forecasting_with_Different_Input_Indicators_using_Machine_Learning_and_Deep_Learning_Techniques_A_Review)    

Wagner, B., & Cleland, K. (2023). Using autoregressive integrated moving average models 

for time series analysis of observational data. _BMJ_, _383_, p2739. [**https://doi.org/10.1136/bmj.p2739**](https://doi.org/10.1136/bmj.p2739) ****

Xu, K., Chen, L., & Wang, S. (2024, June 4). _Kolmogorov-Arnold Networks for Time Series:_ 

_Bridging Predictive Power and Interpretability_. ArXiv.org. [**https://doi.org/10.48550/arXiv.2406.02496**](https://doi.org/10.48550/arXiv.2406.02496) ****

Xue, M., & Lai, C.-H. (2018). From time series analysis to a modified ordinary differential 

equation. _Journal of Algorithms & Computational Technology_, _12_(2), 85–90. [**https://doi.org/10.1177/1748301817751480**](https://doi.org/10.1177/1748301817751480) ****

********

**Appendix A**

Model Optimization

**Table A1**

_Influence of implemented factors on LSTM’s mean squared error (MSE)_

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcDqXqKRGy6iqh4Y-65ESsbpqCqNn5qaxW0FakzRrP-Z9ekkxNLCrRDxQuEsGMdfd0cT8aK0k2Xp59hIdOexNwtV8zsrll3dSg5Ipf3aOcvq85k_fT5_OFQlbxhUaGT0AbjT77G2Q?key=vlUKlWhgXK-AxquuT7sgiP0b)

**Table A2**

_Occurrence of minimum loss epochs out of 50 epochs_

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdhz-egR3NfibmeEzt1Z88-1g4XmmaMiW8pSxsOB2N0qkFRMs-ggJQc2Q8zTYnsXJf1C72XX_P8fixPg14AOKD3kVXZpvCHYvYNxaCaFEcgZ2AlobWdDeRmV3P7I22Ctsry1LY6Lg?key=vlUKlWhgXK-AxquuT7sgiP0b)

**Table A3**

_Influence of sequence length on LSTM’s mean squared error (MSE)_

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeEaCjBfWvHeEnOxT0-9Vz7W6l10H_uN7BYtsBpRZ79rmfMFKhBlETboUHBWFJo7B5MTl4fgZtdC8jPQQDsWOQ0FC9OqimCmqDSyymjsOBtiruVDSns5EIdGGn6Ab5jaNHJxmagXQ?key=vlUKlWhgXK-AxquuT7sgiP0b)

********

**Appendix B**

Optimized Model Results

**Table B1**

_Mean squared error (MSE) of all stock forecasting models_

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcUZcEWrw8PObtZTH15jBvtFN66NwarPQlUKLCmSrZVRYKqNtl_Sj1sNWefrU7zh7GILRwjG7oijoF6K5BvBJgIwjrZXEUcTYY7AqB-7dGStR5I-vaew98NJgdBbUmYW_-7HavDbg?key=vlUKlWhgXK-AxquuT7sgiP0b)

**Table B2**

_Aggregate mean squared error (MSE) graph comparison_

__![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf2wbkNH3qGrDFmsUuWO3F7lFmBrf5-UzDew9X9OIRQybQdUZDAQrnkrkXSMM7vgR322mZujLwQRCCEXjJl_b_j6AfEOwO9xUzSKBgvIeyO3FleKwdX6Oxz403laAfTW0_bAiaOzA?key=vlUKlWhgXK-AxquuT7sgiP0b "Chart")__
