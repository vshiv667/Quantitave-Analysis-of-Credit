## Quantitave-Analysis-of-Credit
When a new applicant applies for credit, as a part of the application, the company collects information which is available in the form of Variables 2 to 21. The company then decides an amount to be credited (the variable CREDIT_EXTENDED.) For these 1000 accounts, we also have information on how profitable each account turned out to be (variable PROFIT). A negative value indicates a net loss. This typically happens when the debtor defaults on his/her payments.
The goal in this case is to investigate how one can use this data to better manage the bank's credit extension program. Specifically, our goal is to develop a Supervised model to classify a new account as “profitable” or “not profitable”. 

## Visualise Metrics:

**ROC**
<p align="center">
  <img  height="500" src="/roc1.PNG">
</p>

**Lift**
<p align="center">
  <img  height="500" src="/lift1.PNG">
</p>

**tree**
<p align="center">
  <img  height="500" src="/tree.PNG">
</p>

**knn**
<p align="center">
  <img  height="500" src="/knn.PNG">
</p>

## Conclusion

- Logistic Regression model accuracy at our optimal cut-off 0.66 is 0.68

- Tree model accuracy with Tree Size = 8 is 0.71

- Knn model accuracy with K=3 is 0.73

Knn model has the highest accuracy. But, if we care more about inference or interpretability, we should use tree model.

But in this business context, we care about whether an account is PROFITABLE or NOT PROFITABLE, to help us determine whether credit should be extended or not. So, we should use a Knn model here

## Libraries:
R: dplyr, tidyr, ggplot2



