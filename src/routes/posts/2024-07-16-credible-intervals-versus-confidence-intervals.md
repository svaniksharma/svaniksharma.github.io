---
title: 'Credible Intervals versus Confidence Intervals'
slug: credible-versus-confidence
date: 2024-07-16T07:07:07+01:00
math: 'mathjax'
---

Suppose you have data $X_i | \mu, \sigma^2 \sim N(\mu, \sigma^2)$, for $i = 1, ..., n$, i.e, $n$ normally
distributed data points with mean $\mu$ and variance $\sigma^2$ (or standard deviation $\sigma$). Assume you
don't know $\sigma^2$, which is usually the case anyway. 
You could estimate $\mu \approx \bar{X}$ and call it a day, but you probably want some measure of uncertainty, i.e, an interval estimate. 

*Note*: Throughout this article, I use the convention that uppercase variables are random variables and lowercase variables are fixed (observed)
quantities.

## Confidence Intervals

This is the typical approach taught in any statistics class. However, its interpretation is nuanced. 
We want to estimate $\mu$ and find some interval in which it resides. That is, we want to find:
$P(L \le \mu \le U) = 1 - \alpha$, where $\alpha$ is usually some small quantity (e.g, 0.05). 
In this case, $L$ and $U$ are functions of the sample, which itself can be thought of as a random
vector $\boldsymbol{X} = (X_1, X_2, ..., X_n)$. 
Then, taking $\alpha = 0.05$, we want to calculate $P(L \le \mu \le U) = 0.95$. This is a *95% confidence interval*
for $\mu$. It tells us that the probability the interval $(L, U)$ will cover the true population mean $\mu$ is 95%. 
That is, if we were to take 100 different samples and (somehow) calculate 100 intervals $(L, U)$, 95 of those intervals
would contain the population mean but 5 of them would **not**.
To make this more concrete, let's calculate $L$ and $U$ for the sample we proposed in the beginning. We can use the fact that 
$$\frac{\bar{X} - \mu}{\frac{S}{\sqrt{n}}}$$
is distributed as a $t$ distribution with $n-1$ degrees of freedom (which we denote $t_{n-1}$).
Since the $t$-distribution is unimodal and symmetric, then the 97.5% and 2.5% quantiles are negatives of each other, i.e, 
$t_{n-1, \alpha/2} = -t_{n-1, 1 - \alpha/2}$, and:

$$P(-t_{n-1, 1-\alpha/2} \le \frac{\bar{X} - \mu}{\frac{S}{\sqrt{n}}} \le t_{n-1, 1-\alpha/2}) = 0.95$$

Hence,

$$P(\bar{X} - t_{n-1, 1-\alpha/2} \frac{S}{\sqrt{n}} \le \mu \le \bar{X} + t_{n-1, 1-\alpha/2} \frac{S}{\sqrt{n}}) = 0.95$$

And so the confidence interval is given by $L = \bar{X} - t_{n-1, 1-\alpha/2} \frac{S}{\sqrt{n}}$ and $U = \bar{X} + t_{n-1, 1-\alpha/2} \frac{S}{\sqrt{n}}$.

Notice that $L$ and $U$ are functions of $\bar{X}$ and $S$, the sample mean and sample standard deviation. Hence, given any particular sample, 
**we cannot know whether the interval we compute upon observing a sample does actually contain the true population mean**. 

## Credible Intervals

This is a Bayesian approach. We first set up a joint prior distribution for $\mu$ and $\sigma$. One choice is to use 
a vague, *noninformative prior* distribution: $p(\mu, \sigma^2) \propto \sigma^{-2}$.
The justification for this is hard to explicate succinctly here. See [Gelman](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf), 
particularly Section 2.5. There are many other reasonable choices, but I'll use this one since I've given no particular information
about $\mu$ nor $\sigma$. Also, though the prior is unnormalized (doesn't integrate to 1), it can still be used since the posterior distribution is a probability density.
Then, 
\begin{align}
p(\mu, \sigma^2 | \boldsymbol{X}) 
&\propto p(\boldsymbol{X} | \mu, \sigma^2) p(\mu, \sigma^2) \\\\
&\propto \prod_{i=1}^n p(X_i | \mu, \sigma^2) p(\mu, \sigma^2) \\\\
&\propto \prod_{i=1}^n N(X_i | \mu, \sigma^2) \sigma^{-2} 
\end{align}

If we integrate the last product with respect to $\sigma^2$, then we get the distribution of $p(\mu | \boldsymbol{X})$.
The integration step is slightly involved, but we get $\mu | \boldsymbol{X} \sim t_{n-1}(\bar{X}, \frac{S^2}{n})$. This is 
called a *posterior* distribution.
Now, to compute an interval estimate from $\mu$, we will draw $S$ points from $\mu | \boldsymbol{X}$, then find the 2.5%
and 97.5% sample quantiles and this will be our interval. Generally, we can express this as follows:

$$P(l \le \mu \le u | \boldsymbol{X} = \boldsymbol{x}) = \int_{l(\boldsymbol{x})}^{u(\boldsymbol{x})} p(\mu | \boldsymbol{X} = \boldsymbol{x}) d\mu = 1 - \alpha$$

This looks similar to the confidence interval. But the difference is that we are calculating the probability of $\mu$ being covered
by $(l, u)$ **given** an observed sample $\boldsymbol{x}$. In other words, the probability $\mu$ is contained in $(l, u)$ is 95%. This is 
less confusing than the confidence interval. Notice that there are two parts to this:
1. Take our original $n$ normally-distributed sample points and compute the posterior probability distribution $\mu | \boldsymbol{X} \sim t_{n-1}(\bar{X}, \frac{S^2}{n})$.
2. Draw a random sample of $S$ points from the posterior $\mu | \boldsymbol{X}$, sort them, and use the 2.5% and 97.5% points to compute a posterior probability interval (credible interval).
In other words, the original sample is used to specify the posterior distribution of $\mu$ and then a *new* sample is simulated from the posterior distribution of $\mu$ to compute the 
credible interval. This is why we can say there's a 95% probability $\mu$ lies in $(l, u)$: $l$ and $u$ are the result of a simulation, and they should contain 95% of the $S$ values 
between them.

While the interpretation of credible intervals is easier to understand, there's no guarantee they will produce the same results as a confidence interval. That being said, they usually 
get pretty close. For our purposes, they are almost the same:

```r
pop_mean <- 5.5
pop_sd <- 3
n <- 1000
set.seed(14151)
sample <- rnorm(n, mean = pop_mean, sd = pop_sd)

# Confidence interval

alpha <-  0.05
sample_mean <- mean(sample)
sample_sd <- sd(sample)
crit <- qt(alpha / 2, df = n - 1)
upper <- sample_mean + crit * sample_sd / sqrt(n)
lower <- sample_mean - crit * sample_sd / sqrt(n)
print(paste0(
  upper,
  ", ",
  lower
))

# Credible interval

set.seed(14560)
v = n - 1
simul <- rt(1000, df = v)
simul <- simul / sqrt(v / (v - 2))
simul <- simul * sample_sd / sqrt(n)
simul <- simul + sample_mean
interval <- sort(simul)[c(25, 976)]
print(interval)
```

For the confidence interval, we get $(5.229, 5.588)$. For the credible interval, we get $(5.232, 5.585)$. Also, in this artificial example, both intervals contain the true population mean.

## Conclusion

Ultimately, the distinction between the two is quite important. It's easy to think a 90% confidence interval has a 90% probability of containing the mean, but this is incorrect:

1. A **$p$% confidence interval** for a parameter means that if we were to generate 100 (or 1000 or 10000) samples and compute the interval using some procedure like the one
described above, then $p$ (or $10p$ or $100p$) of those intervals would cover the true population parameter.
2. A **$p$% credible interval** for a parameter means that there is a $p$% chance that the interval computed from the posterior distribution of the parameter will contain 
the true population parameter.

## References

* [Bayesian Data Analysis 3rd Edition, Gelman](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)
* [Introduction to Mathematical Statistics, Hogg](https://minerva.it.manchester.ac.uk/~saralees/statbook2.pdf)
* [Computing a shifted and scaled t-distribution in R](https://stats.stackexchange.com/questions/567944/how-can-i-sample-from-a-shifted-and-scaled-student-t-distribution-with-a-specifi)
