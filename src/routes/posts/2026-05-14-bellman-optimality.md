---
title: 'Bellman Optimality'
date: 2026-05-14T07:07:07+01:00
slug: bellman_optimality
math: 'mathjax'
---

I just wanted to write some notes down on Bellman optimality equations in detail, since some of the sources I have read don't write the details out.


# Bellman Optimality Equations

For a policy $\pi$, we will determine:

1.  Computing $v_\pi$ in terms of $q_\pi$.
2.  Computing $q_\pi$ in terms of $v_\pi$.
3.  Computing $v_\pi$ in terms of $v_\pi$ (Bellman optimality)
4.  Computing $q_\pi$ in terms of $q_\pi$ (Bellman optimality)


## Iterated Expectation

The subsequent derivations rely on iterated expectations. In particular, given random variables $X$, $Y$, $Z$, we need to prove:
$$

\begin{align*}
\mathbb{E}[X|Z] = \mathbb{E}[\mathbb{E}[X|Y, Z]|Z]
\end{align*}

$$
We will assume these variables are continuous (in the discrete case replace integrals with sums):
$$

\begin{align*}
\mathbb{E}[X|Z] &= \int x p(x|z) dx \\
&= \int x \int p(x, y | z) dy dx & \text{Sum rule} \\
&= \int x \int p(x|y, z) p(y|z) dy dx & \text{Bayes' Theorem} \\
&= \int x \int p(x|y, z) p(y|z) dx dy & \text{Fubini's Theorem} \\
&= \int p(y|z) \int x p(x|y, z) dx dy \\
&= \mathbb{E}[\mathbb{E}[X|Y, Z]|Z]
\end{align*}

$$


## Computing $v_\pi$ in terms of $q_\pi$

$$

\begin{align*}
v_\pi(s) &= \mathbb{E}_\pi[G_t | S_t = s] \\
&= \mathbb{E}[\mathbb{E}_\pi[G_t | S_t = s, A_t] | S_t = s] \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) q_\pi(a|s)
\end{align*}

$$


## Computing $q_\pi$ in terms of $v_\pi$

$$

\begin{align*}
q_\pi(s, a) &= \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \\
&= \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \mathbb{E}[\mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a, S_{t+1}] | S_t = s, A_t = a] \\
&= \mathbb{E}[R_{t+1} + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s, A_t = a, S_{t+1}]| S_t = s, A_t = a] & \text{$R_{t+1}$ is only dependent on $S_t$, $A_t$, not on $\pi$} \\
&= \mathbb{E}[R_{t+1} + \gamma \mathbb{E}_\pi[G_{t+1} | S_{t+1}]| S_t = s, A_t = a] & \text{Markov Property} \\
&= \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a] \\
&= \sum_{s', r} p(s', r | s, a) \Bigl[r + \gamma v_\pi(s')\Bigl]
\end{align*}

$$


## Computing $v_\pi$ in terms of $v_\pi$

$$

\begin{align*}
v_\pi(s) &= \sum_{a \in \mathcal{A}(s)} \pi(a|s) q_\pi(a|s) & \text{Based on first section} \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \mathbb{E}[\mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a, S_{t+1}]|S_t = s, A_t = a] \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \mathbb{E}[R_{t+1} + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s, A_t = a, S_{t+1}]| S_t = s, A_t = a] & \text{$R_{t+1}$ is only dependent on $S_t$, $A_t$, not on $\pi$} \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \mathbb{E}[R_{t+1} + \gamma \mathbb{E}_\pi[G_{t+1} | S_{t+1}]|S_t = s, A_t = a] \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t = s, A_t = a] & \text{Markov Property} \\
&= \sum_{a \in \mathcal{A}(s)} \pi(a|s) \sum_{s', r} p(s', r|s, a) \Bigl[r + \gamma v_\pi(s')\Bigl]
\end{align*}

$$


## Computing $q_\pi$ in terms of $q_\pi$

$$

\begin{align*}
q_\pi(s, a) &= \sum_{s', r} p(s', r | s, a) \Bigl[r + \gamma v_\pi(s')\Bigl] & \text{Based on second section} \\
&= \sum_{s', r} p(s', r | s, a) \Bigl[r + \gamma \sum_{a' \in \mathcal{A}(s')} \pi(a'|s') q_\pi(s', a')\Bigl] & \text{Based on first section} \\ 
\end{align*}

$$

