---
title: 'Nowhere Dense Sets'
date: 2026-06-14T07:07:07+01:00
slug: nowhere_dense_sets
math: 'mathjax'
---

Problem 2.14 in *Probability and Measure* by Billingsley requires
knowing the definition of nowhere-dense sets. Appendix 15 provides a
definition of nowhere-dense sets and sets of the first category with
respect to $\mathbb{R}$:

> "The set $$A$$ is, by definition, ****dense**** in the set $$B$$ if for
> each $$x \in B$$ and each open interval $$J$$ containing $$x$$, $$J$$
> meets $$A$$. This is the same thing as requiring $$B \subseteq
> \overline{A}$$ (the closure of $$A$$).  The set $$E$$ is, by
> definition, ****nowhere dense**** if each open interval $$I$$ contains
> some open interval $$J$$ that does not meet $$E$$. This makes sense:
> if $$I$$ contains an open interval $$J$$ that does not meet $$E$$,
> then $$E$$ is not dense in $$I$$; the definition requires that $$E$$
> be dense in no interval $$I$$."

The usual topological definition of density is as follows: Let $(X,
d_X)$ be a metric space. Let $A, B \subset X$. We say that $A$ is
****dense**** in $B$ if and only if $\overline{A} = B$. The topological
definition of nowhere-dense is: $A$ is nowhere-dense in $X$ if
$(\overline{A})^{\mathrm{o}} = \emptyset$. We show that these
definitions are consistent with the ones given in the textbook when $X
= \mathbb{R}$.

Let's show that the definition of density is consistent. First, we
suppose that $A$ is dense in $B$ if and only if for every $x \in B$
and every open interval $J$ containing $x$, $J \cap A \ne
\emptyset$. We show that this means $\overline{A} = B$. Let's show
that $B \subset \overline{A}$. Let $x \in B$. We show that $x \in
\overline{A}$. There exists open interval $J$, $x \in J$, such that $J
\cap A \ne \emptyset$. If $x \in J \cap A$, then $x \in \overline{A}$
(and so we're done). Otherwise, if $x \not\in J \cap A$, then $(J
\setminus \{x\}) \cap A \ne \emptyset$. Since this is true for every
open interval $J$ containing $x$, $x \in A'$ (where $A'$ denotes the
limit points of $A$). Therefore, $x \in \overline{A}$. Now we show
that $\overline{A} \subset B$. Let $x \in \overline{A}$. For every
open interval $x \in J$, $J \cap A \ne \emptyset$. This means $x \in
B$ ; if not, then $A$ would not be dense in $B$ (because for every $x
\in B$ and every open interval $J$ containing $x$, $J \cap A \ne
\emptyset$). So, $\overline{A} = B$.

Now, let's suppose that $\overline{A} = B$. Then, we show that for
every $x \in B$ and every open interval $J$ containing $x$, $J \cap A
\ne \emptyset$. Let $x \in B$. Then, $x \in \overline{A}$. So, $x \in
A$ or $x \in A'$. Choose an arbitrary open interval $J$ containing
$x$. If $x \in A$, then $J \cap A \ne \emptyset$ because $x \in J \cap
A$. Otherwise, if $x \in A'$, then $(J \setminus \{x\}) \cap A \ne
\emptyset$. This immediately implies $J \cap A \ne \emptyset$.

This proves that the definitions of density are equivalent (for $X =
\mathbb{R}$). Now, we show that the definition of nowhere-dense is
consistent when $X = \mathbb{R}$. First, let's clarify the definition
given by Billingsley: $A$ is nowhere-dense if for every open interval
$I$, there exists open interval $J \subset I$ so that $J \cap A =
\emptyset$. This is the same thing as saying $A$ is not dense in any
open interval $I$. To see this, note that if $A$ is not dense in any
$I$ (nowhere-dense), then for some $x \in I$ and some open interval
$J$ containing $x$, $J \cap A = \emptyset$. Then, for any open
interval $J' \subset J$, note that $J' \cap A = \emptyset$. So, if we
take this interval small enough, we can find a $J' \subset I$ so that
$J' \cap A = \emptyset$.

Now, we show that the topological definition is consistent. First,
suppose that Billingsley's definition is true. We show that
$(\overline{A})^{\mathrm{o}} = \emptyset$. Suppose for contradiction
that $x \in \overline{A}^{\mathrm{o}}$. Then, there exists open
interval $I$ such that $x \in I$ and $I \subset \overline{A}$. Since
$A$ is nowhere-dense, there exists open interval $J \subset I$ such
that $J \cap A = \emptyset$. But $J \subset \overline{A}$. Since $J
\cap A = \emptyset$, that means $J \subset A'$. Consider any $y \in
J$. Then, $y \in A'$. Since $J$ is open, $(J \setminus \{y\}) \cap A
\ne \emptyset$. This implies that $J \cap A \ne \emptyset$, which is a
contradiction. So, $(\overline{A})^{\mathrm{o}} = \emptyset$.

Next, let's prove the other direction: suppose that
$(\overline{A})^{\mathrm{o}} = \emptyset$. Suppose there exists open
interval $I$ such that for every open interval $J$, $J \subset I$, $J
\cap A \ne \emptyset$. Choose any $x \in I$. For every open $J$ containing
$x$, $J \cap A \ne \emptyset$. Then, either $x \in A$ or $x \in
A'$. This implies that $x \in \overline{A}$. Since $x \in I$ was
arbitrary, this means that $I \subset \overline{A}$. For any $x$, $I$
is an open neighborhood contained in $\overline{A}$, so
$(\overline{A})^{\mathrm{o}} \ne \emptyset$. But this contradicts the
hypothesis, so we must have that for every open interval $I$, there
exists an open interval $J$, $J \subset I$, such that $J \cap A =
\emptyset$.

This shows that the topological definition of nowhere-dense is the
same as the definition given by Billingsley when we take $X = \mathbb{R}$.

