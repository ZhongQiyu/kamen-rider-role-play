# 2.1 Definitions

## Payoff Matrix

### Range
- A = (a_ij) where i = 1, ..., m and j = 1, ..., n belonging to R^(m*n)
- This models the possible actions for player II

### Operator
a_ij = payoff from player II to player I if player I selects action i
This applies if player II selects action i as well

## Worst Case Analysis

### Player I
If she selects action i, in the worst case her *gain* will be `min(a_ij)` where `j = 1, ..., n`
Hence, the largest gain she can guarantee `max(min(a_ij))` where `i` and `j` holds their ranges

### Player II
If he selects action j, in the worst case his *loss* will be `max(a_ij)` where `i = 1, ..., m`
Hence, smallest loss he can guarantee `min(max(a_ij))` where `i` and `j` holds in their ranges

### Pick A Hand
We will formally prove later max(min()) is smaller than or equal to min(max())

## Mixed Strategies

Each action is selected with some probability

### Player I

delta_m = {x belongs to the real number space of m so that x is larger than 0 while x_i = 1}

### Player II

delta_n = {y belongs to the real number space of n so that y is larger than 0 while y_i = 1}

## Pure Strategy

A mixed strategy in which a particular action is played with probability 1.
Notation: e_i = (0, ..., 0, 1, 0, ..., 0) = pure strategy with action i

## Expected gain of player I (= expected loss of player II)

player I employs x belongs to , player II employs y belongs to delta
x^T*A*y = sigma_m(i = 1, ..., m) 

## Expected 

Example 2.3: (Illustration of (1) and (2))
m = n = 2:
A = (a_11, a_12, a_21, a_22)
(1) E[gain player I | player II plays 1] = 

(2) min(X^T*A*y) = min(y belongs to [0,1]) {(x_1*a_11 + x_2*a_21)*y_1 + (x_1*a_12 + x_2*a_12)(1-x)}

(3) Over x belongs to delta_m of the function x goes to min(y belongs to delta_n) 
Each action is selected with some probability

... (4)

# Minimax Theorem

The two players' safety values coincide.

## Theorem 2.5 (von Neumann's Minimax Theorem)
For any two-player zero-sum game with `m*n` payoff matrix A, there is a number V called the value of the game,
satisfying 
max(min(X^T*A*y)) = V = min(max(X^T*A*y)) // this seems to be a follow-up for pure-strategy

Proof Later