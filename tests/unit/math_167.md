`8.13.24`

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

***REVISION NEEDED***

`8.14.24`

# Ex. 21

- Nim w/ 9, 10, 11, 12
	- 1001, 1010, 1011, 1100
	- This with (circle-plus) outputs 0100

- .. .. 9, 10, 11, 8
	- 1001, 1010, 1011, 1000
	- This then gives 0000

- Analysis on the Operator:
	- We take all the bits together
	- Then we count on how many zeros and ones there are
	- It is only when we have odd counts of ones that the output bit (rounded) will give 1
	- Otherwise all bits get to be zero.

- Distinction from other operators
	- AND: chaining ones only to give 1
	- OR: chaining zeroes only to give 0
	- NOR: flip-the-blip
	- XOR: propagate the difference of bits, to give 1 only when different bits are provided

# Ex. 22

## Player I

`(0 2 5 1)`

qqqqqqqqq

0y1 + 5(1 - y1) = 5 - 5y1

2y1 + (1-y1) = ...

5 - 5x1 = 1 + y1
4 = 6x1
2/3 = x1

## Player II

0x1 + 2(1-x2) = 2-2x1
5x1 + 1(1-x1) = 4x + 1

1 = 6x1
1/6 = x1

# Ex. 23

2 Players since call at other 1 or 2
I wins if the sum is odd
II wins if the sum is even
loser pays winner the sum of #s called

    1I 2II
1II -2  3
2I  3  -4

If 1II -2x1 + 3(1-x2) = 3 - 5x1
If 2II 3x1 - 4(1-x2) = -4 - 7x1

  A B
C 3 0
D 0 3
E 2 2

determine all mining
stats  x prob II plays A
if C plays
C 3x + 0(1 - x) = 3x
D Ox + 3(1 - x) = 3 - 3x
E 2x + 2(1 - x) = 2

winning stats for player I
y1 of C
y2 of D
1 - y1 - y2 of E

if player 2 plays
A 3y1 + 0y2 + 2(1 - y1 - y2) = 2 + y1 - 2y2
B 0y1 + 3y2 + 2(1 - y1 - y2) = 2 + y2 - 2y1

3y1 = 3y2
y1 = y2 | y

they only play one kind of E

# Saddle Point

- A belongs to R^(m*n)
- (i*, j*) is a saddle point if
- max(j belongs 1, ..., m) (a_ij) = (a_i*j) = min(a_i*j)

if (i*, j*) is r saddle point, then (a_i*j*) is the value of the genre
e(choice) is plays(choice) w/ 100%

# Ex. 24

(
(20 1 4 3 1)
( 2 3 8 4 4)
(10 8 7 6 9)
( 5 6 1 2 2)
( 3 7 9 1 5)
)

# Ex. 26

Show that all saddle points of a zero result the matrix