# Glossary:
- s - number of stones-in-a-row required to win
- n - grid size (nxn)
- line - a winning length of s stones in a row
- stones - x's / o's (think playing on a go board)
- holes - empty spaces (that could potentially complete a line)

## Introduction:
While developing this project, I neglected to look into the document too deeply, and this is a decision I will regret with my heart of hearts until the day I die (or submit this project, at the very least).
Because of this, I had designed my system to operate with (relatively) high speed on large board sizes - n>100 for example.
This added quite a bit more work to the project, and as a result, I had less time to optimise my heuristics and implement other speedups that would improve small-board performance. These design decisions are described in the 'Board Representation' section.

## Board representation:
In order to represent large board sizes efficiently, I decided to use a set of three sparse numpy arrays. These arrays represent the locations of X's moves, O's moves, and the Blocs on the board.
The primary operation performed on this system is a convolution of the win conditions across one or a combination of these arrays.
The output of this operation is an array with smaller dimensions that represent the number of pieces present in each possible win state.
Because this operation is a convolution, for larger board states, it is performed in the frequency domain via fft transformation. This means that the board state checking is much faster for larger board states than a simple iterative approach would be. (but also much slower for the board sizes actually described in the project description).
My heuristic functions use this operation to determine the number of viable win conditions, and the number of moves required to complete these win conditions.

## Agents:
I created the Agent class to allow more flexibility in the AI that plays against eachother. Any player is an instance of an agent - be it a human player or an AI. This way, it is easy to create different AIs with many parameters, and to select between them. Looking back, this was much more work than it was worth, and likely makes it impossible to integrate my AI with the rest of the class' for the tournament. Oh well.

## Heuristic Functions:
### heuristic_1:
`heuristic_1` calculates the number of potential wins for each player, and weights them based on how many pieces each condition requires to win. The equations used to do so is as follows:

![heuristic 1 equation](/docs/imgs/h1_eq.png)

where `count(k,p)` is the number of possible s-in-a-rows that have `k` pieces in them for player `p`.

Note that the weight for each of these counts (2^k) is not optimal - I know that, for example, two (s-1)-in-a-rows is equivalent to one s-in-a-row, but I didn't do a full combinatorial game theory analysis of the move combinations as I would have liked. 

This approach fails to consider the case where multiple possible s-in-a-rows share a hole, as it assumes all are independent.

### heuristic_2:
`heuristic_2` is similar to heuristic 1, except it ignores whether a state is valid or not, and does not weight its moves by `k` instead of 2^k. This way, it runs 2/3 of the convolutions of heuristic_1 and avoids the comparisons that determine if a state is valid. This way, it is faster than the first heuristic, but not as informed.

## Speed:
my system is currently very slow (due to all the array operations). That said, it is *very* parallelizable, so if the heuristic function was gpu accelerated, I believe this would be very quick. (the way the heuristic is found from the convolution involves a lot of array comparisons, which are a perfect use case for SIMD operations on the GPU). I have written my program to use CuPy if available, but because I don't have an Nvidia GPU, I am not able to test it.

because of all the time I wasted on mediocre and irrelevant speed improvements, my AI is relatively unoptimised, and could do with improvement.
I have not implemented any sorting into alpha-beta pruning, given more time this would be the first thing I do.

All of my functions are very unoptimised as well: for example, I could cache the evaluated heuristic for each state, to avoid calculating the heuristic twice for the same state. This would provide huge speedups.
There are also many calculations that could be done once and saved instead of repeatedly recalculating, for example the convolution of the bloc state array does not change, so this only needs to be computed once.
There are several other changes like this that would provide (probably) a large boost to compute time if implemented, but unfortunately I did not have a chance to before the project deadline.

## Improvements:
- The weighting used in the heuristics was not calculated to be optimal at all: if I had spare time, I would have done some form of optimisation for the weights, and tried to determine a formula that would give the optimal bias for each entry.
(I know that two s-1 rows is a guranteed win, and so is equivalent to one row of s, and that three s-2 rows is equivalent to one s-1 row, but I didn't take this into account really)
- Currently, the algorithm just drops everything when it runs out of time. Ideally, it should determine the branching factor and predict the maximum depth it can reach without exceeding the time limit, but again I was unable to do this with the time available to me.

## Input:
the game is called through the command line like so:
python lineemup.py n b [bx,by] s d1 d2 t a1 a2 mode

This runs one game with the specified parameters and saves a game trace.
