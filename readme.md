https://github.com/The-Chris-Davies/COMP472_Miniproject2

# Line-Em-Up:
A 2 player N-in-a-row game similar to tic-tac-toe. The board size is nxn, and s pieces in a row are required to win. Additionally, an arbitrary number of 'blocks' can be placed on the board prior to the first move, which prevent players from putting their pieces in these positions.

## Execution:
To run this program, call lineemup.py like so:
    python lineemup.py n b [bx,by]* s d1 d2 t a1 a2 mode
where:  
- `n` - the board size (`n` by `n`)
- `b` - the number of blocks to place
- `[bx, by]` - a sequence of `b` block positions
- `s` - the number of pieces in a row required to win the game
- `d1` - the maximum recursion depth that the first AI can search to
- `d2` - the maximum recursion depth for the second AI
- `t` - the maximum time allowed per move
- `a1` - set to `True` to use alpha-beta pruning for the first AI
- `a2` - set to `True` to use alpha-beta pruning for the second AI
- `mode` - whether each player is a human or AI. can be one of:
    - `H-H` - two human players
    - `H-AI` - player X is human, player O is AI
    - `AI-H` - player X is AI, player O is human
    - `AI-AI` - both players are AI

This runs a single game with the specified parameters, and saves a game trace. Multiple games can be run using the python interpreter, see `lineemup.run_experiments()` as an example.

## Requirements:
- NumPy
- SciPy

Or, alternatively, to run convolutions on the GPU,
- CuPy
