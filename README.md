# Using-MDP-to-solve-Tic-Tac-Toe



Now imagine that you are playing it against a purely random opponent. That is, when the opponent needs to make a move, it will choose uniformly randomly from the empty spots, and put its mark in it. We let the opponent plays first: from the empty 9 spots, it first picks an arbitrary one and marks it at the beginning of the game, and then it is your turn.
• For any game state s that is not terminal (i.e., no winner and there are still empty spots), you receive R(s) = 1
• At any game state s where you are the winner, R(s) = 10, and if you are the loser, then R(s) = −10
• If it is a draw state (all spots are occupied and there is no winner), then R(s) = 0.
