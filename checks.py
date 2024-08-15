import numpy as np

def check_three_in_a_row(board, player):
    rows, cols = board.shape
    reward = 0

    # Check horizontal lines
    for r in range(rows):
        for c in range(cols - 2):  # Check up to the last 3 columns
            if np.all(board[r, c:c+3] == player):
                reward += 1  # Reward for connecting 3 pieces horizontally

    # Check vertical lines
    for c in range(cols):
        for r in range(rows - 2):  # Check up to the last 3 rows
            if np.all(board[r:r+3, c] == player):
                reward += 1  # Reward for connecting 3 pieces vertically

    # Check diagonal lines (bottom-left to top-right)
    for r in range(rows - 2):
        for c in range(cols - 2):
            if np.all(np.diagonal(board[r:r+3, c:c+3]) == player):
                reward += 1  # Reward for connecting 3 pieces diagonally (bottom-left to top-right)

    # Check diagonal lines (top-left to bottom-right)
    for r in range(2, rows):
        for c in range(cols - 2):
            if np.all(np.diagonal(np.fliplr(board[r-2:r+1, c:c+3])) == player):
                reward += 1  # Reward for connecting 3 pieces diagonally (top-left to bottom-right)

    return reward

def check_blocked_opportunity(board, player, opponent):
    rows, cols = board.shape
    blocked_reward = 0

    def is_blocked(segment, player, opponent):
        """ Check if the segment is a blocked opportunity. """
        # Check for exactly 3 opponent's pieces and 1 empty space
        if np.count_nonzero(segment == opponent) == 3 and np.count_nonzero(segment == 0) == 1:
            # Ensure that the empty space is not at the ends of the segment
            if segment[0] == 0 or segment[-1] == 0:
                return False
            return True
        return False

    # Check horizontal lines
    for r in range(rows):
        for c in range(cols - 3):
            segment = board[r, c:c+4]
            if is_blocked(segment, player, opponent):
                blocked_reward += 1

    # Check vertical lines
    for c in range(cols):
        for r in range(rows - 3):
            segment = board[r:r+4, c]
            if is_blocked(segment, player, opponent):
                blocked_reward += 1

    # Check diagonal lines (bottom-left to top-right)
    for r in range(rows - 3):
        for c in range(cols - 3):
            segment = np.diagonal(board[r:r+4, c:c+4])
            if is_blocked(segment, player, opponent):
                blocked_reward += 1

    # Check diagonal lines (top-left to bottom-right)
    for r in range(3, rows):
        for c in range(cols - 3):
            segment = np.diagonal(np.fliplr(board[r-3:r+1, c:c+4]))
            if is_blocked(segment, player, opponent):
                blocked_reward += 1

    return blocked_reward

def check_blunder(board, player, opponent):
    rows, cols = board.shape
    reward = 0

    def is_winning_opportunity(segment, opponent):
        """ Check if the segment is a winning opportunity for the opponent with both ends open. """
        return (
            np.count_nonzero(segment == opponent) == 3 and
            np.count_nonzero(segment == 0) == 2 and
            segment[0] == 0 and segment[-1] == 0
        )

    # Check horizontal lines
    for r in range(rows):
        for c in range(cols - 4 + 1):
            segment = board[r, c:c+5]
            if is_winning_opportunity(segment, opponent):
                reward -= 5

    # Check vertical lines
    for c in range(cols):
        for r in range(rows - 4 + 1):
            segment = board[r:r+5, c]
            if is_winning_opportunity(segment, opponent):
                reward -= 5

    # Check diagonal lines (bottom-left to top-right)
    for r in range(rows - 4 + 1):
        for c in range(cols - 4 + 1):
            segment = np.diagonal(board[r:r+5, c:c+5])
            if is_winning_opportunity(segment, opponent):
                reward -= 5

    # Check diagonal lines (top-left to bottom-right)
    for r in range(4, rows):
        for c in range(cols - 4 + 1):
            segment = np.diagonal(np.fliplr(board[r-4:r+1, c:c+5]))
            if is_winning_opportunity(segment, opponent):
                reward -= 5

    return reward

def evaluate_board_state(board, player, opponent):
    """
    Evaluate the board state for the given player and opponent.
    Returns the total reward/penalty for the current board state.
    """
    total_reward = 0

    total_reward += check_three_in_a_row(board, player)  # Reward for player's connected pieces
    total_reward += check_blocked_opportunity(board, player, opponent)  # Reward for blocking opponent
    total_reward += check_blunder(board, player, opponent)  # Penalty for opponent's potential win

    return total_reward
