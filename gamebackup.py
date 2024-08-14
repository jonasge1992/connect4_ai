import numpy as np
import random
import pygame
import sys
import math

class Connect4_AI_Game:

    def __init__(self):
        # Define colors
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        # Define board dimensions
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.SQUARESIZE = 100
        self.WINDOW_LENGTH = 4
        self.RADIUS = int(self.SQUARESIZE / 2 - 5)

        # Define player and AI pieces
        self.PLAYER = 0
        self.AI = 1
        self.EMPTY = 0
        self.PLAYER_PIECE = 1
        self.AI_PIECE = 2

        # Initialize pygame
        pygame.init()

        # Set up the display
        self.width = self.COLUMN_COUNT * self.SQUARESIZE
        self.height = (self.ROW_COUNT + 1) * self.SQUARESIZE
        self.size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Connect 4")

        # Set up the font
        self.myfont = pygame.font.SysFont("monospace", 75)

        # Create the board
        self.board = self.create_board()
        self.game_over = False
        self.turn = random.randint(self.PLAYER, self.AI)

    def create_board(self):
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))

    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def is_valid_location(self, board, col):
        return board[self.ROW_COUNT-1][col] == 0

    def get_next_open_row(self, board, col):
        for r in range(self.ROW_COUNT):
            if board[r][col] == 0:
                return r

    def print_board(self, board):
        print(np.flip(board, 0))

    def get_board(self, board):
        return np.flip(board, 0)

    def winning_move(self, board, piece):
        # Check horizontal locations for win
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diagonals
        for c in range(self.COLUMN_COUNT-3):
            for r in range(3, self.ROW_COUNT):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = self.PLAYER_PIECE
        if piece == self.PLAYER_PIECE:
            opp_piece = self.AI_PIECE

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(self.EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(self.EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(self.EMPTY) == 1:
            score -= 4

        return score

    def score_position(self, board, piece):
        score = 0

        # Score center column
        center_array = [int(i) for i in list(board[:, self.COLUMN_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(self.ROW_COUNT):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(self.COLUMN_COUNT - 3):
                window = row_array[c:c + self.WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(self.COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(self.ROW_COUNT - 3):
                window = col_array[r:r + self.WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonal
        for r in range(self.ROW_COUNT - 3):
            for c in range(self.COLUMN_COUNT - 3):
                window = [board[r + i][c + i] for i in range(self.WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        # Score negative sloped diagonal
        for r in range(self.ROW_COUNT - 3):
            for c in range(self.COLUMN_COUNT - 3):
                window = [board[r + 3 - i][c + i] for i in range(self.WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    def is_terminal_node(self, board):
        return self.winning_move(board, self.PLAYER_PIECE) or self.winning_move(board, self.AI_PIECE) or len(self.get_valid_locations(board)) == 0

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(board, self.AI_PIECE):
                    return (None, 100000000000000)
                elif self.winning_move(board, self.PLAYER_PIECE):
                    return (None, -10000000000000)
                else: # Game is over, no more valid moves
                    return (None, 0)
            else: # Depth is zero
                return (None, self.score_position(board, self.AI_PIECE))
        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, self.AI_PIECE)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else: # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, self.PLAYER_PIECE)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.COLUMN_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def pick_best_move(self, board, piece):
        valid_locations = self.get_valid_locations(board)
        best_score = -10000
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = board.copy()
            self.drop_piece(temp_board, row, col, piece)
            score = self.score_position(temp_board, piece)
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def draw_board(self, board):
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                pygame.draw.rect(self.screen, self.BLUE, (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE))
                pygame.draw.circle(self.screen, self.BLACK, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)

        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                if board[r][c] == self.PLAYER_PIECE:
                    pygame.draw.circle(self.screen, self.RED, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
                elif board[r][c] == self.AI_PIECE:
                    pygame.draw.circle(self.screen, self.YELLOW, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
        pygame.display.update()

    def play_step(self, move):
        return move

    def game_over(self, event):
        if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def play_game(self):
        self.draw_board(self.board)
        pygame.display.update()

        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Player's turn
            if self.turn == self.PLAYER:
                col = self.play_step(random.randint(0,5))
                if self.is_valid_location(self.board, col):
                    row = self.get_next_open_row(self.board, col)
                    self.drop_piece(self.board, row, col, self.PLAYER_PIECE)

                    if self.winning_move(self.board, self.PLAYER_PIECE):
                        label = self.myfont.render("Player 1 wins!!", 1, self.RED)
                        self.screen.blit(label, (40, 10))
                        self.game_over = True

                    self.turn += 1
                    self.turn = self.turn % 2

                    self.print_board(self.board)
                    self.draw_board(self.board)

            # AI's turn
            if self.turn == self.AI and not self.game_over:
                col, self.minimax_score = self.minimax(self.board, 5, -math.inf, math.inf, True)

                if self.is_valid_location(self.board, col):
                    row = self.get_next_open_row(self.board, col)
                    self.drop_piece(self.board, row, col, self.AI_PIECE)

                    if self.winning_move(self.board, self.AI_PIECE):
                        label = self.myfont.render("Player 2 wins!!", 1, self.YELLOW)
                        self.screen.blit(label, (40, 10))
                        self.game_over = True

                    self.print_board(self.board)
                    self.draw_board(self.board)

                    self.turn += 1
                    self.turn = self.turn % 2

            if self.game_over:
                pygame.time.wait(3000)

# To start the game
if __name__ == "__main__":
    game = Connect4_AI_Game()
    game.play_game()
