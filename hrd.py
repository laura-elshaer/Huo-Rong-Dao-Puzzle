from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys
import itertools
import heapq
# =============================================================================

char_goal = '1'
char_single = '2'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str or None
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single,
                                       self.coord_x, self.coord_y,
                                       self.orientation)


    def move(self, rows, columns):
        self.coord_x += rows
        self.coord_y += columns


class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location
        information.
        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self, file):
        """
        Print out the current board.
        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='', file=file, flush=True)
            print(file=file, flush=True)
        print(file=file, flush=True)

    def legalMoves(self):
        empty_spaces = []
        for i in range(self.height):
         for j in range(self.width):
            if self.grid[i][j] == '.':
                empty_spaces.append((j, i))
                if len(empty_spaces) == 2:
                    return tuple(empty_spaces)



class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.


def goalTest(board):
  
    grid=board.grid
    if grid[3][1] == '1' and grid[3][2] == '1' and grid[4][1] == '1' and grid[4][2] == '1':
        return True

    return False


def manhattan_distance(board):
    main_x=1
    main_y=3
    for piece in board.pieces:
        if  piece.is_goal:
          return abs((piece.coord_x - main_x) + (piece.coord_y - main_y))


def goal_moves(piece, empty_squares):
  
    y = piece.coord_y
    x = piece.coord_x
    way = []
    amount_of_moves=0
    conditions = {
    'right': ((x + 2, y) in empty_squares and (x + 2, y + 1) in empty_squares),
    'left': ((x - 1, y) in empty_squares and (x - 1, y + 1) in empty_squares),
    'up': ((x, y - 1) in empty_squares and (x + 1, y - 1) in empty_squares),
    'down': ((x, y + 2) in empty_squares and (x + 1, y + 2) in empty_squares)
}
    
    if (x + 2, y) in empty_squares and (x + 2, y + 1) in empty_squares:
        way.append([1, 0])
        amount_of_moves += 1
    if conditions['left']:
        way.append([-1, 0])
        amount_of_moves+=1
    if conditions['up']:
        way.append([0, -1])
        amount_of_moves += 1
    if conditions['down']:
        way.append([0, 1])
        amount_of_moves += 1

    return way, amount_of_moves




def single_moves(piece, empty_squares):
 
    x, y = piece.coord_x, piece.coord_y
    way = [[1, 0], [-1, 0], [0, -1], [0, 1]]
    valid_way = [dir for dir in way if (x + dir[0], y + dir[1]) in empty_squares]
    amount_of_moves = len(valid_way)
    return valid_way, amount_of_moves

def vertical_pieces(piece, empty_squares):
    y = piece.coord_y
    x = piece.coord_x
    way = []
    amount_of_moves = 0 
    moves=([1,0], [-1,0], [0,-1], [0,1])

   
    if (x + 1, y) in empty_squares and (x + 1, y + 1) in empty_squares:
        way.append(moves[0])
        amount_of_moves += 1

    
    if (x - 1, y) in empty_squares and (x - 1, y + 1) in empty_squares:
        way.append(moves[1])
        amount_of_moves += 1

    if (x, y - 1) in empty_squares:
        way.append(moves[2])
        amount_of_moves += 1

    if (x, y + 2) in empty_squares:
        way.append(moves[3])
        amount_of_moves += 1

    return way, amount_of_moves


def horizontal_pieces(piece, empty_squares):
    y = piece.coord_y
    x = piece.coord_x
    way = []
    amount_of_moves=0
    x2,y2=x+1, y+1
    if (x2 + 1, y) in empty_squares:
        way.append([1, 0])
        amount_of_moves=+1
    if (x - 1, y) in empty_squares:
        way.append([-1, 0])
        amount_of_moves=+1
    if (x, y - 1) in empty_squares and (x2, y - 1) in empty_squares:
        way.append([0, -1])
        amount_of_moves=+1
    # move down
    if (x, y2) in empty_squares and (x2, y2) in empty_squares:
        way.append([0, 1])
        amount_of_moves=+1

    return way



def intermediate(original_state, piece, way): #generates the states between start and goal
    ans= []

    for method in way:
        copy = deepcopy(original_state.board)
        for main in copy.pieces:
            if main.coord_x == piece.coord_x and main.coord_y == piece.coord_y:
               changedPiece=main
        x_axis,y_axis=method[0], method[1]
        changedPiece.move(x_axis, y_axis)
        new_board = Board(copy.pieces)
        update_depth=original_state.depth + 1
        update_value=manhattan_distance(new_board)
        updated_state = State(new_board, update_value,
                         update_depth , original_state)
        ans.append(updated_state)
    return ans


def successor(state):
   
    available = state.board.legalMoves()
    ans = []
    pieces=state.board.pieces
    for piece in pieces:
        if piece.is_single:
            way, amount_of_moves = single_moves(piece,available )
            ans.extend(intermediate(state, piece, way))
        
        elif piece.orientation == 'v':
            way, amount_of_moves = vertical_pieces(piece, available )
            ans.extend(intermediate(state, piece, way))
        elif piece.orientation == 'h':
            way = horizontal_pieces(piece, available )
            ans.extend(intermediate(state, piece, way))
        elif piece.is_goal:
             way,amount_of_moves = goal_moves(piece, available )
             ans.extend(intermediate(state, piece, way))
    
    return ans


def solvedPuzzle(answer) :
    ans=[]
    given=answer
    for _ in range(answer.depth + 1):
        ans.insert(0, given)
        given = given.parent
    return ans


def dfs(start):
     front=[start] #store the starting state in a frontier 
     visited=set() #here, im initializing the visted set
     while True: 
        if not front: #if theres no element in the frontier then return None
            return None
        #if there is something, we remove it from frontier and add it to set
        current=front.pop() 
        hash_value=hash(str(current.board.grid))
        visited.add(hash_value)
   
        if goalTest(current.board):
            return current
        
        next=successor(current)
        unvisited = [
            successor
            for successor in next
            if hash(str(successor.board.grid)) not in visited
        ] # add all successors not visited here
        front.extend(unvisited)




def astar_algorithm(given):
    visited = set()
    count = 0 #associates an id WITH EACH entered ser

    f_func = given.depth + manhattan_distance(given.board) #A* is arranged based on this function
    front = [(f_func, count, given)] #We start with the id of inserted set

    while front:
        _, _, current = heapq.heappop(front) #here, current will be set to the state

        if goalTest(current.board): #if this current is the goal, stop here and return ir
            return current

        current_hashValue = hash(str(current.board.grid))
        if current_hashValue not in visited: #set keeps track of what we vsiited
            visited.add(current_hashValue)

       

            for next in successor(current):
                next_f = next.depth + manhattan_distance(next.board)
                count += 1
                heapq.heappush(front, (next_f, count, next))



def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if not g_found:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)

    return board


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)
    if args.algo == 'dfs':
        original_state = dfs(
            State(board, manhattan_distance(board), 0, None))

        file = open(args.outputfile, 'w')

        for i in range(len(solvedPuzzle(original_state))):
            print(solvedPuzzle(original_state)[i].board.display(file))

        file.close()
    
    elif args.algo == 'astar':

        startPoint = astar_algorithm(
            State(board, manhattan_distance(board), 0, None))

        file = open(args.outputfile, 'w')
        iteration=len(solvedPuzzle(startPoint))
        for j in range(iteration):
            print(solvedPuzzle(startPoint)[j].board.display(file))
      
        file.close()