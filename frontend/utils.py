import pygame
import os
import time

# Constants #
# Screen setup
WIDTH  = 115*8 #//2
HEIGHT = 115*8 #//2
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)#, pygame.FULLSCREEN)
BACKGROUND = pygame.Surface((WIDTH, HEIGHT))
PIECES = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
MARKERS = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
pygame.display.set_caption("Chess Game")
# Colors
WHITE = (235, 236, 208) # 202, 203, 179
BLACK = (115, 149, 82)  # 99, 128, 70
SPECIAL_WHITE = (245, 246, 130)
SPECIAL_BLACK = (185, 202, 67)
# Board setup
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
# Piece dictionary
img_piece = {"white_pawn": 1,
        "black_pawn": 2,
        "white_knight": 3,
        "black_knight": 4,
        "white_bishop": 5,
        "black_bishop": 6,
        "white_rook": 7,
        "black_rook": 8,
        "white_queen": 9,
        "black_queen": 10,
        "white_king": 11,
        "black_king": 12}
piece_img = dict((v, k) for k, v in img_piece.items())
piecedict = {0: "", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}

# Functions #
# Loading
def load_images():
    pieces = ['rook', 'knight', 'bishop', 'queen', 'king', 'pawn']
    images = {}
    for piece in pieces:
        images[f'white_{piece}'] = pygame.transform.scale(
            pygame.image.load(os.path.join("frontend/img", f"white-{piece}2.png")),
            (SQUARE_SIZE, SQUARE_SIZE)
        )
        images[f'black_{piece}'] = pygame.transform.scale(
            # pygame.image.load(os.path.join("img", f"black-{piece}2.png")),
            pygame.image.load(os.path.join("frontend/img", f"resized/black-{piece}.png")),
            (SQUARE_SIZE, SQUARE_SIZE)
        )
    images["circle"] = pygame.transform.scale(
            # pygame.image.load(os.path.join("img", f"circle2.png")),
            pygame.image.load(os.path.join("frontend/img", f"resized/circle.png")),
            (SQUARE_SIZE, SQUARE_SIZE))
    return images

def load_sounds():
    pygame.mixer.init()
    sounds_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sounds")
    sounds = {
    "start": pygame.mixer.Sound(os.path.join(sounds_dir, "start.mp3")),
    "move": pygame.mixer.Sound(os.path.join(sounds_dir, "move.mp3")),
    "capture": pygame.mixer.Sound(os.path.join(sounds_dir, "capture.mp3")),
    "check": pygame.mixer.Sound(os.path.join(sounds_dir, "check.mp3")),
    "castle": pygame.mixer.Sound(os.path.join(sounds_dir, "castle.mp3")),
    "end": pygame.mixer.Sound(os.path.join(sounds_dir, "end.mp3")),
    "illegal": pygame.mixer.Sound(os.path.join(sounds_dir, "illegal.mp3")),
    "promote": pygame.mixer.Sound(os.path.join(sounds_dir, "promote.mp3"))}
    return sounds

# Drawing
def draw_background(screen, images):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
def draw_pieces(screen, board, images):
    bitboards = board.reportBitboards()
    for piecenr in range(12):  # Iterate through all 12 piece types
        current_bitboard = bitboards[piecenr]
        while current_bitboard:
            square = current_bitboard.bit_length() - 1
            row = square // 8
            col = square % 8
            screen.blit(images[piece_img[piecenr + 1]], (col * SQUARE_SIZE, abs(row-7) * SQUARE_SIZE))
            current_bitboard &= ~(1 << square)  # Clear the least significant bit
def draw_select_piece(screen, screen2, square, legal_moves, images):
    # updates the background, so that the right squares are highlighted
    # updates the markers, so that the right squares are highlighted
    col = square % 8
    row = square // 8
    # change background
    if (col + row)%2 == 1: pygame.draw.rect(BACKGROUND, SPECIAL_WHITE, (col * SQUARE_SIZE, abs(row-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    if (col + row)%2 == 0: pygame.draw.rect(BACKGROUND, SPECIAL_BLACK, (col * SQUARE_SIZE, abs(row-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    # add markers of possible moves
    legal_pos = []
    for move in legal_moves:
        legal_pos.append((((move >> 6) & 0x3F)%8, ((move >> 6) & 0x3F)//8))
    
    for x, y in legal_pos: # add markers of possible moves
        if (x+y)%2 == 0: screen2.blit(images["circle"], (x * SQUARE_SIZE, abs(y-7) * SQUARE_SIZE))
        else: screen2.blit(images["circle"], (x * SQUARE_SIZE, abs(y-7) * SQUARE_SIZE))
def draw_deselect_piece(screen, square):
    col = square % 8
    row = square // 8
    if (col+row)%2 == 0: pygame.draw.rect(screen, BLACK, (col * SQUARE_SIZE, abs(row-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    else: pygame.draw.rect(screen, WHITE, (col * SQUARE_SIZE, abs(row-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
def draw_make_move(screen, screen2, board, move, last_move, images):
    # updates the background, so that the right squares are highlighted
    # updates the pieces, so that the right pieces are shown
    screen2.fill((0, 0, 0, 0))
    draw_pieces(screen2, board, images)

    # remove the highlighted background of the last move
    lx, ly = ((last_move >> 6) & 0x3F)%8, ((last_move >> 6) & 0x3F)//8
    if (lx+ly)%2 == 0: pygame.draw.rect(screen, BLACK, (lx * SQUARE_SIZE, abs(ly-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    else: pygame.draw.rect(screen, WHITE, (lx * SQUARE_SIZE, abs(ly-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    lx, ly = (last_move & 0x3F)%8, (last_move & 0x3F)//8
    if (lx+ly)%2 == 0: pygame.draw.rect(screen, BLACK, (lx * SQUARE_SIZE, abs(ly-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    else: pygame.draw.rect(screen, WHITE, (lx * SQUARE_SIZE, abs(ly-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # now draw the highlighted background of the current move
    x, y = (move & 0x3F)%8, (move & 0x3F)//8
    if (x+y)%2 == 0: pygame.draw.rect(screen, SPECIAL_BLACK, (x * SQUARE_SIZE, abs(y-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    else: pygame.draw.rect(screen, SPECIAL_WHITE, (x * SQUARE_SIZE, abs(y-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    x, y = ((move >> 6) & 0x3F)%8, ((move >> 6) & 0x3F)//8
    if (x+y)%2 == 0: pygame.draw.rect(screen, SPECIAL_BLACK, (x * SQUARE_SIZE, abs(y-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    else: pygame.draw.rect(screen, SPECIAL_WHITE, (x * SQUARE_SIZE, abs(y-7) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def get_from(move: int) -> int: return move & 0x3F
def get_to(move: int) -> int: return (move >> 6) & 0x3F
def is_castling(move: int) -> bool: return (move >> 12) & 0x1
def get_promoted_piece(move: int) -> int: return (move >> 13) & 0x3
def is_promotion(move: int) -> bool: return (move >> 15) & 0x1


# Sounds
def play_sound(move, fromPiece, toPiece, board, sounds):
    # // bits 0-5: from square
    # // bits 6-11: to square
    # // bits 12: castling flag
    # // bits 13-14: promotion piece
    # // bits 15: promotion flag
    
    if board.isCheck():
        sounds["check"].play()
    elif is_promotion(move) != 0:
        sounds["promote"].play()
    elif toPiece != 0:
        sounds["capture"].play()
    elif fromPiece == 6:
        if abs(get_from(move) - get_to(move)) == 2:
            sounds["castle"].play()
        else:
            sounds["move"].play()
    else:
        sounds["move"].play()
    time.sleep(0.1)

# Game End
def printCheckmate(board, sounds, fen, san, whiteToMove):
    sounds["end"].play()
    print("\n")
    print("-"*25, "\nWhite" if whiteToMove else "\nBlack", " has won the game!")
    print("-"*25,"\nMove history:")
    printHistory(fen, san)
    print("-"*25)

def printDraw(board, sounds, fen, san, whiteToMove):
    sounds["end"].play()
    print("\n")
    print("-"*25, "\nDraw!")
    print("-"*25,"\nMove history:")
    printHistory(fen, san)
    print("-"*25)

def toSAN(move, fromPiece, toPiece, piecesAttacking, board):
    # Extract move details
    tosq = get_to(move)
    fromsq = get_from(move)

    # print(fromsq, tosq, fromPiece, toPiece)

    # Handle castling
    if fromPiece == 6:
        if abs(tosq - fromsq) == 2:  # Castling move
            if tosq > fromsq: return "O-O"  # Kingside castling
            else: return "O-O-O"  # Queenside castling
    
    san = ""
    file_to = chr(tosq % 8 + ord('a'))  # Destination square's file
    rank_to = str(tosq // 8 + 1)  # Destination square's rank
    rank_from = str(fromsq // 8 + 1) # Source rank
    file_from = chr(fromsq % 8 + ord('a'))  # Source file

    # Handle pawn moves
    if fromPiece == 1:
        if toPiece != 0:  # Pawn capture
            san = f"{file_from}x{file_to}{rank_to}"
        else:  # Regular pawn move
            san = f"{file_to}{rank_to}"

        # Handle promotion
        if rank_to == '8' or rank_to == '1':
            promotion_piece = get_promoted_piece(move) + 2
            if promotion_piece == "2": san += f"=N"
            elif promotion_piece == "3": san += f"=B"
            elif promotion_piece == "4": san += f"=R"
            elif promotion_piece == "5": san += f"=Q"

    # Handle other pieces' moves
    else: 
        san += piecedict[fromPiece]

        # disambiguation logic
        if len(piecesAttacking) > 1:
            # More than one piece can make this move, need to disambiguate
            same_file = any(p != fromsq and p % 8 == fromsq % 8 for p in piecesAttacking)
            same_rank = any(p != fromsq and p // 8 == fromsq // 8 for p in piecesAttacking)
            
            if same_file and same_rank:
                san += file_from + rank_from
            elif same_file:
                san += rank_from
            else:
                san += file_from
        
        if toPiece != 0:
            san += "x" + file_to + rank_to
        else: 
            san += file_to + rank_to

    # Handle check or checkmate
    if board.isCheckmate():
        san += "#"
    elif board.isDraw():
        san += " 1/2-1/2"
    elif board.isCheck():
        san += "+"

    return san

def get_pieces_attacking(tosq, fromPiece, board):
    # returns list of squares from which the same piece types attack the square
    pieces_attacking = []
    moves = board.generateAllLegalMoves()
    for move in moves:
        # Extract the 'from' square and the piece type for this move
        from_square = get_from(move)
        piece_type = board.getPieceOfSquare(get_from(move))
        # Check if this move's destination is the square we're interested in
        # and if the piece type matches the one we're looking for
        if get_to(move) == tosq and piece_type == fromPiece:
            # If so, add the 'from' square to our list
            pieces_attacking.append(from_square)
    
    return pieces_attacking

def printCurrentPGN(san, plycount):
    string = ""
    if plycount % 2 == 1: # whiteToMove
        string += str(plycount//2 + 1) + ". " + san
        print(string, end="", flush=True)
    else:
        print(" ",san)

def printHistory(fen, san):
    components = fen.split()
    # The side to move is the 2nd component (index 1)
    side_to_move = components[1]
    # The ply count is the last component
    ply_count = int(components[-1])

    string = ""

    for i, s in enumerate(san):
        if side_to_move == 'b':
            string += " " + s
            print(string)
            string = ""
            side_to_move = 'w'
        else: 
            string += str((ply_count+i)//2+1) + ". " + s
            side_to_move = 'b'
    
    if side_to_move == 'b':
        print(str((ply_count+len(san))//2) + ". " + san[-1])
