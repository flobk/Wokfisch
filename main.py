from frontend import *
# Pc
from backend.buildPc.Timer import Timer # type: ignore
from backend.buildPc.Board import Board # type: ignore
from backend.buildPc.Wokfisch import Wokfisch # type: ignore
# from backend.buildPc.Board_qNNUE import Board_qNNUE # type: ignore
# from backend.buildPc.Wokfisch_qNNUE import Wokfisch_qNNUE # type: ignore
# Mac
# from backend.buildMac.Timer import Timer # type: ignore
# from backend.buildMac.Board import Board # type: ignore
# from backend.buildMac.Wokfisch import Wokfisch # type: ignore
# from backend.buildMac.Board_qNNUE import Board_qNNUE # type: ignore
# from backend.buildMac.Wokfisch_qNNUE import Wokfisch_qNNUE # type: ignore
import time

def main():
    # Init Engine
    player1 = Wokfisch()
    player2 = Wokfisch()
    # player1 = Wokfisch_qNNUE()
    # player2 = Wokfisch_qNNUE()
    verbose = False

    # Initialize Players
    computerIsWhite = False
    computerIsBlack = True
    # fen = "3r1rk1/1b2b1pp/p3p3/R3Np2/2pB1P2/2P3qP/P1PQ2P1/5RK1 b - - 0 1"
    # fen = "rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2P2N2/PP1P1PPP/RNBQKB1R w KQkq - 1 4"
    # fen = "r2q3r/pb3k2/3ppn1p/n1p2pp1/2P5/P1PBPNB1/5PPP/RQ3RK1 w - - 2 16"
    fen = None
    while computerIsWhite is None:
        whitePlayerInput = input("Computer is white? (y/n) ")
        if whitePlayerInput == "y":
            computerIsWhite = True
        elif whitePlayerInput == "n":
            computerIsWhite = False
        else:
            print("Invalid input")
    while computerIsBlack is None:
        blackPlayerInput = input("Computer is black? (y/n) ")
        if blackPlayerInput == "y":
            computerIsBlack = True
        elif blackPlayerInput == "n":
            computerIsBlack = False
        else:
            print("Invalid input")

    # Initialize Board
    if fen is None:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = Board(str(fen))
        # board = Board_qNNUE(str(fen))
    else:
        # fen = input("FEN: ")
        board = Board(str(fen))
        # board = Board_qNNUE(str(fen))

    # Initialize Timer
    game_duration = 60000
    timer1 = Timer(game_duration)
    timer2 = Timer(game_duration)

    # Move history
    san = []
    plycount2 = int(fen.split()[-1])

    # Print Config
    print("-"*25, "\nComputer is white: ", computerIsWhite)
    print("Computer is black: ", computerIsBlack)
    print("Fen: ", fen)
    print("-"*25)

    # Initialize pygame
    clock, running, selected_square, gameEnd, whiteToMove = pygame.time.Clock(), True, None, False, True
    images, sounds = load_images(), load_sounds()
    draw_background(BACKGROUND, images)
    draw_pieces(PIECES, board, images)
    sounds["start"].play()
    
    pygame.init()
    last_interaction_time = time.time()
    idle_threshold = 0.2  # Time in seconds before entering idle state
    
    while running:
        if computerIsWhite and whiteToMove and not gameEnd:
            # Add small time delay (for closing and sound)
            start_time = time.time()
            while time.time() - start_time < 0.5:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            # Execute Move
            timer1.StartTurn()
            move = player1.returnBestMove(board, timer1, verbose)
            timer1.EndTurn()
            fromPiece, toPiece = board.getPieceOfSquare(get_from(move)), board.getPieceOfSquare(get_to(move))
            piecesAttacking = get_pieces_attacking(get_to(move), fromPiece, board)
            board.makeMove(move)
            san.append(toSAN(move, fromPiece, toPiece, piecesAttacking, board))
            # Display current move
            printCurrentPGN(san[-1], plycount2)
            # print(board.getPositionCount(board.getZobristKey()))
            play_sound(move, fromPiece, toPiece, board, sounds)
            draw_make_move(BACKGROUND, PIECES, board, move, board.getLastMove(), images)
            # Decide on Game End
            if board.isCheckmate():
                printCheckmate(board, sounds, fen, san, whiteToMove)
                gameEnd = True
            elif board.isDraw():
                printDraw(board, sounds, fen, san, whiteToMove)
                gameEnd = True
            # Change gamestate and displaystate
            whiteToMove = not whiteToMove
            plycount2 += 1
            WINDOW.blit(BACKGROUND, (0, 0))
            WINDOW.blit(PIECES, (0, 0))
            WINDOW.blit(MARKERS, (0, 0))
            pygame.display.flip()
            continue
        elif computerIsBlack and not whiteToMove and not gameEnd:
            # Add small time delay (for closing and sound)
            start_time = time.time()
            while time.time() - start_time < 0.5:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            # Execute Move
            timer2.StartTurn()
            move = player2.returnBestMove(board, timer2, verbose)
            timer2.EndTurn()
            fromPiece, toPiece = board.getPieceOfSquare(get_from(move)), board.getPieceOfSquare(get_to(move))
            piecesAttacking = get_pieces_attacking(get_to(move), fromPiece, board)
            board.makeMove(move)
            san.append(toSAN(move, fromPiece, toPiece, piecesAttacking, board))
            # Display current move
            printCurrentPGN(san[-1], plycount2)
            # print(board.getPositionCount(board.getZobristKey()))
            play_sound(move, fromPiece, toPiece, board, sounds)
            draw_make_move(BACKGROUND, PIECES, board, move, board.getLastMove(), images)
            # Decide on Game End
            if board.isCheckmate():
                printCheckmate(board, sounds, fen, san, whiteToMove)
                gameEnd = True
            elif board.isDraw():
                printDraw(board, sounds, fen, san, whiteToMove)
                gameEnd = True
            # Change gamestate and displaystate
            whiteToMove = not whiteToMove
            plycount2 += 1
            WINDOW.blit(BACKGROUND, (0, 0))
            WINDOW.blit(PIECES, (0, 0))
            WINDOW.blit(MARKERS, (0, 0))
            pygame.display.flip()
            continue
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            elif gameEnd == True: time.sleep(0.05)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                last_interaction_time = time.time()
                col = event.pos[0] // SQUARE_SIZE
                row = abs(event.pos[1] // SQUARE_SIZE - 7)
                square = row*8+col

                if selected_square is None: # Select piece
                    # only select if a piece is on the square and it's the right color
                    if board.getPieceOfSquare(square) != 0 and board.rightColor(square):
                        selected_square = square # Select square
                        draw_select_piece(BACKGROUND, MARKERS, selected_square, board.generateLegalMovesOfSquare(selected_square), images)

                else: # Make a move
                    draw_deselect_piece(BACKGROUND, selected_square)
                    move = board.generateMove(selected_square, square) # from, to
                    # Check promotion flag
                    if move & (1 << 15):
                        piece = None
                        print("Promotion")
                        while piece not in ["q", "r", "b", "n"]:
                            piece = input("Enter promotion piece (q, r, b, n): ")
                        promotion_pieces = {"q": 3, "r": 2, "b": 1, "n": 0}
                        if piece in promotion_pieces:
                            move = (move & ~(3 << 13)) | (promotion_pieces[piece] << 13)

                    # Only make the move if it is legal
                    # if True:
                    if move in board.generateLegalMovesOfSquare(selected_square):
                        # Execute move
                        fromPiece, toPiece = board.getPieceOfSquare(get_from(move)), board.getPieceOfSquare(get_to(move))
                        piecesAttacking = get_pieces_attacking(get_to(move), fromPiece, board)
                        board.makeMove(move)
                        san.append(toSAN(move, fromPiece, toPiece, piecesAttacking, board))
                        # Display current move
                        printCurrentPGN(san[-1], plycount2)
                        # print(board.getPositionCount(board.getZobristKey()))
                        play_sound(move, fromPiece, toPiece, board, sounds)
                        draw_make_move(BACKGROUND, PIECES, board, move, board.getLastMove(), images)
                        selected_square = None

                        # Decide on Game End
                        if board.isCheckmate():
                            printCheckmate(board, sounds, fen, san, whiteToMove)
                            gameEnd = True
                            WINDOW.blit(BACKGROUND, (0, 0))
                            WINDOW.blit(PIECES, (0, 0))
                            WINDOW.blit(MARKERS, (0, 0))
                            pygame.display.flip()
                        elif board.isDraw():
                            printDraw(board, sounds, fen, san, whiteToMove)
                            gameEnd = True
                            WINDOW.blit(BACKGROUND, (0, 0))
                            WINDOW.blit(PIECES, (0, 0))
                            WINDOW.blit(MARKERS, (0, 0))
                            pygame.display.flip()
                        whiteToMove = not whiteToMove
                        plycount2 += 1
                    
                    elif (selected_square != square): sounds["illegal"].play()
                    # Deselect the piece and clear selectscreen
                    MARKERS.fill((0, 0, 0, 0))
                    selected_square = None 

        current_time = time.time()
        if current_time - last_interaction_time < idle_threshold:
            # Only update and redraw when not idle
            WINDOW.blit(BACKGROUND, (0, 0))
            WINDOW.blit(PIECES, (0, 0))
            WINDOW.blit(MARKERS, (0, 0))
            pygame.display.flip()
        else: time.sleep(0.05) # In idle state, sleep to reduce CPU usage
        clock.tick(30)
    pygame.quit()

if __name__ == "__main__": main()