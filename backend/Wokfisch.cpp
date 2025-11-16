#include "Board.hpp"
#include "Timer.hpp"
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

class Wokfisch {
public:
    // Constructor //
    // TT = 1024**2 * 8 = 192MB
    Wokfisch() : rootBestMove(0), TT(1024*1024*8), quietHistory{0}, killers{0} {
        for (auto& entry : TT) {
            entry = {0, 0, 0, 0, 0}; // Initialize all fields to zero
        }
    }
    ~Wokfisch() {
        TT.clear();
    }

    
    // Variables //
    // Transposition table
    // We store the results of previous searches, keeping track of the score at that position,
    // as well as specific things how it was searched:
    // 1. Did it go through all the search and fail to find a better move? (Upper limit flag)
    // 2. Did it cause a beta cutoff and stopped searching early (Lower limit flag)
    // 3. Did it search through all moves and find a new best move for the currently searched position (Exact flag)
    // Read more about it here: https://www.chessprogramming.org/Transposition_Table
    // Format: Position key, move, depth, score, flag
    struct TranspositionEntry {
        uint64_t positionKey;
        uint16_t move;
        int depth;
        int score;
        uint8_t flag;
    };
    std::vector<TranspositionEntry> TT;
    // Keeping track of which quiet move move is most likely to cause a beta cutoff.
    // The higher the score is, the more likely a beta cutoff is, so in move ordering we will put these moves first.
    
    int64_t quietHistory[4096] = {0};  // At line 41 in WokfischV2.hpp
    // Keep track of killer moves which are so good that they must be considered first
    uint16_t killers[256] = {0};
    // set root best move as Class variable
    uint16_t rootBestMove;


    // Search //
    uint64_t nodesVisited = 0;
    uint16_t returnBestMove(Board& board, Timer& timer, bool verbose=false) {
        // The move that will eventually be reported as our best move
        rootBestMove = 0;
        // Initialize parameters that exist only during one search
        std::fill(std::begin(killers), std::end(killers), 0);
        int allocatedTime = timer.MillisecondsRemaining() / 8;
        int i = 0;
        int score = 0;
        int depth = 1;
        
        // Decay quiet history instead of clearing it.
        for (; i < 4096; ++i) {
            quietHistory[i] /= 8;
        }

        // Reset node counter
        nodesVisited = 0;

        // Get start time
        auto startTime = std::chrono::high_resolution_clock::now();

        // Iterative deepening
        while (timer.MillisecondsElapsedThisTurn() <= allocatedTime / 5 /* Soft time limit */) {
            // Aspiration windows
            int window = 40;
            int alpha;
            int beta;
            while (true) {
                alpha = score - window;
                beta = score + window;
                
                // Search with the current window
                score = negaMax(board, timer, allocatedTime, 0, depth, alpha, beta, false);

                // Hard time limit
                if (timer.MillisecondsElapsedThisTurn() > allocatedTime) {
                    break;
                }
                
                // If the score is within the window, proceed to the next depth
                if (alpha < score && score < beta) {
                    break;
                }
                window *= 2;
            }
            ++depth;
        }

        // Get end time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        // Calculate nodes per second
        double nodesPerSecond = static_cast<double>(nodesVisited) / (duration.count() / 1000.0);
        
        if (verbose){
            std::cout << "Search eval: " << std::endl;
            std::cout << score << " "<< std::fixed << std::setprecision(2) << nodesPerSecond / 1000000 << "M/s"<< std::endl;
        }

        if (rootBestMove == 0) {
            std::cout << "HELP " << std::endl;
        }
        
        return rootBestMove;
    }
    
    int negaMax(Board& board, Timer& timer, int allocatedTime, int ply, int depth, int alpha, int beta, bool nullAllowed) {
        ++nodesVisited;
        
        // Early Termination Checks  
        if (shouldTerminateEarly(board, timer, allocatedTime, depth, nullAllowed)) {
            return handleEarlyTermination(board, ply, depth, alpha, beta);
        }
        
        // Position Setup  
        uint64_t key = board.zobristKey;
        bool inCheck = board.isCheck();
        bool inQsearch = (depth <= 0);
        bool doPruning = (alpha == beta - 1) && !inCheck;
        
        if (inCheck) depth++; // Check extension
        
        int score = 15 + evaluate(board); // Static eval with tempo
        int bestScore = inQsearch ? score : -INF;
        
        // Transposition Table Lookup  
        TranspositionEntry& ttEntry = TT[key % TT.size()];
        uint16_t ttMove = 0; // This will be set by the helper function
        int ttScore = 0;     // This will be set by the helper function on a cutoff
        uint8_t ttFlag = 0;  // Upper bound

        if (handleTranspositionTable(ttEntry, key, depth, alpha, beta, ttScore, ttMove, ply)) {
            // The helper function returned true, indicating a cutoff is possible.
            // It has already calculated the correct score to return (alpha, beta, or exact score).
            return ttScore;
        }
        
        // Internal Iterative Reduction - reduce depth if no TT move found
        if (ttMove == 0 && depth > 3 && !inQsearch) {
            depth--;
        }

        // Quiescence Stand Pat  
        if (inQsearch) {
            if (score >= beta) return score;
            if (score > alpha) alpha = score;
        }
        
        //  Pruning Techniques  
        if (doPruning && !inQsearch) {
            // Reverse Futility Pruning
            if (depth < 7 && score - depth * 75 > beta) {
                return score;
            }
            
            // Null Move Pruning
            if (nullAllowed && depth >= 3 && score >= beta && !board.isEndgame()) {
                // Adaptive reduction based on depth
                int R = 3 + depth / 6;
                
                board.makeNullMove();
                int nullScore = -negaMax(board, timer, allocatedTime, ply + 1, depth - 1 - R, -beta, -beta + 1, false);
                board.unmakeNullMove();
                
                if (nullScore >= beta) {
                    return beta; // Fail-high cutoff
                }
            }
        }
        
        // Main Search Loop
        std::vector<uint16_t> moves = generateAndOrderMoves(board, ttMove, inQsearch, ply);
        std::vector<uint16_t> quietsEvaluated;
        int movesEvaluated = 0;
        int quietsSearched = 0;
        
        for (const auto& move : moves) {
            // Late Move Pruning
            if (doPruning && quietsSearched > 3 + depth * depth) {
                break;
            }
            
            bool isQuiet = !board.isCapture(move);
            
            board.makeMove(move);
            
            // Search Current Move  
            score = searchMove(board, timer, allocatedTime, ply, depth, alpha, beta, 
                              move, movesEvaluated, inQsearch, doPruning, isQuiet);
            
            board.unmakeMove();
            
            // Time check
            if (depth > 2 && timer.MillisecondsElapsedThisTurn() > allocatedTime) {
                return bestScore;
            }
            
            movesEvaluated++;
            
            // Update Best Move  
            if (score > bestScore) {
                bestScore = score;

                if (score > alpha) {
                    ttMove = move;
                    if (ply == 0) rootBestMove = move;
                    alpha = score;
                    ttFlag = 1; // Exact
                    
                    // Beta cutoff
                    if (score >= beta) {
                        ttFlag = 2; // Lower bound
                        if (isQuiet) updateHistoryAndKillers(move, ply, depth, quietsEvaluated);
                        break;
                    }
                }
            }
            
            if (isQuiet) {
                quietsEvaluated.push_back(move);
                quietsSearched++;
            }
        }
        
        // Terminal Position Check  
        if (movesEvaluated == 0) {
            return inQsearch ? bestScore : (inCheck ? -INF/2 + ply : 0); // FIX #5: Correct mate score
        }
        
        // Store in Transposition Table  
        // Adjust mate scores before storing
        int storeScore = bestScore;
        if (bestScore > INF/2 - 1000) {
            storeScore += ply; // Mate score: adjust back to root perspective
        } else if (bestScore < -INF/2 + 1000) {
            storeScore -= ply; // Mated score: adjust back to root perspective
        }
        
        TT[key % TT.size()] = {key, ttMove, inQsearch ? 0 : depth, storeScore, ttFlag};
        
        return bestScore;
    }

    int searchMove(Board& board, Timer& timer, int allocatedTime, int ply, int depth,
        int alpha, int beta, uint16_t move, int movesEvaluated, 
        bool inQsearch, bool doPruning, bool isQuiet) {

        auto defaultSearch = [&](int searchBeta, int reduction = 1, bool nullAllowed = true) {
            return -negaMax(board, timer, allocatedTime, ply + 1, depth - reduction, 
                            -searchBeta, -alpha, nullAllowed);
        };

        // First move or quiescence: full search
        if (inQsearch || movesEvaluated == 0) {
            return defaultSearch(beta);
        }

        // Late Move Reduction conditions
        bool skipLMR = (depth <= 2 || movesEvaluated <= 4 || !isQuiet);

        if (skipLMR) {
            // No LMR, try zero-window search
            int score = defaultSearch(alpha + 1);
            if (score > alpha && score < beta) {
                return defaultSearch(beta); // Re-search with full window
            }
            return score;
        }

        // Apply LMR
        int reduction = 2 + depth / 8 + movesEvaluated / 16 + 
                    static_cast<int>(doPruning) - 
                    compareTo(quietHistory[move & 4095], 0);

        int score = defaultSearch(alpha + 1, reduction);

        // If reduced search fails high, re-search at full depth
        if (score > alpha) {
            score = defaultSearch(alpha + 1);
            
            // If zero-window search fails high, do full window search
            if (score > alpha && score < beta) {
                score = defaultSearch(beta);
            }
        }

        return score;
    }


    // Helper //
    bool shouldTerminateEarly(Board& board, Timer& timer, int allocatedTime, int depth, bool nullAllowed) {
        return nullAllowed && board.isRepeatedPosition(board.zobristKey);
    }
    
    int handleEarlyTermination(Board& board, int ply, int depth, int alpha, int beta) {
        return 0; // Draw by repetition
    }

    bool handleTranspositionTable(TranspositionEntry& ttEntry, uint64_t key, int depth, 
        int alpha, int beta, int& outScore, uint16_t& outTTMove, int ply) {
        // Check for hash miss
        if (ttEntry.positionKey != key) {
            if (depth > 3) depth--;
            return false;
        }

        outTTMove = ttEntry.move;

        // Check if the stored search was deep enough to be useful for a cutoff.
        // If ttEntry.depth < depth, the previous search was shallower and its score is less reliable.
        if (ttEntry.depth < depth) {
            return false;
        }

        if (ply == 0) return false;

        // Adjust stored score for mate distance
        int adjustedScore = ttEntry.score;
        if (adjustedScore > INF/2 - 1000) {
            adjustedScore -= ply;
        } else if (adjustedScore < -INF/2 + 1000) {
            adjustedScore += ply;
        }

        // Determine if we can cause a cutoff based on flag
        uint8_t flag = ttEntry.flag;

        if (flag == 1) { // EXACT
            outScore = adjustedScore;
            return true;
        }

        if (flag == 2) { // LOWER_BOUND
            if (adjustedScore >= beta) {
                outScore = beta;
                return true;
            }
        }

        if (flag == 0) { // UPPER_BOUND
            if (adjustedScore <= alpha) {
                outScore = alpha;
                return true;
            }
        }

        return false;
    }

    void updateHistoryAndKillers(uint16_t move, int ply, int depth, 
        const std::vector<uint16_t>& quietsEvaluated) {
        // Update killer move
        killers[ply] = move;

        // Calculate bonus/penalty (capped at reasonable values)
        int bonus = std::min(depth * depth * 2, 400);

        // Reward the move that caused the cutoff
        quietHistory[move & 4095] += bonus;

        // Penalize moves that were tried but failed
        for (const auto& quietMove : quietsEvaluated) {
        quietHistory[quietMove & 4095] -= bonus;
        }

        // Clamp all history values to prevent overflow
        quietHistory[move & 4095] = std::max<int64_t>(-10000, std::min<int64_t>(10000, quietHistory[move & 4095]));
        for (const auto& quietMove : quietsEvaluated) {
        quietHistory[quietMove & 4095] = std::max<int64_t>(-10000, std::min<int64_t>(10000, quietHistory[quietMove & 4095]));
        }
    }


    // Evaluation //
    inline int evaluate(Board& board, bool verbose=false) {
        int score = 0;
        // Evaluate each piece type
        int gamePhase = calculateGamePhase(board);
            if (verbose) std::cout << "gamePhase: " << gamePhase << std::endl;
        score += evaluatePiece(board.whitePawns, true, 1, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score -= evaluatePiece(board.blackPawns, false, 1, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score += evaluatePiece(board.whiteKnights, true, 2, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score -= evaluatePiece(board.blackKnights, false, 2, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score += evaluatePiece(board.whiteBishops, true, 3, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score -= evaluatePiece(board.blackBishops, false, 3, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score += evaluatePiece(board.whiteRooks, true, 4, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score -= evaluatePiece(board.blackRooks, false, 4, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score += evaluatePiece(board.whiteQueens, true, 5, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score -= evaluatePiece(board.blackQueens, false, 5, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score += evaluatePiece(board.whiteKing, true, 6, gamePhase);
            if (verbose) std::cout << score << std::endl;
        score -= evaluatePiece(board.blackKing, false, 6, gamePhase);
            if (verbose) std::cout << score << std::endl;

        // score += kingDistanceScore(board.whiteKing, board.blackKing, board.whiteToMove, gamePhase);

        return board.whiteToMove ? score : -score;
    }
    
    inline int evaluatePiece(uint64_t& board, bool whiteToMove, int pieceIndex, int gamePhase) {
        int mg_val = piece_data[pieceIndex].mg_value;
        int eg_val = piece_data[pieceIndex].eg_value;
        const int* mg_table = piece_data[pieceIndex].mg_table;
        const int* eg_table = piece_data[pieceIndex].eg_table;
        int score = 0;
        uint64_t pieceBoard;
        if (!whiteToMove){
            pieceBoard = flipVertical(board); // flip board if black
        } else{
            pieceBoard = board;
        }

        while (pieceBoard) {
            int x = ctz64(pieceBoard);
            int sq = abs(x/8 - 7) * 8 + x%8; // get square to index into piecetables

            // calculate position scores
            score += (mg_val + mg_table[sq]) * gamePhase / 32;
            score += (eg_val + eg_table[sq]) * (32 - gamePhase) / 32;

            // calculate mobility scores


            pieceBoard &= pieceBoard - 1;
        }
        return score;
    }

    inline int calculateGamePhase(const Board& board) {
        // Simple game phase calculation based on remaining material
        int gamePhase = 0;

        gamePhase += popcount64(board.whiteBishops | board.whiteKnights | board.blackBishops | board.blackKnights) * 300;
        gamePhase += popcount64(board.whiteRooks | board.blackRooks) * 500;
        gamePhase += popcount64(board.whiteQueens | board.blackQueens) * 900;

        // Normalize gamePhase to a value between 0 and 32
        const int offset = 1000; // how fast will the endgame be reached?
        const int maxPhase = 6200 - offset; // Adjust as needed based on typical game states
        int state = ((gamePhase - offset) * 32) / maxPhase;
        if (state > 0){return state;}
        else{return 0;}
    }

    inline int kingDistanceScore(uint64_t& whiteKing, uint64_t& blackKing, bool whiteToMove, int gamePhase){
        if (gamePhase < 10){
            uint8_t whiteKingSquare = ctz64(whiteKing);
            uint8_t blackKingSquare = ctz64(blackKing);

            // Calculate Chebyshev distance (maximum of file and rank differences)
            uint8_t fileDiff = abs((whiteKingSquare % 8) - (blackKingSquare % 8));
            uint8_t rankDiff = abs((whiteKingSquare / 8) - (blackKingSquare / 8));
            uint8_t distance = std::max(fileDiff, rankDiff);

            // slowly fade in gamephase
            int kingDistanceScore = (whiteToMove ? 1 : -1) * (8 - distance) * (10-gamePhase) * 3;
            // std::cout << "kingd score: " << kingDistanceScore << std::endl;
            return kingDistanceScore;
        } else{
            return 0;
        }
    }


    // Utility //
    int compareTo(int value, int comparedTo) {
        if (value < comparedTo) return -1;
        if (value > comparedTo) return 1;
        return 0;
    }
    inline uint64_t flipVertical(uint64_t x) {
        return  ( (x << 56)                           ) |
                ( (x << 40) & 0x00ff000000000000ULL ) |
                ( (x << 24) & 0x0000ff0000000000ULL ) |
                ( (x <<  8) & 0x000000ff00000000ULL ) |
                ( (x >>  8) & 0x00000000ff000000ULL ) |
                ( (x >> 24) & 0x0000000000ff0000ULL ) |
                ( (x >> 40) & 0x000000000000ff00ULL ) |
                ( (x >> 56) );
    }
    void printNonZeroEntries(const std::vector<TranspositionEntry>& TT) {
        int nonZeroCount = 0;
        
        for (const auto& entry : TT) {
            if (entry.positionKey != 0 || entry.move != 0 || 
                entry.depth != 0 || entry.score != 0 || entry.flag != 0) {
                nonZeroCount++;
            }
        }
        
        std::cout << "Number of non-zero entries: " << nonZeroCount << std::endl;
    }
    std::vector<uint16_t> generateAndOrderMoves(Board& board, const uint16_t& ttMove, bool inQsearch, int ply) {
        std::vector<uint16_t> moves = board.generateAllLegalMoves();
        
        if (inQsearch) {
            // In quiescence, only search captures
            std::vector<uint16_t> captures;
            captures.reserve(moves.size());
            for (const auto& move : moves) {
                if (board.getPieceOfSquare(board.getTo(move)) != 0) {
                    captures.push_back(move);
                }
            }
            moves = std::move(captures);
        }
        
        // Sort using the scoring function
        std::sort(moves.begin(), moves.end(), [this, &board, &ttMove, ply](const uint16_t& a, const uint16_t& b) {
            return getMoveScore(board, a, ttMove, ply) > getMoveScore(board, b, ttMove, ply);
        });
        
        return moves;
    }
    
    int64_t getMoveScore(Board& board, const uint16_t& move, const uint16_t& ttMove, int ply) {
        // Priority 1: Hash move (must be first)
        if (move == ttMove) {
            return 10000000;
        }
        
        uint8_t capturedPiece = board.getPieceOfSquare(board.getTo(move));
        uint8_t movingPiece = board.getPieceOfSquare(board.getFrom(move));
        
        // Priority 2 & 6: Captures (good and bad)
        if (capturedPiece != 0) {
            // MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            // Multiply victim value by 10, subtract attacker value
            // This naturally separates good captures (positive SEE) from bad ones
            int captureScore = capturedPiece * 10 - movingPiece;
            
            // Good captures: score 1,000,000 to 9,000,000
            // Bad captures: score -9,000,000 to -1,000,000
            return 1000000 + captureScore * 100000;
        }
        
        // Check for promotions (Priority 3)
        // Assuming you have a way to detect promotions - adjust based on your move encoding
        uint8_t promotionPiece = (move >> 12) & 0xF; // Example: if you store promo in bits 12-15
        if (promotionPiece != 0) {
            return 900000 + promotionPiece * 10000; // Queen promo = 950000, etc.
        }
        
        // Priority 4: Killer moves
        if (move == killers[ply]) {
            return 800000;
        }
        
        // Priority 5: History heuristic for quiet moves
        // History scores should be in range [-10000, 10000] after clamping
        // This gives us range [0, 10000] which is well below killer moves
        int history = quietHistory[move & 4095];
        return history;
    }

    
private:
    // Piece square Tables //
    struct PieceData {
        int mg_value;           // Middle-game value
        int eg_value;           // End-game value
        const int* mg_table;    // Pointer to middle-game piece-square table
        const int* eg_table;    // Pointer to end-game piece-square table
    };

    const int mg_pawn_table[64] = {
        0,   0,   0,   0,   0,   0,  0,   0,
        98, 134,  61,  95,  68, 126, 34, -11,
        -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
        0,   0,   0,   0,   0,   0,  0,   0,
    };
    const int eg_pawn_table[64] = {
        0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100,  85,  67,  56,  53,  82,  84,
        32,  24,  13,   5,  -2,   4,  17,  17,
        13,   9,  -3,  -7,  -7,  -8,   3,  -1,
        4,   7,  -6,   1,   0,  -5,  -1,  -8,
        13,   8,   8,  10,  13,   0,   2,  -7,
        0,   0,   0,   0,   0,   0,   0,   0,
    };
    const int mg_knight_table[64] = {
        -167, -89, -34, -49,  61, -97, -15, -107,
        -73, -41,  72,  36,  23,  62,   7,  -17,
        -47,  60,  37,  65,  84, 129,  73,   44,
        -9,  17,  19,  53,  37,  69,  18,   22,
        -13,   4,  16,  13,  28,  19,  21,   -8,
        -23,  -9,  12,  10,  19,  17,  25,  -16,
        -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    };
    const int eg_knight_table[64] = {
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    };
    const int mg_bishop_table[64] = {
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
        0,  15,  15,  15,  14,  27,  18,  10,
        4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    };
    const int eg_bishop_table[64] = {
        -14, -21, -11,  -8, -7,  -9, -17, -24,
        -8,  -4,   7, -12, -3, -13,  -4, -14,
        2,  -8,   0,  -1, -2,   6,   0,   4,
        -3,   9,  12,   9, 14,  10,   3,   2,
        -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    };
    const int mg_rook_table[64] = {
        32,  42,  32,  51, 63,  9,  31,  43,
        27,  32,  58,  62, 80, 67,  26,  44,
        -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    };
    const int eg_rook_table[64] = {
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
        7,  7,  7,  5,  4,  -3,  -5,  -3,
        4,  3, 13,  1,  2,   1,  -1,   2,
        3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    };
    const int mg_queen_table[64] = {
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9,  10, -15, -25, -31, -50,
    };
    const int eg_queen_table[64] = {
        -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
        3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    };
    const int mg_king_table[64] = {
        -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -80,  40,  14,
    };
    const int eg_king_table[64] = {  
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
        10,  17,  23,  15,  20,  45,  44,  13,
        -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43
    };

    PieceData piece_data[7] = {
        {0, 0, nullptr, nullptr},
        {82, 94, mg_pawn_table, eg_pawn_table},     // Pawn
        {337, 281, mg_knight_table, eg_knight_table}, // Knight
        {365, 297, mg_bishop_table, eg_bishop_table}, // Bishop
        {477, 512, mg_rook_table, eg_rook_table},   // Rook
        {1025, 936, mg_queen_table, eg_queen_table}, // Queen
        {0, 0, mg_king_table, eg_king_table}                   // King (assume no table for the king here)
    };

    const int INF = std::numeric_limits<int>::max()-1;
};

PYBIND11_MODULE(Wokfisch, module_handle) {
  module_handle.doc() = "I'm a docstring hehe";

  py::class_<Wokfisch>(module_handle, "Wokfisch")
  .def(py::init<>())
  .def("returnBestMove", &Wokfisch::returnBestMove)
  .def("calculateGamePhase", &Wokfisch::calculateGamePhase)
  .def("evaluate", &Wokfisch::evaluate);
}
