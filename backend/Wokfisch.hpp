#pragma once
#include "Board.hpp"
#include "Timer.hpp"
#include <chrono>


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
    uint64_t quietHistory[4096] = {0};
    // Keep track of killer moves which are so good that they must be considered first
    uint16_t killers[256] = {0};
    // set root best move as Class variable
    uint16_t rootBestMove;


    // Search //
    uint64_t nodesVisited = 0;
    uint16_t returnBestMove(Board board, Timer timer, bool verbose=false) {
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
    int negaMax(Board& board, Timer& timer, int allocatedTime, int ply, int depth, int alpha, int beta, bool nullAllowed){
        ++nodesVisited;
        // Repetition detection
        // There is no need to check for 3-fold repetition, if a single repetition (0 = draw) ends up being the best,
        // we can trust that repeating moves is the best course of action in this position.
        uint64_t key = board.zobristKey;
        if (nullAllowed && board.isRepeatedPosition(key)){
            return 0;
        }
        // Check extension: if we are in check, we should search deeper. More info: https://www.chessprogramming.org/Check_Extensions
        bool inCheck = board.isCheck();
        if (inCheck)
            depth++;
        
        // In-qsearch is a flag that determines whether not we should prune positions ans whether or not to search non-captures.
        // Qsearch, also meaning quiescence search, is a mode that only looks at captures in order to give a more accurate
        // estimate "if all the viable captures happen". In this engine it is interlaced with the main search to save tokens, although
        // in most engines you will see a separate function for it.
        // Tempo is the idea that each move is benefitial to us, so we adjust the static eval using a fixed value.
        // We use 15 tempo for evaluation for mid-game, 0 for end-game.    
        bool inQsearch = (depth <= 0);
        int bestScore = -INF;
        bool doPruning = alpha == beta - 1 && !inCheck;
        int score = 15;
        int phase = 0;

        // Evaluate
        score += evaluate(board);

        // Local method for similar calls to Search, inspired by Tyrant7's approach here: https://github.com/Tyrant7/Chess-Challenge
        // We keep known values, but we create a local method that will be used to implement 3-fold PVS. More on that later on
        auto defaultSearch = [&](int beta, int reduction = 1, bool nullAllowed = true) {
            return -negaMax(board, timer, allocatedTime, ply + 1, depth - reduction, -beta, -alpha, nullAllowed); // Return the score for consistency
        };

        // Transposition table lookup
        // Look up best move known so far if it is available
        TranspositionEntry& ttEntry = TT[key % TT.size()];
        uint64_t ttKey = ttEntry.positionKey;
        uint16_t ttMove = ttEntry.move;
        int ttDepth = ttEntry.depth;
        int ttScore = ttEntry.score;
        uint8_t ttFlag = ttEntry.flag;

        if (ttKey == key){
            // If conditions match, we can trust the table entry and return immediately.
            // This is a token optimized way to express that: we can trust the score stored in TT and return immediately if:
            // 1. The depth remaining is higher or equal to our current
            //   a. Either the flag is exact, or:
            //   b. The stored score has an upper bound, but we scored below the stored score, or:
            //   c. The stored score has a lower bound, but we scored above the scored score
            if (alpha == beta - 1 && ttDepth >= depth && ttFlag != (ttScore >= beta ? 0 : 2)){
                // std::cout << ttScore << std::endl;
                return ttScore;
            }

            // ttScore can be used as a better positional evaluation
            // If the score is outside what the current bounds are, but it did match flag and depth,
            // then we can trust that this score is more accurate than the current static evaluation,
            // and we can update our static evaluation for better accuracy in pruning
            if (ttFlag != (ttScore > score ? 0 : 2))
                score = ttScore;
        }

        // Internal iterative reductions
        // If this is the first time we visit this node, it might not be worth searching it fully
        // because it might be a random non-promising node. If it gets visited a second time, it's worth fully looking into.
        else if (depth > 3)
            depth--;

        // We look at if it's worth capturing further based on the static evaluation
        if (inQsearch){
            if (score >= beta)
                return score;

            if (score > alpha)
                alpha = score;

            bestScore = score;
        }

        else if (doPruning){
            // Reverse futility pruning
            // If our current score is way above beta, depending on the score, we can use this as a heuristic to not look
            // at shallow-ish moves in the current position, because they are likely to be countered by the opponent.
            // More info: https://www.chessprogramming.org/Reverse_Futility_Pruning
            if (depth < 7 && score - depth * 75 > beta)
                return score;

            // Null move pruning
            // The idea is that each move in a chess engine brings some advantage. If we skip our own move, do a search with reduced depth,
            // and our position is still so winning that the opponent can't refute it, we claim that this is too good to be true,
            // and we discard this move. An important observation is the `phase != 0` term, which checks if all remaining
            // pieces are pawns/kings, this reduces the cases of mis-evaluations of zugzwang in the end-game.
            // More info: https://www.chessprogramming.org/Null_Move_Pruning
            if (nullAllowed && score >= beta && depth > 2 && phase != 0){
                board.whiteToMove = !board.whiteToMove; // MIGHT NEED IMPROVEMENT
                defaultSearch(beta, 4 + depth / 6, false);
                board.whiteToMove = !board.whiteToMove;
                if (score >= beta)
                    return beta;
            }
        }

        std::vector<uint16_t> moves = generateAndOrderMoves(board, ttMove, inQsearch, ply);

        std::vector<uint16_t> quietsEvaluated;
        int movesEvaluated = 0;
        ttFlag = 0; // Upper
        
        for (const auto& move : moves) {
            // A quiet move traditionally means a move that doesn't cause a capture to be the best move,
            // is not a promotion, and doesn't give check. For token savings we only consider captures.
            bool isQuiet = board.getPieceOfSquare(board.getTo(move));

            board.makeMove(move);

            // Principal variation search
            // We trust that our move ordering is good enough to ensure the first move searched to be the best move most of the time,
            // so we only search the first move fully and all following moves with a zero width window (beta = alpha + 1).
            // More info: https://en.wikipedia.org/wiki/Principal_variation_search

            // Late move reduction
            // As the search deepens, looking at each move costs more and more. Since we have some other heuristics,
            // like the move score quiet moves, as well as some other facts like whether or not this move is a capture,
            // we can search shallower for not promising moves, most of which came later at our move ordering.
            // More info: https://www.chessprogramming.org/Late_Move_Reductions
            if (inQsearch || movesEvaluated == 0 // No PVS for first move or qsearch
                || (depth <= 2 || movesEvaluated <= 4 || !isQuiet // Conditions not to do LMR
                // || defaultSearch(alpha + 1, depth / 2) > alpha)
                || defaultSearch(alpha + 1, 2 + depth / 8 + movesEvaluated / 16 + static_cast<int>(doPruning) - compareTo(quietHistory[move & 4095], 0)) > alpha)
                && alpha < defaultSearch(alpha + 1) && score < beta){ // Full depth search failed high
                score = defaultSearch(beta); // Do full window search
            }

            board.unmakeMove();

            // If we are out of time, stop searching
            if (depth > 2 && timer.MillisecondsElapsedThisTurn() > allocatedTime){
                return bestScore;
            }

            // Count the number of moves we have evaluated for detecting mates and stalemates
            movesEvaluated++;

            // If the move is better than our current best, update our best score
            if (score > bestScore){
                bestScore = score;
                
                // If the move is better than our current alpha, update alpha and our best move
                if (score > alpha){
                    ttMove = move;
                    if (ply == 0) {
                        rootBestMove = move;
                    }
                    alpha = score;
                    ttFlag = 1; // Exact
                    
                    // If the move is better than our current beta, we can stop searching
                    if (score >= beta){

                        ttFlag++; // Lower
                        break;
                    }
                }
            }

            if (isQuiet){
                quietsEvaluated.emplace_back(move);
            }

            // Late move pruning
            if (doPruning && quietsEvaluated.size() > 3 + depth * depth)
                // std::cout << "LMP" << std::endl;
                break;
        }
        
        // Checkmate / stalemate detection
        // 1000000 = mate score
        if (movesEvaluated == 0)
            return inQsearch ? bestScore : (inCheck ? ply - INF/2 : 0);

        // // Store the current position in the transposition table
        TT[key % TT.size()] = {key, ttMove, inQsearch ? 0 : depth, bestScore, ttFlag};
        
        return bestScore;
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

        gamePhase += popcount64(board.whiteBishops | board.whiteKnights | board.blackBishops | board. blackKnights) * 300;
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


    // Helper functions //
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
        
        if (inQsearch) { // filter out captures
            std::vector<uint16_t> nonQuietMoves;
            for (const auto& move : moves) {
                if (board.getPieceOfSquare(board.getTo(move)) != 0) {
                    nonQuietMoves.push_back(move);
                }
            }
            moves = nonQuietMoves;
        }
        
        std::sort(moves.begin(), moves.end(), [this, &board, &ttMove, ply](const uint16_t& a, const uint16_t& b) {
            return getMoveScore(board, a, ttMove, ply) > getMoveScore(board, b, ttMove, ply);
        });
        
        return moves;
    }
    int64_t getMoveScore(Board& board, const uint16_t& move, const uint16_t& ttMove, int ply) {
        if (move == ttMove) {
            return 9000000000000000LL;
        }
        uint8_t capturePiece = board.getPieceOfSquare(board.getTo(move));
        if (capturePiece) {
            return 1000000000000000LL * static_cast<int64_t>(capturePiece) - static_cast<int64_t>(board.getPieceOfSquare(board.getFrom(move)));
        }
        if (move == killers[ply]) {
            return 500000000000000LL;
        }
        return quietHistory[move & 4095];
    }
    

private:
    // Piece square Tables //
    struct PieceData {
        int mg_value;           // Middle-game value
        int eg_value;           // End-game value
        const int* mg_table;    // Pointer to middle-game piece-square table
        const int* eg_table;    // Pointer to end-game piece-square table
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

    const int INF = std::numeric_limits<int>::max()-1;
};
