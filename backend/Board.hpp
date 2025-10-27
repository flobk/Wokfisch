#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <random>
#include <iomanip>
#include <bitset>
#include "MoveMap.hpp"
#include "constants.hpp"

#ifdef _MSC_VER
    #include <intrin.h>
    
    // MSVC implementation of count trailing zeros
    inline int ctz64(uint64_t x) {
        unsigned long index;
        _BitScanForward64(&index, x);
        return static_cast<int>(index);
    }
    
    // MSVC implementation of population count (count set bits)
    inline int popcount64(uint64_t x) {
        return static_cast<int>(__popcnt64(x));
    }
#else
    // GCC/Clang implementation
    inline int ctz64(uint64_t x) {
        return __builtin_ctzll(x);
    }
    
    inline int popcount64(uint64_t x) {
        return __builtin_popcountll(x);
    }
#endif

class Board{

public:
    // Variables //
    // Bitboards
    uint64_t whitePawns, blackPawns;
    uint64_t whiteKnights, blackKnights;
    uint64_t whiteBishops, blackBishops;
    uint64_t whiteRooks, blackRooks;
    uint64_t whiteQueens, blackQueens;
    uint64_t whiteKing, blackKing;
    uint64_t whitePieces;
    uint64_t blackPieces;
    uint64_t allOccupied;

    // Game State Information
    uint16_t plycount; // The current "time"
    bool whiteToMove;
    int16_t fullmoveNumber;
    uint64_t zobristKey;

    // Game History Information
    uint16_t moveHistory[1000] = {0}; // Stores the moves in an arr
    uint8_t capturedPieceHistory[1000] = {0}; // Stores the captured pieces in a vector. A piece can also be 0.
    uint8_t enPassantFileHistory[1000] = {0}; // Stores the enPassant values in an arr
    uint8_t castlingRightHistory[1000] = {0}; // Stores the castling rights in an arr
    uint16_t halfmoveClockHistory[1000] = {0}; // Stores the halfMoveValues in an arr
    uint64_t zobristKeyHistory[1000] = {0}; // Stores the zobristKey in an arr, for threefold repetition
    std::unordered_map<uint64_t, int> positionHashHistory; // Stores the hashes with the # of times it occured
    
    // Zobrist hash values
    uint64_t pieceHash[12][64]; // 12 types of pieces (6 white + 6 black) and 64 squares
    uint64_t whiteToMoveHash;  // 1 element for current side
    uint64_t castlingHash[16]; // 16 possible castling rights (4 bits)
    uint64_t enPassantHash[8]; // 8 possible en passant files
    
    // Constructors and Initialization
    Board(std::string fen = "") {
        initializeZobristHashes(); // Initialize Zobrist hashe values when the board is created
        if (fen.empty()) {
            reset();
        } else {
            FENtoBoard(fen);
        }
    }
    ~Board() {
        // Clear hash map
        positionHashHistory.clear();
    }
    void reset(){
        // Reset all bitboards to 0
        whitePawns = blackPawns = 0ULL;
        whiteKnights = blackKnights = 0ULL;
        whiteBishops = blackBishops = 0ULL;
        whiteRooks = blackRooks = 0ULL;
        whiteQueens = blackQueens = 0ULL;
        whiteKing = blackKing = 0ULL;

        // Set up white pieces
        whitePawns = 0x000000000000FF00ULL;
        whiteKnights = 0x0000000000000042ULL;
        whiteBishops = 0x0000000000000024ULL;
        whiteRooks = 0x0000000000000081ULL;
        whiteQueens = 0x0000000000000008ULL;
        whiteKing = 0x0000000000000010ULL;

        // Set up black pieces
        blackPawns = 0x00FF000000000000ULL;
        blackKnights = 0x4200000000000000ULL;
        blackBishops = 0x2400000000000000ULL;
        blackRooks = 0x8100000000000000ULL;
        blackQueens = 0x0800000000000000ULL;
        blackKing = 0x1000000000000000ULL;

        // Set up color bitboards and all occupied squares
        whitePieces = whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
        blackPieces = blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
        allOccupied = whitePieces | blackPieces;

        // reset game state information
        whiteToMove = true;
        fullmoveNumber = 1; // standard is 1

        // reset game history information
        plycount = 0;
        std::memset(moveHistory, 0, sizeof(moveHistory));
        std::memset(capturedPieceHistory, 0, sizeof(capturedPieceHistory));
        std::memset(enPassantFileHistory, 0, sizeof(enPassantFileHistory));
        std::memset(castlingRightHistory, 0, sizeof(castlingRightHistory));
        std::memset(halfmoveClockHistory, 0, sizeof(halfmoveClockHistory));
        std::memset(zobristKeyHistory, 0, sizeof(zobristKeyHistory));
        zobristKey = getZobristKey(); // get Key with current init
        zobristKeyHistory[plycount] = zobristKey;
        enPassantFileHistory[0] = 0xFF; // set all bits -> mean no en passant file
        castlingRightHistory[0] = 0xF; // set all castling rights true
        // positionHashHistory.clear();
    }
    void FENtoBoard(std::string fen) {
        // reset Board
        emptyBoard();
        // reset game history information
        std::memset(moveHistory, 0, sizeof(moveHistory));
        std::memset(capturedPieceHistory, 0, sizeof(capturedPieceHistory));
        std::memset(enPassantFileHistory, 0xFF, sizeof(enPassantFileHistory));
        std::memset(castlingRightHistory, 0, sizeof(castlingRightHistory));
        std::memset(halfmoveClockHistory, 0, sizeof(halfmoveClockHistory));
        std::memset(zobristKeyHistory, 0, sizeof(zobristKeyHistory));
        
        std::istringstream ss(fen);
        std::string boardPos, activeColor, castling, enPassant, halfmoveClock, fullmoveN;
        ss >> boardPos >> activeColor >> castling >> enPassant >> halfmoveClock >> fullmoveN;

        // Parse board position
        int rank = 7, file = 0;
        for (char c : boardPos) {
            if (c == '/') {
                rank--;
                file = 0;
            } else if (std::isdigit(c)) {
                file += c - '0';
            } else {
                uint64_t square = 1ULL << (rank * 8 + file);
                switch (c) {
                    case 'P': whitePawns |= square; break;
                    case 'p': blackPawns |= square; break;
                    case 'N': whiteKnights |= square; break;
                    case 'n': blackKnights |= square; break;
                    case 'B': whiteBishops |= square; break;
                    case 'b': blackBishops |= square; break;
                    case 'R': whiteRooks |= square; break;
                    case 'r': blackRooks |= square; break;
                    case 'Q': whiteQueens |= square; break;
                    case 'q': blackQueens |= square; break;
                    case 'K': whiteKing |= square; break;
                    case 'k': blackKing |= square; break;
                }
                file++;
            }
        }

        // Set color bitboards
        whitePieces = whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
        blackPieces = blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
        allOccupied = whitePieces | blackPieces;

        // Set game state information
        whiteToMove = (activeColor == "w");
        plycount = (whiteToMove) ? 0 : 1;
        plycount += (std::stoi(fullmoveN)-1) * 2;

        uint8_t castlingRights = 0;
        if (castling.find('K') != std::string::npos) castlingRights |= 8;
        if (castling.find('Q') != std::string::npos) castlingRights |= 4;
        if (castling.find('k') != std::string::npos) castlingRights |= 2;
        if (castling.find('q') != std::string::npos) castlingRights |= 1;
        castlingRightHistory[plycount] = castlingRights;

        uint8_t enPassantFile = 0xFF; // 255 is default
        switch (enPassant[0]){
            case 'a': enPassantFile = 0x0; break;
            case 'b': enPassantFile = 0x1; break;
            case 'c': enPassantFile = 0x2; break;
            case 'd': enPassantFile = 0x3; break;
            case 'e': enPassantFile = 0x4; break;
            case 'f': enPassantFile = 0x5; break;
            case 'g': enPassantFile = 0x6; break;
            case 'h': enPassantFile = 0x7; break;
        }
        enPassantFileHistory[plycount] = enPassantFile;

        halfmoveClockHistory[plycount] = std::stoi(halfmoveClock);
        fullmoveNumber = std::stoi(fullmoveN);
        zobristKey = getZobristKey();
        zobristKeyHistory[plycount] = zobristKey;
    }
    std::string BoardToFEN() {
        std::stringstream fen;

        // Board position
        for (int rank = 7; rank >= 0; rank--) {
            int emptySquares = 0;
            for (int file = 0; file < 8; file++) {
                uint64_t square = 1ULL << (rank * 8 + file);
                char piece = ' ';
                if (whitePawns & square) piece = 'P';
                else if (blackPawns & square) piece = 'p';
                else if (whiteKnights & square) piece = 'N';
                else if (blackKnights & square) piece = 'n';
                else if (whiteBishops & square) piece = 'B';
                else if (blackBishops & square) piece = 'b';
                else if (whiteRooks & square) piece = 'R';
                else if (blackRooks & square) piece = 'r';
                else if (whiteQueens & square) piece = 'Q';
                else if (blackQueens & square) piece = 'q';
                else if (whiteKing & square) piece = 'K';
                else if (blackKing & square) piece = 'k';

                if (piece != ' ') {
                    if (emptySquares > 0) {
                        fen << emptySquares;
                        emptySquares = 0;
                    }
                    fen << piece;
                } else {
                    emptySquares++;
                }
            }
            if (emptySquares > 0) {
                fen << emptySquares;
            }
            if (rank > 0) {
                fen << '/';
            }
        }

        // Active color
        fen << (whiteToMove ? " w " : " b ");

        // Castling rights
        std::string castling = "";
        uint8_t castlingRights = castlingRightHistory[plycount];
        if (castlingRights & 8) castling += 'K';
        if (castlingRights & 4) castling += 'Q';
        if (castlingRights & 2) castling += 'k';
        if (castlingRights & 1) castling += 'q';
        fen << (castling.empty() ? "-" : castling) << " ";

        // En passant target square
        std::string enPassantSquare;
        uint8_t enPassantFile = enPassantFileHistory[plycount];
        switch (enPassantFile){
            case 0: (!whiteToMove ? enPassantSquare = "a3" : enPassantSquare = "a6"); break;
            case 1: (!whiteToMove ? enPassantSquare = "b3" : enPassantSquare = "b6"); break;
            case 2: (!whiteToMove ? enPassantSquare = "c3" : enPassantSquare = "c6"); break;
            case 3: (!whiteToMove ? enPassantSquare = "d3" : enPassantSquare = "d6"); break;
            case 4: (!whiteToMove ? enPassantSquare = "e3" : enPassantSquare = "e6"); break;
            case 5: (!whiteToMove ? enPassantSquare = "f3" : enPassantSquare = "f6"); break;
            case 6: (!whiteToMove ? enPassantSquare = "g3" : enPassantSquare = "g6"); break;
            case 7: (!whiteToMove ? enPassantSquare = "h3" : enPassantSquare = "h6"); break;
            default: enPassantSquare = ""; break;
        }
        fen << (enPassantSquare.empty() ? "-" : enPassantSquare) << " ";

        // Halfmove clock and fullmove number
        fen << halfmoveClockHistory[plycount] << " " << fullmoveNumber;
        return fen.str();
    }
    void emptyBoard(){
        // Reset all bitboards to 0, keep game state and game history information
        whitePawns = blackPawns = 0ULL;
        whiteKnights = blackKnights = 0ULL;
        whiteBishops = blackBishops = 0ULL;
        whiteRooks = blackRooks = 0ULL;
        whiteQueens = blackQueens = 0ULL;
        whiteKing = blackKing = 0ULL;
        whitePieces = whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
        blackPieces = blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
        allOccupied = whitePieces | blackPieces;
    }
    
    // Zobrist Hashing functions
    void initializeZobristHashes() {
        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 generator(rd()); // Seed the generator
        std::uniform_int_distribution<uint64_t> distribution(0, UINT64_MAX); // Uniform distribution for 64-bit integers

        // Initialize piece hashes
        for (int pieceType = 0; pieceType < 12; ++pieceType) {
            for (int square = 0; square < 64; ++square) {
                pieceHash[pieceType][square] = distribution(generator);
            }
        }

        whiteToMoveHash = distribution(generator);

        // Initialize castling rights
        for (int i = 0; i < 16; ++i) {
            castlingHash[i] = distribution(generator);
        }

        // Initialize en passant hashes
        for (int i = 0; i < 8; ++i) {
            enPassantHash[i] = distribution(generator);
        }
    }
    inline uint64_t getZobristKey() {
        uint64_t hash = 1;

        // Hash in pieces on the board
        uint64_t bitboard = whitePawns;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[0][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = blackPawns;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[1][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = whiteKnights;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[2][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = blackKnights;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[3][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = whiteBishops;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[4][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = blackBishops;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[5][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = whiteRooks;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[6][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = blackRooks;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[7][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = whiteQueens;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[8][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = blackQueens;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[9][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = whiteKing;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[10][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }
        bitboard = blackKing;
        while (bitboard) {
            uint8_t square = ctz64(bitboard); // Find the index of the least significant bit set
            hash ^= pieceHash[11][square]; // XOR with the Zobrist key for this piece on this square
            bitboard &= bitboard - 1; // Clear the least significant bit set
        }

        // Add castling rights
        uint8_t castlingRights = castlingRightHistory[plycount];
        hash ^= castlingHash[castlingRights];

        // Add en passant file
        uint8_t enPassantFile = enPassantFileHistory[plycount];
        if (~enPassantFile) { // if not all bits are set
            hash ^= enPassantHash[enPassantFile];
        }

        // Incorporate the turn
        (whiteToMove ? hash ^= whiteToMoveHash : hash);

        return hash;
    }

    // Move Generation
    inline uint64_t pawnMoveableSquare(uint8_t from) {
        uint64_t pawn = 1ULL << from;
        uint64_t empty = ~allOccupied;
        uint64_t enemies;
        uint64_t singlePush, doublePush, leftCapture, rightCapture;
        uint64_t startRank, notHFile, notAFile;
        
        if (whiteToMove) {
            enemies = blackPieces;
            singlePush = (pawn << 8) & empty;
            doublePush = ((pawn & RANK_2) << 16) & empty & (empty << 8);
            leftCapture = (pawn << 7) & ~FILE_H & enemies;
            rightCapture = (pawn << 9) & ~FILE_A & enemies;
        } else {
            enemies = whitePieces;
            singlePush = (pawn >> 8) & empty;
            doublePush = ((pawn & RANK_7) >> 16) & empty & (empty >> 8);
            leftCapture = (pawn >> 9) & ~FILE_H & enemies;
            rightCapture = (pawn >> 7) & ~FILE_A & enemies;
        }
        return singlePush | doublePush | leftCapture | rightCapture;
    }
    inline uint64_t rookMoveableSquare(uint8_t from){
        const MagicEntry& entry = ROOK_MAGICS[from];
        uint64_t index = ((allOccupied & entry.mask) * entry.magic) >> (64 - entry.shift);
        uint64_t attacks = entry.moves[index];
        uint64_t enemyPieces = whiteToMove ? blackPieces : whitePieces;
        return attacks & (enemyPieces | ~allOccupied);
    }
    inline uint64_t knightMoveableSquare(uint8_t from) {
        uint64_t knight_attacks = knight_lookup[from];
        uint64_t blockers;
        blockers = whiteToMove ? whitePieces : blackPieces;
        return knight_attacks & ~blockers;
    }
    inline uint64_t bishopMoveableSquare(uint8_t from){
        const MagicEntry& entry = BISHOP_MAGICS[from];
        uint64_t index = ((allOccupied & entry.mask) * entry.magic) >> (64 - entry.shift);
        uint64_t attacks = entry.moves[index];
        uint64_t enemyPieces = whiteToMove ? blackPieces : whitePieces;
        return attacks & (enemyPieces | ~allOccupied);
    }
    inline uint64_t queenMoveableSquare(uint8_t from){
        uint64_t rookAttacks = rookMoveableSquare(from);
        uint64_t bishopAttacks = bishopMoveableSquare(from);
        return rookAttacks | bishopAttacks;
    }
    inline uint64_t kingMoveableSquare(uint8_t from) {
        uint64_t king_attacks = king_lookup[from];
        uint64_t blockers = whiteToMove ? whitePieces : blackPieces;
        uint64_t moveable_square = king_attacks & ~blockers;
        return moveable_square;
    }
    // calculate squares from enemy to own king (including enemy itself)
    inline uint64_t generateCheckedSquares() {
        uint64_t checkedSquares = 0xFFFFFFFFFFFFFFFFULL; // Start with all squares set
        uint64_t kingBB = whiteToMove ? whiteKing : blackKing;
        int kingSquare = ctz64(kingBB);
        uint64_t opponentPieces = whiteToMove ? blackPieces : whitePieces;
        uint64_t ownPieces = whiteToMove ? whitePieces : blackPieces;
        uint64_t checkers = 0ULL;

        // Check for attacks by pawns
        uint64_t pawnAttacks = whiteToMove ? 
            ((kingBB << 7) & ~FILE_H) | ((kingBB << 9) & ~FILE_A) :
            ((kingBB >> 7) & ~FILE_A) | ((kingBB >> 9) & ~FILE_H);
        uint64_t opponentPawns = whiteToMove ? blackPawns : whitePawns;
        checkers |= pawnAttacks & opponentPawns;

        // Check for attacks by knights
        uint64_t knightAttacks = knightMoveableSquare(kingSquare);
        uint64_t opponentKnights = whiteToMove ? blackKnights : whiteKnights;
        checkers |= knightAttacks & opponentKnights;

        // Check for attacks by bishops and queens
        uint64_t bishopAttacks = bishopMoveableSquare(kingSquare);
        uint64_t opponentBishopsQueens = whiteToMove ? (blackBishops | blackQueens) : (whiteBishops | whiteQueens);
        uint64_t bishopCheckers = bishopAttacks & opponentBishopsQueens;
        checkers |= bishopCheckers;

        // Check for attacks by rooks and queens
        uint64_t rookAttacks = rookMoveableSquare(kingSquare);
        uint64_t opponentRooksQueens = whiteToMove ? (blackRooks | blackQueens) : (whiteRooks | whiteQueens);
        uint64_t rookCheckers = rookAttacks & opponentRooksQueens;
        checkers |= rookCheckers;

        // Count the number of checkers
        int numCheckers = popcount64(checkers);

        if (numCheckers == 0) {
            return checkedSquares; // No check, all squares are valid
        } else if (numCheckers > 1) {
            return 0ULL; // Double check, no moves are valid (except for king moves)
        } else {
            // Single check
            uint64_t checkerSquare = 1ULL << ctz64(checkers);
            if (bishopCheckers) {
                return (bishopMoveableSquare(kingSquare) & bishopMoveableSquare(ctz64(checkerSquare))) | checkerSquare;
            } else if (rookCheckers) {
                return (rookMoveableSquare(kingSquare) & rookMoveableSquare(ctz64(checkerSquare))) | checkerSquare;
            } else {
                return checkerSquare; // For pawn or knight checks, only capturing the checker is valid
            }
        }
    }
    // calculate seen squares by enemy pieces
    inline uint64_t generateSeenSquares() {
        uint64_t seenSquares = 0ULL;

        uint64_t kingmask;
        uint64_t opponentPieces, opponentPawns, opponentKnights, opponentBishops, opponentRooks, opponentQueens, opponentKing;
        if (whiteToMove) {
            opponentPieces = blackPieces;
            opponentPawns = blackPawns;
            opponentKnights = blackKnights;
            opponentBishops = blackBishops;
            opponentRooks = blackRooks;
            opponentQueens = blackQueens;
            opponentKing = blackKing;

            kingmask = 1ULL << ctz64(whiteKing);
            whiteKing ^= kingmask;
            whitePieces ^= kingmask;
            allOccupied ^= kingmask;
        } else {
            opponentPieces = whitePieces;
            opponentPawns = whitePawns;
            opponentKnights = whiteKnights;
            opponentBishops = whiteBishops;
            opponentRooks = whiteRooks;
            opponentQueens = whiteQueens;
            opponentKing = whiteKing;

            kingmask = 1ULL << ctz64(blackKing);
            blackKing ^= kingmask;
            blackPieces ^= kingmask;
            allOccupied ^= kingmask;
        }

        while (opponentPieces) {
            uint8_t from = ctz64(opponentPieces);
            uint64_t piece = 1ULL << from;
            seenSquares |= 
                ((piece & opponentPawns) ? (whiteToMove ? 
                    ((piece >> 7) & ~FILE_A) | ((piece >> 9) & ~FILE_H) :
                    ((piece << 7) & ~FILE_H) | ((piece << 9) & ~FILE_A)) : 0ULL) |
                ((piece & opponentKnights) ? knightMoveableSquare(from) : 0ULL) |
                ((piece & opponentBishops) ? bishopMoveableSquare(from) : 0ULL) |
                ((piece & opponentRooks) ? rookMoveableSquare(from) : 0ULL) |
                ((piece & opponentQueens) ? queenMoveableSquare(from) : 0ULL) |
                ((piece & opponentKing) ? kingMoveableSquare(from) : 0ULL);
            
            opponentPieces &= opponentPieces - 1;  // Clear the least significant bit
        }

        if (whiteToMove){
            whiteKing ^= kingmask;
            whitePieces ^= kingmask;
            allOccupied ^= kingmask;
        } else{
            blackKing ^= kingmask;
            blackPieces ^= kingmask;
            allOccupied ^= kingmask;
        }
        return seenSquares;
    }
    // generate pin masks
    inline uint64_t generatePinD12(){
        uint64_t pinD12 = 0;
        if (whiteToMove){
            uint8_t kingSquare = ctz64(whiteKing);
            const MagicEntry& entry = BISHOP_MAGICS[kingSquare];
            uint64_t index = ((blackPieces & entry.mask) * entry.magic) >> (64 - entry.shift);
            uint64_t pinners = entry.moves[index] & (blackQueens | blackBishops);
            while (pinners) {
                uint8_t pinnerSquare = ctz64(pinners);
                uint64_t betweenSquares = betweenD12(pinnerSquare, kingSquare);
                if (popcount64(betweenSquares & whitePieces) == 2){
                    pinD12 |= betweenSquares;
                }
                pinners &= pinners - 1;
            }
        } else{
            uint8_t kingSquare = ctz64(blackKing);
            const MagicEntry& entry = BISHOP_MAGICS[kingSquare];
            uint64_t index = ((whitePieces & entry.mask) * entry.magic) >> (64 - entry.shift);
            uint64_t pinners = entry.moves[index] & (whiteQueens | whiteBishops);
            while (pinners) {
                uint8_t pinnerSquare = ctz64(pinners);
                uint64_t betweenSquares = betweenD12(pinnerSquare, kingSquare);
                if (popcount64(betweenSquares & blackPieces) == 2){
                    pinD12 |= betweenSquares;
                }
                pinners &= pinners - 1;
            }
        }
        return pinD12;
    };
    inline uint64_t betweenD12(uint8_t piecesquare, uint8_t kingsquare){
        uint64_t mask = 0;
    
        // Calculate row and column difference
        int row1 = piecesquare / 8;
        int col1 = piecesquare % 8;
        int row2 = kingsquare / 8;
        int col2 = kingsquare % 8;
        
        int step;
        
        if (col2 > col1){
            if (row2 > row1){
                step = 9; // Moving up-right
            }else{
                step = -7; // Moving down-right
            }
        } else{
            if (row2 > row1){
                step = 7; // Moving up-left
            }else{
                step = -9; // Moving down-left
            }
        }

        // Calculate the range of squares between square1 and square2
        int steps = (kingsquare - piecesquare) / step; // add one extra step to include the square behind the king
        
        int sq = piecesquare; // start stepping at piecesquare
        while ((steps >= 0) && (sq >= 0) && (sq <=63)) {
            mask |= 1ULL << sq;
            sq += step;
            steps--;
        }
        return mask;
    };
    inline uint64_t generatePinHV(){
        uint64_t pinHV = 0;
        if (whiteToMove){
            uint8_t kingSquare = ctz64(whiteKing);
            const MagicEntry& entry = ROOK_MAGICS[kingSquare];
            uint64_t index = ((blackPieces & entry.mask) * entry.magic) >> (64 - entry.shift);
            uint64_t pinners = entry.moves[index] & (blackQueens | blackRooks);
            while (pinners) {
                uint8_t pinnerSquare = ctz64(pinners);
                uint64_t betweenSquares = betweenHV(kingSquare, pinnerSquare);
                if (popcount64(betweenSquares & whitePieces) == 2){
                    pinHV |= betweenSquares;
                }
                pinners &= pinners - 1;
            }
        } else{
            uint8_t kingSquare = ctz64(blackKing);
            const MagicEntry& entry = ROOK_MAGICS[kingSquare];
            uint64_t index = ((whitePieces & entry.mask) * entry.magic) >> (64 - entry.shift);
            uint64_t pinners = entry.moves[index] & (whiteQueens | whiteRooks);
            while (pinners) {
                uint8_t pinnerSquare = ctz64(pinners);
                uint64_t betweenSquares = betweenHV(kingSquare, pinnerSquare);
                if (popcount64(betweenSquares & blackPieces) == 2){
                    pinHV |= betweenSquares;
                }
                pinners &= pinners - 1;
            }
        }
        return pinHV;
    };
    inline uint64_t betweenHV(uint8_t square1, uint8_t square2){
        uint64_t mask = 0;
        if (square1 % 8 == square2 % 8) { // If on the same file (vertically aligned)
            int minSquare = std::min(square1, square2);
            int maxSquare = std::max(square1, square2);
            for (int i = minSquare; i <= maxSquare; i += 8) {
                mask |= (1ULL << i);
            }
        }
        else if (square1 / 8 == square2 / 8) { // If on the same rank (horizontally aligned)
            int minSquare = std::min(square1, square2);
            int maxSquare = std::max(square1, square2);
            for (int i = minSquare; i <= maxSquare; ++i) {
                mask |= (1ULL << i);
            }
        }
        return mask;
    };
    // generate all legal moves in a position
    inline std::vector<uint16_t> generateAllLegalMoves() {
        std::vector<uint16_t> allLegalMoves;
        allLegalMoves.reserve(218);
        uint64_t seenSquares = generateSeenSquares();
        uint64_t checkedSquares = generateCheckedSquares();
        uint64_t currentSidePieces = whiteToMove ? whitePieces : blackPieces;
        
        uint64_t pinHV = generatePinHV();
        uint64_t pinD12 = generatePinD12();
        uint64_t allPins = pinHV | pinD12;

        uint64_t rook_nopin;
        uint64_t rook_pin;
        uint64_t bishop_nopin;
        uint64_t bishop_pin;
        uint64_t queen_nopin;
        uint64_t queen_pinHV;
        uint64_t queen_pinD12;
        uint64_t pawns_nopin;
        uint64_t pawns_pinHV;
        uint64_t pawns_pinD12;
        uint64_t knight;
        uint64_t king;

        if (whiteToMove){
            rook_nopin = whiteRooks & ~allPins;
            rook_pin = whiteRooks & pinHV;
            bishop_nopin = whiteBishops & ~allPins;
            bishop_pin = whiteBishops & pinD12;

            queen_nopin = whiteQueens & ~allPins; 
            queen_pinHV = whiteQueens & pinHV;
            queen_pinD12 = whiteQueens & pinD12;

            pawns_nopin = whitePawns & ~allPins;
            pawns_pinHV = whitePawns & pinHV;
            pawns_pinD12 = whitePawns & pinD12;

            knight = whiteKnights & ~allPins; // a pinned knight can never move
            king = whiteKing;
        } else{
            rook_nopin = blackRooks & ~allPins;
            rook_pin = blackRooks & pinHV;
            bishop_nopin = blackBishops & ~allPins;
            bishop_pin = blackBishops & pinD12;

            queen_nopin = blackQueens & ~allPins; 
            queen_pinHV = blackQueens & pinHV;
            queen_pinD12 = blackQueens & pinD12;

            pawns_nopin = blackPawns & ~allPins;
            pawns_pinHV = blackPawns & pinHV;
            pawns_pinD12 = blackPawns & pinD12;

            knight = blackKnights & ~allPins; // a pinned knight can never move
            king = blackKing;
        }

        // check for enPassantMove. Add if possible
        uint8_t enPassantFile = enPassantFileHistory[plycount];
        if (enPassantFile != 0xFF){
            if (whiteToMove){
                uint8_t enPassantSquare = enPassantFile + 40;
                // Check if capture from left is possible, cant be file 0
                if ((whitePawns & (1ULL << (enPassantFile + 31))) && enPassantFile != 0){
                    uint16_t move = ((enPassantFile + 31) & 0x3F) | ((enPassantSquare & 0x3F) << 6);
                    makeMove(move);
                    whiteToMove = !whiteToMove; // flip because we want to see check on own king
                    if (!isCheck()){
                        allLegalMoves.emplace_back(move);
                    }
                    whiteToMove = !whiteToMove;
                    unmakeMove();
                }
                 // Check if capture from right is possible, cant be file 7
                if ((whitePawns & (1ULL << (enPassantFile + 33))) && enPassantFile != 7){
                    uint16_t move = ((enPassantFile + 33) & 0x3F) | ((enPassantSquare & 0x3F) << 6);
                    makeMove(move);
                    whiteToMove = !whiteToMove; // flip because we want to see check on own king
                    if (!isCheck()){
                        allLegalMoves.emplace_back(move);
                    }
                    whiteToMove = !whiteToMove;
                    unmakeMove();
                }
            } else{
                uint8_t enPassantSquare = enPassantFile + 16;
                // Check if capture from left is possible, cant be file 0
                if ((blackPawns & (1ULL << (enPassantFile + 23))) && enPassantFile != 0){
                    uint16_t move = ((enPassantFile + 23) & 0x3F) | ((enPassantSquare & 0x3F) << 6);
                    makeMove(move);
                    whiteToMove = !whiteToMove; // flip because we want to see check on own king
                    if (!isCheck()){
                        allLegalMoves.emplace_back(move);
                    }
                    whiteToMove = !whiteToMove;
                    unmakeMove();
                }
                // Check if capture from right is possible, cant be file 7
                if ((blackPawns & (1ULL << (enPassantFile + 25))) && enPassantFile != 7){
                    uint16_t move = ((enPassantFile + 25) & 0x3F) | ((enPassantSquare & 0x3F) << 6);
                    makeMove(move);
                    whiteToMove = !whiteToMove; // flip because we want to see check on own king
                    if (!isCheck()){
                        allLegalMoves.emplace_back(move);
                    }
                    whiteToMove = !whiteToMove;
                    unmakeMove();
                }
            }
        }

        // Add castling if possible
        if (!isCheck()){
            uint8_t castlingRights = castlingRightHistory[plycount];
            if (whiteToMove){
                // castling right has to be set, rook has to be at square 7, squares 5 and 6 cant be seen or occupied 
                if (((castlingRights & 8) >> 3) && (whiteRooks & (1ULL << 7)) && !(seenSquares & WKS_SEEN) && !(allOccupied & WKS_OCC)){
                    uint16_t move = (4 & 0x3F) | ((6 & 0x3F) << 6) | (1 << 12);
                    allLegalMoves.emplace_back(move);
                }
                // castling right has to be set, rook has to be at square 0, squares 3 and 2 cant be seen and square 1 cant be occupied
                if (((castlingRights & 4) >> 2) && (whiteRooks & (1ULL << 0)) && !(seenSquares & WQS_SEEN) && !(allOccupied & WQS_OCC)){
                    uint16_t move = (4 & 0x3F) | ((2 & 0x3F) << 6) | (1 << 12);
                    allLegalMoves.emplace_back(move);
                }
            }
            else{
                // castling right has to be set, rook has to be at square 63, squares 61 and 62 cant be seen or occupied
                if (((castlingRights & 2) >> 1) && (blackRooks & (1ULL << 63)) && !(seenSquares & BKS_SEEN) && !(allOccupied & BKS_OCC)){
                    uint16_t move = (60 & 0x3F) | ((62 & 0x3F) << 6) | (1 << 12);
                    allLegalMoves.emplace_back(move);
                }
                // castling right has to be set, rook has to be at square 56, squares 59 and 58 cant be seen and square 57 cant be occupied
                if ((castlingRights & 1) && (blackRooks & (1ULL << 56)) && !(seenSquares & BQS_SEEN) && !(allOccupied & BQS_OCC)){
                    uint16_t move = (60 & 0x3F) | ((58 & 0x3F) << 6) | (1 << 12);
                    allLegalMoves.emplace_back(move);
                }
            }
        }

        // Pawns not pinned
        while (pawns_nopin){
            uint8_t from = ctz64(pawns_nopin);
            uint64_t moveableSquares = pawnMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                if (whiteToMove ? (to > 55) : (to < 8)){
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((0 & 0x3) << 13) | (1 << 15)); // knight
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((1 & 0x3) << 13) | (1 << 15)); // bishop
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((2 & 0x3) << 13) | (1 << 15)); // rook
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((3 & 0x3) << 13) | (1 << 15)); // queen
                } else{
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                }
                legal_squares &= legal_squares - 1;
            }
            pawns_nopin &= pawns_nopin - 1;
        }
        // Pawns HV pinned
        while (pawns_pinHV){
            uint8_t from = ctz64(pawns_pinHV);
            uint64_t moveableSquares = pawnMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares & pinHV;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                if (whiteToMove ? (to > 55) : (to < 8)){
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((0 & 0x3) << 13) | (1 << 15)); // knight
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((1 & 0x3) << 13) | (1 << 15)); // bishop
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((2 & 0x3) << 13) | (1 << 15)); // rook
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((3 & 0x3) << 13) | (1 << 15)); // queen
                } else{
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                }
                legal_squares &= legal_squares - 1;
            }
            pawns_pinHV &= pawns_pinHV - 1;
        }
        // Pawns D12 pinned
        while (pawns_pinD12){
            uint8_t from = ctz64(pawns_pinD12);
            uint64_t moveableSquares = pawnMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares & pinD12;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                if (whiteToMove ? (to > 55) : (to < 8)){
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((0 & 0x3) << 13) | (1 << 15)); // knight
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((1 & 0x3) << 13) | (1 << 15)); // bishop
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((2 & 0x3) << 13) | (1 << 15)); // rook
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6) | ((3 & 0x3) << 13) | (1 << 15)); // queen
                } else{
                    allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                }
                legal_squares &= legal_squares - 1;
            }
            pawns_pinD12 &= pawns_pinD12 - 1;
        }

        // Rooks not pinned
        while (rook_nopin){
            uint8_t from = ctz64(rook_nopin);
            uint64_t moveableSquares = rookMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            rook_nopin &= rook_nopin - 1;
        }
        // Rooks pinned
        while (rook_pin){
            uint8_t from = ctz64(rook_pin);
            uint64_t moveableSquares = rookMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares & pinHV;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            rook_pin &= rook_pin - 1;
        }
        // Bishop not pinned
        while (bishop_nopin){
            uint8_t from = ctz64(bishop_nopin);
            uint64_t moveableSquares = bishopMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            bishop_nopin &= bishop_nopin - 1;
        }
        // Bishop pinned
        while (bishop_pin){
            uint8_t from = ctz64(bishop_pin);
            uint64_t moveableSquares = bishopMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares & pinD12;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            bishop_pin &= bishop_pin - 1;
        }

        while (queen_nopin){
            uint8_t from = ctz64(queen_nopin);
            uint64_t moveableSquares = queenMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            queen_nopin &= queen_nopin - 1;
        }

        while (queen_pinHV){
            uint8_t from = ctz64(queen_pinHV);
            uint64_t moveableSquares = rookMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares & pinHV;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            queen_pinHV &= queen_pinHV - 1;
        }

        while (queen_pinD12){
            uint8_t from = ctz64(queen_pinD12);
            uint64_t moveableSquares = bishopMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares & pinD12;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            queen_pinD12 &= queen_pinD12 - 1;
        }

        while (knight){
            uint8_t from = ctz64(knight);
            uint64_t moveableSquares = knightMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & checkedSquares;
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            knight &= knight - 1;
        }

        while (king){
            uint8_t from = ctz64(king);
            uint64_t moveableSquares = kingMoveableSquare(from);
            uint64_t legal_squares = moveableSquares & ~seenSquares; // king cant move into seen squares
            while (legal_squares){
                uint8_t to = ctz64(legal_squares);
                allLegalMoves.emplace_back((from & 0x3F) | ((to & 0x3F) << 6));
                legal_squares &= legal_squares - 1;
            }
            king &= king - 1;
        }

        return allLegalMoves;
    };
    
    // Move execution
    inline void makeMove(uint16_t move){
        uint8_t from = move & 0x3F;
        uint8_t to = (move >> 6) & 0x3F;
        uint64_t fromMask = 1Ull << from;
        uint64_t toMask = 1ULL << to;
        uint64_t fromToMask = fromMask ^ toMask;
        uint8_t movedPiece = getPieceOfSquare(from);
        uint8_t capturedPiece = getPieceOfSquare(to);
        uint8_t enPassantFile = enPassantFileHistory[plycount];
        uint8_t castlingRights = castlingRightHistory[plycount];

        // Update bitboards //
        if (whiteToMove){
            // Move the piece
            whitePieces ^= fromToMask;
            switch (movedPiece) {
                case 1: whitePawns ^= fromToMask; break;
                case 2: whiteKnights ^= fromToMask; break;
                case 3: whiteBishops ^= fromToMask; break;
                case 4: whiteRooks ^= fromToMask; break;
                case 5: whiteQueens ^= fromToMask; break;
                case 6: whiteKing ^= fromToMask; break;
            }
            
            // Handle en passant Capture
            if ((movedPiece == 1) && (enPassantFile != 0xFF) && (to == (enPassantFile + 40))) {
                // if a pawn goes to the en passant square -> then en passant capture 
                // en passant is happening
                // clear black pawn one rank down
                blackPawns &= ~(1ULL << (enPassantFile + 32));
                blackPieces &= ~(1ULL << (enPassantFile + 32));
            } 
            // Handle normal captures
            else if (capturedPiece != 0){
                blackPieces &= ~toMask;
                switch (capturedPiece) {
                    case 1: blackPawns &= ~toMask; break;
                    case 2: blackKnights &= ~toMask; break;
                    case 3: blackBishops &= ~toMask; break;
                    case 4: blackRooks &= ~toMask; break;
                    case 5: blackQueens &= ~toMask; break;
                    case 6: blackKing &= ~toMask; break;
                }
            }
        } else { // Black's move
            // Move the piece
            blackPieces ^= fromToMask;
            switch (movedPiece) {
                case 1: blackPawns ^= fromToMask; break;
                case 2: blackKnights ^= fromToMask; break;
                case 3: blackBishops ^= fromToMask; break;
                case 4: blackRooks ^= fromToMask; break;
                case 5: blackQueens ^= fromToMask; break;
                case 6: blackKing ^= fromToMask; break;
            }
            // Handle en passant Capture
            if ((movedPiece == 1) && (enPassantFile != 0xFF) && (to == enPassantFile + 16)) {
                // if a pawn goes to the en passant square -> then en passant capture 
                // en passant is happening
                // clear black pawn one rank up
                whitePawns &= ~(1ULL << (enPassantFile + 24));
                whitePieces &= ~(1ULL << (enPassantFile + 24));
            } 
            // Handle normal captures
            else if (capturedPiece != 0){
                whitePieces &= ~toMask;
                switch (capturedPiece) {
                    case 1: whitePawns &= ~toMask; break;
                    case 2: whiteKnights &= ~toMask; break;
                    case 3: whiteBishops &= ~toMask; break;
                    case 4: whiteRooks &= ~toMask; break;
                    case 5: whiteQueens &= ~toMask; break;
                    case 6: whiteKing &= ~toMask; break;
                }
            }
        }
         // Handle promotion
        if ((move >> 15) & 0x1){
            // no need to adjust whitePieces or blackPieces
            if (whiteToMove){
                whitePawns &= ~toMask; // clear Pawn from promotion square
                switch ((move >> 13) & 0x3){
                    case 3: whiteQueens  |= toMask; break;
                    case 2: whiteRooks   |= toMask; break;
                    case 1: whiteBishops |= toMask; break;
                    case 0: whiteKnights |= toMask; break;
                }
            } else{
                blackPawns &= ~toMask; // clear pawn from promotion square
                switch ((move >> 13) & 0x3) {
                    case 3: blackQueens |= toMask; break;
                    case 2: blackRooks |= toMask; break;
                    case 1: blackBishops |= toMask; break;
                    case 0: blackKnights |= toMask; break;
                }
            }
        }
        // Handle castling
        if (movedPiece == 6){
            if (std::abs(static_cast<int>(from) - static_cast<int>(to)) == 2){
                switch (to){
                    case 6: // white king side
                        whiteRooks &= ~(1ULL << 7); // set 7th bit to zero
                        whiteRooks |= (1ULL << 5); // set 5th bit to one
                        // also change whitePieces
                        whitePieces &= ~(1ULL << 7);
                        whitePieces |= (1ULL << 5);
                        // set first two bits of castlingRights to zero
                        castlingRights &= ~(0x8);
                        castlingRights &= ~(0x4);
                        break;
                    case 2: // white queen side
                        whiteRooks &= ~(1ULL << 0); // set 0th bit to zero
                        whiteRooks |= (1ULL << 3); // set 3th bit to one
                        // also change whitePieces
                        whitePieces &= ~(1ULL << 0);
                        whitePieces |= (1ULL << 3);
                        // set first two bits of castlingRights to zero
                        castlingRights &= ~(0x8);
                        castlingRights &= ~(0x4);
                        break;
                    case 62: // black king side
                        blackRooks &= ~(1ULL << 63); // set 63th bit to zero
                        blackRooks |= (1ULL << 61); // set 59th bit to one
                        // also change blackPieces
                        blackPieces &= ~(1ULL << 63);
                        blackPieces |= (1ULL << 61);
                        // set last two bits of castlingRights to zero
                        castlingRights &= ~(0x2);
                        castlingRights &= ~(0x1);
                        break;
                    case 58: // black queen side
                        blackRooks &= ~(1ULL << 56); // set 56th bit to zero
                        blackRooks |= (1ULL << 59); // set 59th bit to one
                        // also change blackPieces
                        blackPieces &= ~(1ULL << 56);
                        blackPieces |= (1ULL << 59);
                        // set last two bits of castlingRights to zero
                        castlingRights &= ~(0x2);
                        castlingRights &= ~(0x1);
                        break;
                }
            } else{ // only king move -> only update castling rights
                if (whiteToMove){
                    castlingRights &= ~(0x8);
                    castlingRights &= ~(0x4);
                } else{
                    castlingRights &= ~(0x2);
                    castlingRights &= ~(0x1);
                }
            }
        }
        if (movedPiece == 4){ // if rook has moved set castling rights
            switch (from){ // see where the rook has moved from
                case 7: // white king side
                    castlingRights &= ~(0x8);
                    break;
                case 0: // white queen side
                    // set second bit to zero
                    castlingRights &= ~(0x4);
                    break;
                case 63: // black king side
                    // set third bit to zero
                    castlingRights &= ~(0x2);
                    break;
                case 56: // black queen side
                    // set fourth bit to zero
                    castlingRights &= ~(0x1);
                    break;
            }
        }
        // modifiy enPassantFile
        enPassantFile = 0xFF;
        if (movedPiece == 1){
            if (whiteToMove ? (to-from == 16) : (from-to == 16)){
                enPassantFile = to%8;
            }
        }
        
        // Set all Pieces
        allOccupied = whitePieces | blackPieces;

        // Update game state information //
        plycount++;
        whiteToMove = !whiteToMove;
        moveHistory[plycount] = move;
        capturedPieceHistory[plycount] = capturedPiece;  // save captured Piece
        castlingRightHistory[plycount] = castlingRights; // save modified castling right
        enPassantFileHistory[plycount] = enPassantFile;  // save modified enPassant file
        uint8_t halfMoveValue = (capturedPiece || movedPiece == 1) ? 0 : halfmoveClockHistory[plycount - 1] + 1;
        halfmoveClockHistory[plycount] = halfMoveValue; // save current halfmoveValue to halfmoveClockHistory
        zobristKey = getZobristKey();
        zobristKeyHistory[plycount] = zobristKey; // save zobristKey to history
        if (whiteToMove) fullmoveNumber++;
    };
    inline void unmakeMove(){
        if (plycount == 0) return;  // No moves to undo

        uint16_t move = moveHistory[plycount]; // decrease plycount

        uint8_t from = move & 0x3F;
        uint8_t to = (move >> 6) & 0x3F;
        uint64_t fromMask = 1Ull << from;
        uint64_t toMask = 1ULL << to;
        uint64_t fromToMask = fromMask ^ toMask;
        uint8_t movedPiece = getPieceOfSquare(to);
        uint8_t capturedPiece = capturedPieceHistory[plycount];
        uint8_t enPassantFile = enPassantFileHistory[plycount-1]; // get enPassantfile from last move
        uint8_t castlingRights = castlingRightHistory[plycount];
       
        // Update bitboards //
        // We're undoing the last move, so the current turn is opposite of the move we're undoing
        if (whiteToMove){
            // Move the piece back
            blackPieces ^= fromToMask;
            switch (movedPiece) {
                case 1: blackPawns ^= fromToMask; break;
                case 2: blackKnights ^= fromToMask; break;
                case 3: blackBishops ^= fromToMask; break;
                case 4: blackRooks ^= fromToMask; break;
                case 5: blackQueens ^= fromToMask; break;
                case 6: blackKing ^= fromToMask; break;
            }
            
            // Handle en passant Capture
            // We have moved a pawn and en passant file was set
            if ((movedPiece == 1) && (enPassantFile != 0xFF) && (to == enPassantFile + 16)) {
                // if pawn has moved to the en passant square -> then en passant capture happened
                // en passant has happened
                // restore white pawn one file up
                whitePawns |= (1ULL << (enPassantFile + 24));
                whitePieces |= (1ULL << (enPassantFile + 24));
            }
            // Handle normal captures
            else if (capturedPiece != 0){
                whitePieces |= toMask;
                switch (capturedPiece) {
                    case 1: whitePawns |= toMask; break;
                    case 2: whiteKnights |= toMask; break;
                    case 3: whiteBishops |= toMask; break;
                    case 4: whiteRooks |= toMask; break;
                    case 5: whiteQueens |= toMask; break;
                    case 6: whiteKing |= toMask; break;
                }
            }
        } else{ // remove black move
            // Move the piece back
            whitePieces ^= fromToMask;
            switch (movedPiece) {
                case 1: whitePawns ^= fromToMask; break;
                case 2: whiteKnights ^= fromToMask; break;
                case 3: whiteBishops ^= fromToMask; break;
                case 4: whiteRooks ^= fromToMask; break;
                case 5: whiteQueens ^= fromToMask; break;
                case 6: whiteKing ^= fromToMask; break;
            }
            
            // Handle en passant Capture
            // We have moved a pawn and en passant file was set
            if ((movedPiece == 1) && (enPassantFile != 0xFF) && (to == enPassantFile + 40)) {
                // if pawn has moved to the en passant square -> then en passant capture happened
                // en passant has happened
                // restore white pawn one file up
                blackPawns |= (1ULL << (enPassantFile + 32));
                blackPieces |= (1ULL << (enPassantFile + 32));
            }
            // Handle normal captures
            else if (capturedPiece != 0){
                // printBitboard(blackPawns);
                blackPieces |= toMask;
                switch (capturedPiece) {
                    case 1: blackPawns |= toMask; break;
                    case 2: blackKnights |= toMask; break;
                    case 3: blackBishops |= toMask; break;
                    case 4: blackRooks |= toMask; break;
                    case 5: blackQueens |= toMask; break;
                    case 6: blackKing |= toMask; break;
                }
            }
        }
        // Handle promotion
        if ((move >> 15) & 0x1){
            // no need to adjust whitePieces or blackPieces
            if (!whiteToMove){ // undoing white move
                whitePawns |= fromMask; 
                switch ((move >> 13) & 0x3) {
                    case 0: whiteKnights &= ~fromMask; break;
                    case 1: whiteBishops &= ~fromMask; break;
                    case 2: whiteRooks &= ~fromMask; break;
                    case 3: whiteQueens &= ~fromMask; break;
                }
            } else{
                blackPawns |= fromMask;
                switch ((move >> 13) & 0x3) {
                    case 0: blackKnights &= ~fromMask; break;
                    case 1: blackBishops &= ~fromMask; break;
                    case 2: blackRooks &= ~fromMask; break;
                    case 3: blackQueens &= ~fromMask; break;
                }
            }
        }
        // Handle castling
        if (movedPiece == 6){
            // if castling happened
            if (std::abs(static_cast<int>(from) - static_cast<int>(to)) == 2){
                switch (to) { // see where the king has moved to
                    case 6: // white king side
                        whiteRooks &= ~(1ULL << 5); // set 5th bit to zero
                        whiteRooks |= (1ULL << 7); // set 7th bit to one
                        // also change whitePieces
                        whitePieces &= ~(1ULL << 5);
                        whitePieces |= (1ULL << 7);
                        break;
                    case 2: // white queen side
                        whiteRooks &= ~(1ULL << 3); // set 3rd bit to zero
                        whiteRooks |= (1ULL << 0); // set 0th bit to one
                        // also change whitePieces
                        whitePieces &= ~(1ULL << 3);
                        whitePieces |= (1ULL << 0);
                        break;
                    case 62: // black king side
                        blackRooks &= ~(1ULL << 61); // set 61th bit to zero
                        blackRooks |= (1ULL << 63); // set 63th bit to one
                        // also change blackPieces
                        blackPieces &= ~(1ULL << 61);
                        blackPieces |= (1ULL << 63);
                        break;
                    case 58: // black queen side
                        blackRooks &= ~(1ULL << 59); // set 59th bit to zero
                        blackRooks |= (1ULL << 56); // set 56th bit to one
                        // also change blackPieces
                        blackPieces &= ~(1ULL << 59);
                        blackPieces |= (1ULL << 56);
                        break;
                }
            }
        }

        // Set all Pieces
        allOccupied = whitePieces | blackPieces;

        // Update game state information //
        whiteToMove = !whiteToMove;
        capturedPieceHistory[plycount] = 0; // clear last captured Piece
        castlingRightHistory[plycount] = 0; // clear last castling right
        enPassantFileHistory[plycount] = 0; // clear last enPassant file
        halfmoveClockHistory[plycount] = 0; // clear last halfmoveValue
        zobristKeyHistory[plycount] = 0;    // clear last zobristKey
        if (!whiteToMove) fullmoveNumber--;
        zobristKey = getZobristKey();
        plycount--;
    };

    // Game end functions
    inline bool isCheck() {
        uint8_t kingsquare = ctz64(whiteToMove ? whiteKing : blackKing);
        
        // Pawn attacks
        // check left pawn attack
        bool left_pawn_attack = whiteToMove ? 
            // only check by panws if kingsquare < 48
            (kingsquare < 48 && kingsquare % 8 != 0 && (blackPawns & (1ULL << (kingsquare + 7)))) :
            // no check by pawns if kingsquare > 15
            (kingsquare > 15 && kingsquare % 8 != 0 && (whitePawns & (1ULL << (kingsquare - 9))));
        // check right pawn attack
        bool right_pawn_attack = whiteToMove ? 
            (kingsquare < 48 && kingsquare % 8 != 7 && (blackPawns & (1ULL << (kingsquare + 9)))) :
            (kingsquare > 15 && kingsquare % 8 != 7 && (whitePawns & (1ULL << (kingsquare - 7))));

        if (left_pawn_attack || right_pawn_attack) {
            return true;
        }

        // Knight attacks
        uint64_t opponent_knights = whiteToMove ? blackKnights : whiteKnights;
        if (knight_lookup[kingsquare] & opponent_knights) {
            return true;
        }

        // Slider attacks
        uint64_t opponent_bishops = whiteToMove ? blackBishops : whiteBishops;
        uint64_t opponent_queens = whiteToMove ? blackQueens : whiteQueens;
        uint64_t opponent_rooks = whiteToMove ? blackRooks : whiteRooks;
        uint64_t enemyPieces = whiteToMove ? blackPieces : whitePieces;

        // Bishop & Queen attacks
        const MagicEntry& bishop_entry = BISHOP_MAGICS[kingsquare];
        uint64_t bishop_index = ((allOccupied & bishop_entry.mask) * bishop_entry.magic) >> (64 - bishop_entry.shift);
        uint64_t bishop_attacks = bishop_entry.moves[bishop_index] & (enemyPieces | ~allOccupied);
        if (bishop_attacks & (opponent_bishops | opponent_queens)) {
            return true;
        }

        // Rook & Queen attacks
        const MagicEntry& rook_entry = ROOK_MAGICS[kingsquare];
        uint64_t rook_index = ((allOccupied & rook_entry.mask) * rook_entry.magic) >> (64 - rook_entry.shift);
        uint64_t rook_attacks = rook_entry.moves[rook_index] & (enemyPieces | ~allOccupied);
        if (rook_attacks & (opponent_rooks | opponent_queens)) {
            return true;
        }

        // King attacks
        uint64_t opponent_king = whiteToMove ? blackKing : whiteKing;
        if (king_lookup[kingsquare] & opponent_king) {
            return true;
        }

        return false;
    }
    inline bool isCheckmate(){
        // First, check if the current player is in check
        if (!isCheck()) {
            return false;
        }
        // If in check, generate all legal moves
        std::vector<uint16_t> legalMoves = generateAllLegalMoves();
        // If there are no legal moves, it's checkmate
        return legalMoves.empty();
    };
    inline bool isDraw() {
        // 1. Insufficient material
        if (isInsufficientMaterial()) {
            return true;
        }
        // 2. Stalemate
        if (isStalemate()) {
            return true;
        } 
        // 3. Fifty-move rule
        if (isFiftyMoveRule()) {
            return true;
        }
        // 4. Threefold repetition
        if (isThreefoldRepetition()) {
            return true;
        }
        return false;
    }
    bool isInsufficientMaterial() {
        // King vs. King
        if (whitePieces == whiteKing && blackPieces == blackKing) {
            return true;
        }
        // King and Bishop vs. King
        if ((whitePieces == (whiteKing | whiteBishops) && popcount64(whiteBishops) == 1 && blackPieces == blackKing) ||
            (blackPieces == (blackKing | blackBishops) && popcount64(blackBishops) == 1 && whitePieces == whiteKing)) {
            return true;
        }
        // King and Knight vs. King
        if ((whitePieces == (whiteKing | whiteKnights) && popcount64(whiteKnights) == 1 && blackPieces == blackKing) ||
            (blackPieces == (blackKing | blackKnights) && popcount64(blackKnights) == 1 && whitePieces == whiteKing)) {
            return true;
        }
        return false;
    }
    bool isStalemate() {
        if (isCheck()) {
            return false;
        }
        return generateAllLegalMoves().empty();
    }
    bool isThreefoldRepetition() {
        int startind = plycount - halfmoveClockHistory[plycount];
        int endind = plycount;
        std::unordered_map<uint64_t, int> keyCount;

        //  Loop through the Zobrist key history
        for (int i = startind; i <= endind; ++i) {
            uint64_t key = zobristKeyHistory[i];
            // Increment the count for this key
            keyCount[key]++;
            // If any key appears 3 times, we have a threefold repetition
            if (keyCount[key] >= 3) {
                return true;
            }
        }
        return false;
    }
    bool isFiftyMoveRule() {
        return halfmoveClockHistory[plycount] >= 100;  // 50 moves by each player
    }
    bool isRepeatedPosition(uint64_t key) {
        int currentHalfmove = halfmoveClockHistory[plycount];
        
        // Start from the current position and go back
        // We only need to check as far back as the current halfmove clock value
        for (int i = plycount - 2; i >= 0 && i >= plycount - currentHalfmove; i -= 2) {
            if (zobristKeyHistory[i] == key) {
                return true;
            }
        }
        
        return false;
    }
    
    // Utility functions
    inline uint8_t getPieceOfSquare(uint8_t square) {
        uint64_t squareBB = 1ULL << square; 
        if (squareBB & (whitePawns | blackPawns)) return 1;
        if (squareBB & (whiteKnights | blackKnights)) return 2;
        if (squareBB & (whiteBishops | blackBishops)) return 3;
        if (squareBB & (whiteRooks | blackRooks)) return 4;
        if (squareBB & (whiteQueens | blackQueens)) return 5;
        if (squareBB & (whiteKing | blackKing)) return 6;
        return 0;
    }
    bool rightColor(uint8_t square){
        uint64_t squareBB = 1ULL << square;
        return (whiteToMove && (whitePieces & squareBB)) || (!whiteToMove && (blackPieces & squareBB));
    }
    uint16_t inputMove(uint8_t from, uint8_t to, uint8_t promotionPiece = 0){
        uint16_t data = (from & 0x3F) | ((to & 0x3F) << 6);

        uint8_t movedPiece = getPieceOfSquare(from);
        bool isCastling = false;
        if (movedPiece == 6){
            if (whiteToMove){
                if (from == 4){
                    if ((to == 6) | (to == 2)){
                        isCastling = true;
                    }
                } 
            } else{
                if (from == 60){
                    if ((to == 62) | (to == 58)){
                        isCastling = true;
                    }
                } 
            }
        }
        data |= isCastling << 12;

        bool isPromotion = false;
        if (movedPiece == 1){
            if (whiteToMove){
                if (to > 55){
                    isPromotion = true;
                }
            } else{
                if (to < 8){
                    isPromotion = true;
                }
            }
        }
        data |= isPromotion << 15;

        if (isPromotion){
            std::cout << "Promotion piece: " << std::to_string(promotionPiece) << std::endl;
            data |= (promotionPiece  & 0x3) << 13;
        }
        return data;
    }
    uint16_t generateMove(int from, int to){
        std::vector<uint16_t> allMoves = generateAllLegalMoves();
        for (uint16_t move : allMoves) {
            if ((from == (move & 0x3F)) && (to == ((move >> 6) & 0x3F))) {
                return move;
            }
        }
        return 0; // Return 0 if no matching move is found
    }
    std::vector<uint16_t> generateLegalMovesOfSquare(int from) {
        std::vector<uint16_t> allMoves = generateAllLegalMoves();
        std::vector<uint16_t> movesOfSquares;

        for (uint16_t move : allMoves) {
            if (from == (move & 0x3F)) {
                movesOfSquares.emplace_back(move);
            }
        }

        return movesOfSquares;
    }
    std::vector<uint64_t> reportBitboards(){
        std::vector<uint64_t> bitboards;
        bitboards.push_back(whitePawns);
        bitboards.push_back(blackPawns);
        bitboards.push_back(whiteKnights);
        bitboards.push_back(blackKnights);
        bitboards.push_back(whiteBishops);
        bitboards.push_back(blackBishops);
        bitboards.push_back(whiteRooks);
        bitboards.push_back(blackRooks);
        bitboards.push_back(whiteQueens);
        bitboards.push_back(blackQueens);
        bitboards.push_back(whiteKing);
        bitboards.push_back(blackKing);
        return bitboards;
    }
    std::vector<uint16_t> returnMoveHistory(){
        return std::vector<uint16_t>(moveHistory, moveHistory + plycount);
    }
    uint16_t getLastMove(){
        if (plycount == 0 || plycount == 1){
            return 0;
        }
        return moveHistory[plycount-2];
    }
    bool isCapture(uint16_t move){
        return (whiteToMove ? blackPieces : whitePieces) & (1Ull << (move >> 6) & 0x3F);
    }
    std::vector<float> generateSparseVector(){
        std::vector<float> sparseVector;
        

        return sparseVector;
    }

    // Move functions
    static inline uint8_t getFrom(uint16_t move) {return move & 0x3F;}
    static inline uint8_t getTo(uint16_t move) {return (move >> 6) & 0x3F;}
    static inline bool isCastling(uint16_t move) {return (move >> 12) & 0x1;}
    static inline uint8_t getPromotedPiece(uint16_t move) {return (move >> 13) & 0x3;}
    static inline bool isPromotion(uint16_t move) {return (move >> 15) & 0x1;}
    
    // Print functions
    void print(std::ostream& out = std::cout) const {
        const char* pieces = "pnbrqkPNBRQK";
        const char* files = "abcdefgh";
        const char* ranks = "12345678";

        out << "\n  +---+---+---+---+---+---+---+---+\n";

        for (int rank = 7; rank >= 0; --rank) {
            out << rank + 1 << " |";
            for (int file = 0; file < 8; ++file) {
                int square = rank * 8 + file;
                char piece = '.';
                for (int i = 0; i < 12; ++i) {
                    uint64_t bb = 0ULL;
                    switch (i) {
                        case 0: bb = whitePawns; break;
                        case 1: bb = whiteKnights; break;
                        case 2: bb = whiteBishops; break;
                        case 3: bb = whiteRooks; break;
                        case 4: bb = whiteQueens; break;
                        case 5: bb = whiteKing; break;
                        case 6: bb = blackPawns; break;
                        case 7: bb = blackKnights; break;
                        case 8: bb = blackBishops; break;
                        case 9: bb = blackRooks; break;
                        case 10: bb = blackQueens; break;
                        case 11: bb = blackKing; break;
                    }
                    if (bb & (1ULL << square)) {
                        piece = pieces[i];
                        break;
                    }
                }
                out << " " << piece << " |";
            }
            out << "\n  +---+---+---+---+---+---+---+---+\n";
        }

        out << "    a   b   c   d   e   f   g   h\n\n";

        // out << "Turn: " << (whiteToMove ? "White" : "Black") << "\n";
        // out << "Castling rights: ";
        // uint8_t castlingRights = castlingRightHistory[plycount];
        // if (castlingRights & 8) out << "K";
        // if (castlingRights & 4) out << "Q";
        // if (castlingRights & 2) out << "k";
        // if (castlingRights & 1) out << "q";
        // if (castlingRights == 0) out << "-";
        // out << "\n";

        // uint8_t enPassantFile = enPassantFileHistory[plycount];
        // out << "enPassantFile: " << std::to_string(enPassantFile) << "\n";         
        // out << "Halfmove clock: " << halfmoveClockHistory[plycount] << "\n";
        // out << "Fullmove number: " << fullmoveNumber << "\n";
    }
    void printBitboard(uint64_t bitboard) {
        for (int rank = 7; rank >= 0; --rank) {
            std::cout << "  +---+---+---+---+---+---+---+---+\n";
            std::cout << rank + 1 << " |";
            for (int file = 0; file < 7; ++file) {
                int square = rank * 8 + file;
                std::cout << (bitboard & (1ULL << square) ? " 1  " : "    ");
            }
            int square = rank * 8 + 7;
            std::cout << (bitboard & (1ULL << square) ? " 1 " : "   ");
            std::cout << "|\n";
        }
        std::cout << "  +---+---+---+---+---+---+---+---+\n";
        std::cout << "    a   b   c   d   e   f   g   h\n\n";
    }
    void printAll(){
        std::cout << "whitePawns: " << std::endl;
        printBitboard(whitePawns);
        std::cout << "blackPawns: " << std::endl;
        printBitboard(blackPawns);
        std::cout << "whiteKnights: " << std::endl;
        printBitboard(whiteKnights);
        std::cout << "blackKnights: " << std::endl;
        printBitboard(blackKnights);
        std::cout << "whiteBishops: " << std::endl;
        printBitboard(whiteBishops);
        std::cout << "blackBishops: " << std::endl;
        printBitboard(blackBishops);
        std::cout << "whiteRooks: " << std::endl;
        printBitboard(whiteRooks);
        std::cout << "blackRooks: " << std::endl;
        printBitboard(blackRooks);
        std::cout << "whiteQueens: " << std::endl;
        printBitboard(whiteQueens);
        std::cout << "blackQueens: " << std::endl;
        printBitboard(blackQueens);
        std::cout << "whiteKing: " << std::endl;
        printBitboard(whiteKing);
        std::cout << "blackKing: " << std::endl;
        printBitboard(blackKing);
        std::cout << "whitePieces: " << std::endl;
        printBitboard(whitePieces);
        std::cout << "blackPieces: " << std::endl;
        printBitboard(blackPieces);
        std::cout << "allOccupied: " << std::endl;
        printBitboard(allOccupied);

        std::cout << "whiteToMove: " << whiteToMove << std::endl;
        std::cout << "castlingRights: " << std::bitset<4>(castlingRightHistory[plycount]) << std::endl;
        std::cout << "enPassantFile: " << std::to_string(enPassantFileHistory[plycount]) << std::endl;
        std::cout << "halfmoveClock: " << halfmoveClockHistory[plycount] << std::endl;
        std::cout << "fullmoveNumber: " << fullmoveNumber << std::endl;
    }
    std::string moveToString(uint16_t move) {
        uint8_t from = move & 0x3F;
        uint8_t to = (move >> 6) & 0x3F;
        bool isCastling = (move >> 12) & 0x1;
        uint8_t promotedPiece = (move >> 13) & 0x3;
        bool isPromotion = (move >> 15) & 0x1;

        static const std::array<char, 8> files = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
        static const std::array<char, 8> ranks = {'1', '2', '3', '4', '5', '6', '7', '8'};
        static const std::array<char, 4> promotionPieces = {'n', 'b', 'r', 'q'};

        std::string moveStr;

        // Add 'from' square
        moveStr += files[from % 8];
        moveStr += ranks[from / 8];

        // Add 'to' square
        moveStr += files[to % 8];
        moveStr += ranks[to / 8];

        // Handle promotion
        if (isPromotion) {
            moveStr += promotionPieces[promotedPiece];
        }

        // Handle castling (optional, as the king's move already indicates castling)
        if (isCastling) {
            if (to % 8 > from % 8) {
                moveStr += " O-O";   // Kingside castling
            } else {
                moveStr += " O-O-O"; // Queenside castling
            }
        }

        return moveStr;
    }
    int getPlycount(){return plycount;};
    void setPlycount(int x){plycount = x;};
};