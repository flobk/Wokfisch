// clang++ -O3 -march=native -std=c++17 -o Test.out TestSuite.cpp && ./Test.out
// or
// clang++ -O1 -march=native -std=c++14 -o Test.exe TestSuite.cpp && ./Test.exe

// runs a bunch of perft tests from https://www.chessprogramming.org/Perft_Results
#include "Board.hpp"
#include <iostream>
#include <ctime>
#include <chrono>

// uint64_t perft(Board& board, int depth) {
//     if (depth == 0) {
//         return 1;
//     }
//     uint64_t nodes = 0;
//     std::vector<Move> moves = board.generateAllLegalMoves();
//     for (const Move& move : moves) {
//         board.makeMove(move);
//         uint64_t child_nodes = perft(board, depth - 1);
//         nodes += child_nodes;
//         board.unmakeMove();
//     }
//     return nodes;
// }

// without bulk leaf counting (pure recursive)
inline uint64_t perft(Board& board, int depth) {
    if (depth == 0) {
        return 1;
    }
    uint64_t nodes = 0;
    std::vector<uint16_t> moves = board.generateAllLegalMoves();
    for (const uint16_t& move : moves) {
        board.makeMove(move);
        nodes += perft(board, depth - 1);
        board.unmakeMove();
    }
    return nodes;
}
// // with bulk leaf counting
// inline uint64_t perft(Board& board, int depth) {
//     if (depth == 1) {
//         return board.generateAllLegalMoves().size();
//     }
//     uint64_t nodes = 0;
//     std::vector<uint16_t> moves = board.generateAllLegalMoves();
//     for (const uint16_t& move : moves) {
//         board.makeMove(move);
//         uint64_t child_nodes = perft(board, depth - 1);
//         nodes += child_nodes;
//         board.unmakeMove();
//     } 
//     return nodes;
// }

struct TestPosition {
    std::string fen;
    std::vector<uint64_t> correct_moves;
    int max_depth;
};

int main() {
    std::vector<TestPosition> test_positions = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", {1, 20, 400, 8902, 197281, 4865609, 119060324, 3195901860}, 6}, // 6
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", {1, 48, 2039, 97862, 4085603, 193690690, 8031647685}, 5}, // 5
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", {1, 14, 191, 2812, 43238, 674624, 11030083, 178633661, 3009794393}, 7}, // 7
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", {1, 6, 264, 9467, 422333, 15833292, 706045033}, 6}, // 6
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", {1, 44, 1486, 62379, 2103487, 89941194}, 5}, // 5
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", {1, 46, 2079, 89890, 3894594, 164075551, 6923051137}, 5} // 5
     };
     
     // on mac m2
     // 25 s p6 stockfish
     // 40 s p6 mine
     // on windows 5800x
      // 

    uint64_t total_nodes = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (size_t pos = 0; pos < test_positions.size(); ++pos) {
        std::cout << "Testing position " << pos + 1 << std::endl;
        Board board(test_positions[pos].fen);
        
        for (int i = 1; i <= test_positions[pos].max_depth; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            uint64_t nodes = perft(board, i);
            total_nodes += nodes; // Sum up all nodes
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "Depth " << i << ": " << nodes << " nodes, Time: " 
                      << duration.count() << " milliseconds";
            
            if (nodes == test_positions[pos].correct_moves[i]) {
                std::cout << " - Correct";
            } else {
                std::cout << " - Incorrect (Expected: " << test_positions[pos].correct_moves[i] << ")";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    std::cout << "Total nodes: " << total_nodes << ", Nodes per second: " 
              << (total_nodes / total_duration.count()) << std::endl;
    std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
}