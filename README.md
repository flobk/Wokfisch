# “Wokfisch”
- an homage to “stockfish”, aka. the strongest chess engine
- “Wokfisch” gets cooked by “stockfish” ..
- .. but is still strong enough to beat >99% of players
- roughly ~2500 Elo, (for example it draws against the “Carlsen Bot” on chess.com)
- written in C++, frontend in python
- move generation written by myself
    - helpful reference: https://web.archive.org/web/20250910134338/https://www.codeproject.com/articles/Worlds-fastest-Bitboard-Chess-Movegenerator (original website doesnt exist anymore)
    - code calculates legal chess positions fast (see TestSuite.cpp)
    - I achieve the following N positions per second:
      
    | Processor (single core) | with bulk-leaf counting | without bulk-leaf counting |
    | --- | --- | --- |
    | Ryzen 5800X3D | 150M | 20M |
    | Apple M2 | 170M | 24M |
    - Bitboard-based, magic Bitboards for sliders, pre-calculated pins and check masks. FEN notation, 16bit move encoding, makeMove / unmakeMove for traversing the search tree
- the chess engine itself is inspired by https://www.youtube.com/watch?v=U4ogK0MIzqk and https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results/blob/main/Bots/Bot_514.cs (Gediminas Masaitis)
    - Search: NegaMax with Alpha-Beta Pruning, iterative Deepening, Aspiration windows, Quiescence search, zobrist hashing, NMP/LMR/LMP, MVV-LVA, etc.
    - Eval: basic PSQT, game phase calculation
    - I plan to use a Neural Net (NNUE) in the future for eval, the implementation is halfway done

---
- To get started navigate into the backend dir and git clone pybind11 (github.com/pybind/pybind11.git)
- adjust the backend/CMakeLists.txt to your python version and path
- create a build directory and navigate into it 
- use 'cmake ..' and then 'make' (or 'cmake --build .') to build
- navigate back to the main directory and run main.py
- in the beginning of the main.py, one can adjust starting position, which side plays white, etc.
