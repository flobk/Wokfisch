#include <chrono>
#include <cstdint>

class Timer {
public:
    std::chrono::milliseconds timeRemaining;
    std::chrono::milliseconds startMoveTime;
    std::chrono::steady_clock::time_point turnStartTime;

    Timer(int64_t gameDuration) {
        timeRemaining = std::chrono::milliseconds(gameDuration);
    }

    int MillisecondsRemaining() const {
        return static_cast<int>(timeRemaining.count());
    }

    int MillisecondsElapsedThisTurn() const {
        if (turnStartTime.time_since_epoch().count() == 0) {
            return 0;
        }
        auto now = std::chrono::steady_clock::now();
        return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - turnStartTime).count());
    }

    void StartTurn() {
        turnStartTime = std::chrono::steady_clock::now();
    }

    void EndTurn() {
        timeRemaining -= std::chrono::milliseconds(MillisecondsElapsedThisTurn());
        if (timeRemaining < std::chrono::milliseconds(0)) {
            timeRemaining = std::chrono::milliseconds(0);
        }
    }
};
