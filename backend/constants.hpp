#pragma once
#include <cstdint>

// Constants
constexpr uint64_t FILE_A = 0x0101010101010101ULL;
constexpr uint64_t FILE_H = 0x8080808080808080ULL;
constexpr uint64_t RANK_2 = 0x0000000000000FF00ULL;
constexpr uint64_t RANK_4 = 0x00000000000FF000000ULL;
constexpr uint64_t RANK_5 = 0x000000000FF00000000ULL;
constexpr uint64_t RANK_7 = 0x000FF000000000000ULL;
constexpr uint64_t WKS_OCC =  (1ULL << 5) | (1ULL << 6);
constexpr uint64_t WKS_SEEN = (1ULL << 5) | (1ULL << 6);
constexpr uint64_t WQS_OCC =  (1ULL << 1) | (1ULL << 2) | (1ULL << 3);
constexpr uint64_t WQS_SEEN = (1ULL << 2) | (1ULL << 3);
constexpr uint64_t BKS_OCC =  (1ULL << 61) | (1ULL << 62);
constexpr uint64_t BKS_SEEN = (1ULL << 61) | (1ULL << 62);
constexpr uint64_t BQS_OCC =  (1ULL << 57) | (1ULL << 58) | (1ULL << 59);
constexpr uint64_t BQS_SEEN = (1ULL << 58) | (1ULL << 59);