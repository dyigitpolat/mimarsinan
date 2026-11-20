#pragma once

#include <limits>
#include <ostream>

namespace nevresim {

template <int S, int Latency, typename ComparePolicy>
class TTFSExecution;

/// The significant digits an execution's outputs are written to the host with.
///
/// Zero keeps the stream's default (six digits): spike counts are integers it
/// writes exactly, and a float program's values are its own. The quantized
/// timing execution writes its latched levels (S - k) / S, values the integer
/// program holds exactly as a ratio of integers, so it writes every digit a
/// double holds and the host reads back the level itself (six digits read 5/7
/// as 0.714286 and 127/128 as 0.992188, off the 1/(2S) grid from S 128 on).
template <typename Exec>
struct output_digits
{
    static constexpr int value = 0;
};

template <int S, int Latency, typename ComparePolicy>
struct output_digits<TTFSExecution<S, Latency, ComparePolicy>>
{
    static constexpr int value = std::numeric_limits<double>::max_digits10;
};

/// Sets ``os`` to write the outputs of ``Exec`` as declared above.
template <typename Exec>
void set_output_format(std::ostream& os)
{
    if constexpr (output_digits<Exec>::value > 0)
    {
        os.precision(output_digits<Exec>::value);
    }
}

} // namespace nevresim
