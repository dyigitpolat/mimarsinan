#pragma once

/// The one timing decision of the compiled timing somas.
///
/// Every compiled soma of a timing program decides here: nevresim's quantized
/// timing neuron (``NeuronCompute<.., TTFSQuantizedCompute<S, Compare>>``) and
/// mimarsinan's SANA-FE quantized and per-cycle timing somas (the plugins
/// include this header). A soma builds its membrane with one of the builders
/// below and takes its fire step from ``fire_step``; it compares and rounds no
/// floating value by itself.
///
/// The rule is the Python executor's for its integral neurons
/// (``chip_simulation/ttfs/membrane_lattice.py`` and
/// ``models/spiking/wire_semantics.py:ttfs_quantized_staircase_lattice_np``):
/// a neuron whose weights, bias and threshold are integral and whose every
/// input read with a nonzero weight lies on the 1/(2S) grid has the exact
/// membrane V = m / (2S) with m an integer. m is formed in 64-bit integers
/// from each input's numerator round(a 2S), and the crossing is decided in
/// integers: the first step k with V + k theta / S >= theta (inclusive) or
/// > theta (strict). Every other neuron keeps its backend's own float path,
/// the program's own arithmetic, unchanged.
///
/// The reach: the lattice holds while sum |round(a 2S)| |w| + 2S |b| < 2^53,
/// the reference executor's bound read per sample on both sides, and theta
/// < 2^62, so 2 theta fits 64 bits (the code generation refuses a larger
/// integral threshold by name). Every integer formed here is then below 2^63.
///
/// The header is self-contained C++17 (the plugins build at C++17, nevresim
/// at C++20) and constant-evaluable.

#include <cstdint>
#include <iterator>
#include <type_traits>

namespace nevresim::timing {

/// The decision's version. Every compiled soma that decides here exports it
/// (SANA-FE's timing somas: ``mimarsinan_timing_decision_abi``), and the
/// toolchain refuses by name a soma built from other decision sources
/// (``chip_simulation/timing_decision.py``).
inline constexpr int abi = 1;

/// Float dust on the 1/(2S) input grid (``membrane_lattice.GRID_DUST``).
inline constexpr double grid_dust = 1e-4;
/// Float dust on an integral weight, bias or threshold (``segment_arrays.INTEGRAL_DUST``).
inline constexpr double integral_dust = 1e-6;
/// float64's exact integer range, the reference's lattice bound (``membrane_lattice.EXACT_INTEGER_RANGE``).
inline constexpr double exact_integer_range = 9007199254740992.0;
/// The largest integral threshold whose double 2 theta fits 64 bits.
inline constexpr double threshold_range = 4611686018427387904.0;
/// The threshold floor of the backends' float paths (the executor's ``np.maximum(theta, 1e-12)``).
inline constexpr double threshold_floor = 1e-12;

namespace detail {

inline constexpr double two_52 = 4503599627370496.0;

constexpr double magnitude(double x) noexcept { return x < 0.0 ? -x : x; }

/// numpy's ``rint``: round half to even. Exact for every finite double.
constexpr double rint(double x) noexcept
{
    if (!(magnitude(x) < two_52)) return x;  // already integral (or NaN)
    const double t = static_cast<double>(static_cast<std::int64_t>(x));  // toward zero
    const double f = x - t;  // exact: Sterbenz
    if (f > 0.5) return t + 1.0;
    if (f < -0.5) return t - 1.0;
    if (f == 0.5 || f == -0.5)
    {
        const bool odd = (static_cast<std::int64_t>(t) % 2) != 0;
        return odd ? (f > 0.0 ? t + 1.0 : t - 1.0) : t;
    }
    return t;
}

/// ``rint`` by the 1.5 2^52 shift (one add and one subtract under the
/// default round-to-nearest-even) where |x| < 2^51; ``rint`` elsewhere.
constexpr double rint_fast(double x) noexcept
{
    constexpr double shift = 6755399441055744.0;
    if (magnitude(x) < 2251799813685248.0) return (x + shift) - shift;
    return rint(x);
}

/// The smallest integral double >= x.
constexpr double ceil(double x) noexcept
{
    if (!(magnitude(x) < two_52)) return x;
    const double t = static_cast<double>(static_cast<std::int64_t>(x));
    return (x > t) ? t + 1.0 : t;
}

constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept
{
    const std::int64_t q = a / b;
    return ((a % b != 0) && ((a < 0) != (b < 0))) ? q - 1 : q;
}

constexpr std::int64_t ceil_div(std::int64_t a, std::int64_t b) noexcept
{
    return -floor_div(-a, b);
}

} // namespace detail

/// True when x is integral up to the dust (the executor's ``|x - round(x)| <= INTEGRAL_DUST``).
constexpr bool integral(double x) noexcept
{
    return detail::magnitude(x - detail::rint(x)) <= integral_dust;
}

/// True when theta is an integral threshold the integer decision holds.
constexpr bool lattice_threshold(double theta) noexcept
{
    return integral(theta) && theta >= 1.0 && detail::rint(theta) < threshold_range;
}

/// True when the code generation must refuse theta: integral, and 2 theta beyond 64 bits.
constexpr bool beyond_range(double theta) noexcept
{
    return integral(theta) && !(detail::rint(theta) < threshold_range);
}

/// The membrane of one neuron as a timing soma decides on it.
struct Membrane
{
    bool lattice{false};   ///< decided on the integer program's own value
    std::int64_t m{0};     ///< the exact membrane's numerator over 2S (lattice)
    double v{0.0};         ///< the float membrane (every other neuron)
};

/// How a backend's float path reads a non-lattice membrane: nevresim ramps the
/// membrane by theta/S each step (``ramp``); SANA-FE's somas take the executor's
/// closed form ceil(S (1 - v / theta)) (``closed_form``).
enum class FloatPath { ramp, closed_form };

/// The integer decision: the fire step in [0, S) of the membrane m / (2S) on
/// the integer threshold theta >= 1, or -1 when it never fires in the window.
constexpr int lattice_fire_step(std::int64_t m, std::int64_t theta, int s, bool strict) noexcept
{
    const std::int64_t two_theta = 2 * theta;
    const std::int64_t k_raw = strict
        ? static_cast<std::int64_t>(s) + 1 - detail::ceil_div(m, two_theta)
        : static_cast<std::int64_t>(s) - detail::floor_div(m, two_theta);
    if (k_raw >= s) return -1;
    return k_raw < 0 ? 0 : static_cast<int>(k_raw);
}

/// nevresim's float path: the membrane ramped by theta / S per step, compared before each ramp.
constexpr int ramp_fire_step(double v, double theta, int s, bool strict) noexcept
{
    for (int step = 0; step < s; ++step)
    {
        if (strict ? (theta < v) : (theta <= v)) return step;
        v += theta / s;
    }
    return -1;
}

/// The threshold a float path divides by or ramps with: never below the floor (``std::max(theta, 1e-12)``).
constexpr double floored_threshold(double theta) noexcept
{
    return (theta < threshold_floor) ? threshold_floor : theta;
}

/// SANA-FE's float path: the executor's closed form on the floored threshold.
constexpr int closed_form_fire_step(double v, double theta, int s) noexcept
{
    const double safe = floored_threshold(theta);
    const double k_raw = detail::ceil(static_cast<double>(s) * (1.0 - v / safe));
    if (k_raw >= static_cast<double>(s)) return -1;
    return k_raw < 0.0 ? 0 : static_cast<int>(k_raw);
}

/// The fire step of a membrane: the integer decision on the lattice, the backend's float path elsewhere.
constexpr int fire_step(const Membrane& membrane, double theta, int s, bool strict, FloatPath path) noexcept
{
    if (membrane.lattice && lattice_threshold(theta))
    {
        return lattice_fire_step(membrane.m, static_cast<std::int64_t>(detail::rint(theta)), s, strict);
    }
    return path == FloatPath::ramp ? ramp_fire_step(membrane.v, theta, s, strict)
                                   : closed_form_fire_step(membrane.v, theta, s);
}

/// The latched output level (S - k) / S of a fire step, 0 when it never fires.
constexpr double level(int s, int step) noexcept
{
    return step < 0 ? 0.0 : static_cast<double>(s - step) / s;
}

/// True when a fire step lies within the window (the neuron fires).
constexpr bool in_window(int step) noexcept { return step >= 0; }

/// The constants of a neuron's lattice reading, taken once when its
/// parameters load: whether its weights, bias and threshold are integral
/// (the threshold in range), sum |round(w)| and 2S round(b).
struct LatticeNeuron
{
    bool integral{false};
    double sum_abs_weights{0.0};
    double two_s_bias{0.0};
};

template <typename Weights>
constexpr LatticeNeuron lattice_neuron(const Weights& weights, double bias, double theta, int s) noexcept
{
    LatticeNeuron out{};
    if (!lattice_threshold(theta) || !integral(bias)) return out;
    double sum = 0.0;
    for (auto w = std::begin(weights); w != std::end(weights); ++w)
    {
        const double wd = static_cast<double>(*w);
        if (!integral(wd)) return out;
        sum += detail::magnitude(detail::rint(wd));
    }
    out.integral = true;
    out.sum_abs_weights = sum;
    out.two_s_bias = 2.0 * s * detail::rint(bias);
    return out;
}

/// The membrane of a neuron on the lattice from the input levels it reads
/// (``lattice_membrane``), in one pass: m = sum round(a 2S) round(w) +
/// 2S round(b) when the neuron is integral, every input it reads with a
/// nonzero weight lies on the 1/(2S) grid and the reference's bound holds
/// (sum |round(a 2S)| |round(w)| + |2S round(b)| < 2^53, under which the
/// double accumulation of these integers is exact); not on the lattice
/// otherwise, and the caller forms its float membrane.
namespace detail {

/// ``lattice_membrane`` for any input magnitude: rint per term, the
/// reference's bound summed exactly (the path of an input beyond 2^51 / 2S).
template <typename Weights, typename Signals>
constexpr Membrane lattice_membrane_any(
    const LatticeNeuron& neuron, const Weights& weights, const Signals& signals, int s) noexcept
{
    Membrane out{};
    const double two_s = 2.0 * s;
    double m = neuron.two_s_bias;
    double bound = magnitude(neuron.two_s_bias);
    auto w = std::begin(weights);
    auto a = std::begin(signals);
    for (; w != std::end(weights); ++w, ++a)
    {
        const double wd = static_cast<double>(*w);
        const double q = static_cast<double>(*a) * two_s;
        const double q_int = rint(q);
        if (wd != 0.0 && magnitude(q - q_int) > grid_dust) return out;
        m += q_int * rint(wd);
        bound += magnitude(q_int) * magnitude(rint(wd));
    }
    if (!(bound < exact_integer_range) || !(m == m)) return out;
    out.lattice = true;
    out.m = static_cast<std::int64_t>(m);
    return out;
}

} // namespace detail

template <typename Weights, typename Signals>
constexpr Membrane lattice_membrane(
    const LatticeNeuron& neuron, const Weights& weights, const Signals& signals, int s) noexcept
{
    using weight_value_t = std::remove_cv_t<std::remove_reference_t<decltype(*std::begin(weights))>>;
    constexpr double shift = 6755399441055744.0;  // 1.5 2^52: (q + shift) - shift is rint(q) for |q| < 2^51
    Membrane out{};
    if (!neuron.integral) return out;
    const double two_s = 2.0 * s;
    // Four independent lanes (the terms are integers, so any order sums them exactly under the bound):
    // m, the numerator; dev, the largest distance from the grid of an input read with a nonzero weight;
    // q, the largest |a 2S| (the shift's range, and the quick bound below).
    double m[4] = {neuron.two_s_bias, 0.0, 0.0, 0.0};
    double dev[4] = {0.0, 0.0, 0.0, 0.0};
    double q_lane[4] = {0.0, 0.0, 0.0, 0.0};
    const std::size_t n = static_cast<std::size_t>(std::end(weights) - std::begin(weights));
    const auto w = std::begin(weights);
    const auto a = std::begin(signals);
    auto term = [&](std::size_t j, std::size_t lane) {
        const double wd = static_cast<double>(w[j]);
        const double q = static_cast<double>(a[j]) * two_s;
        const double q_int = (q + shift) - shift;
        const double d = (wd != 0.0) ? detail::magnitude(q - q_int) : 0.0;
        dev[lane] = d > dev[lane] ? d : dev[lane];
        if constexpr (std::is_integral_v<weight_value_t>)
        {
            m[lane] += q_int * wd;
        }
        else
        {
            m[lane] += q_int * detail::rint_fast(wd);
        }
        const double q_abs = detail::magnitude(q);
        q_lane[lane] = q_abs > q_lane[lane] ? q_abs : q_lane[lane];
    };
    std::size_t j = 0;
    for (; j + 4 <= n; j += 4)
    {
        term(j, 0);
        term(j + 1, 1);
        term(j + 2, 2);
        term(j + 3, 3);
    }
    for (; j < n; ++j) term(j, 0);
    const double m_total = (m[0] + m[1]) + (m[2] + m[3]);
    double dev_max = 0.0;
    double q_max = 0.0;
    for (std::size_t lane = 0; lane < 4; ++lane)
    {
        dev_max = dev[lane] > dev_max ? dev[lane] : dev_max;
        q_max = q_lane[lane] > q_max ? q_lane[lane] : q_max;
    }
    if (!(q_max < 2251799813685248.0)) return detail::lattice_membrane_any(neuron, weights, signals, s);
    if (dev_max > grid_dust || !(m_total == m_total)) return out;
    if (!((q_max + 1.0) * neuron.sum_abs_weights + detail::magnitude(neuron.two_s_bias) < exact_integer_range))
    {
        return detail::lattice_membrane_any(neuron, weights, signals, s);
    }
    out.lattice = true;
    out.m = static_cast<std::int64_t>(m_total);
    return out;
}

/// The membrane of a neuron from its weights and the input levels it reads, in
/// one call (``lattice_neuron`` then ``lattice_membrane``); ``v`` is the float
/// membrane the backend formed, kept where the neuron is off the lattice.
template <typename Weights, typename Signals>
constexpr Membrane neuron_membrane(
    const Weights& weights, const Signals& signals, double bias, double theta, int s, double v) noexcept
{
    Membrane out = lattice_membrane(lattice_neuron(weights, bias, theta, s), weights, signals, s);
    out.v = v;
    return out;
}

/// The membrane of a soma handed its value by the stage (SANA-FE's preset):
/// the executor's numerator m over 2S when it decided the neuron on the
/// lattice, its float membrane otherwise.
constexpr Membrane preset_membrane(double v, bool lattice, double numerator) noexcept
{
    Membrane out{false, 0, v};
    if (!lattice) return out;
    out.lattice = true;
    out.m = static_cast<std::int64_t>(detail::rint(numerator));
    return out;
}

/// One single-spike arrival of summed weight ``current`` decoded at ``steps_left`` = S - k:
/// adds its integer numerator over S to ``numerator_over_s``.
constexpr void add_arrival(std::int64_t& numerator_over_s, double current, long steps_left) noexcept
{
    numerator_over_s += static_cast<std::int64_t>(detail::rint(current)) * static_cast<std::int64_t>(steps_left);
}

/// The membrane a per-cycle soma reconstructed from its arrivals: on the
/// lattice (the stage's flag) m = 2 (numerator over S) + 2S round(b).
constexpr Membrane arrivals_membrane(
    std::int64_t numerator_over_s, double bias, int s, bool lattice, double v) noexcept
{
    Membrane out{false, 0, v};
    if (!lattice) return out;
    out.lattice = true;
    out.m = 2 * numerator_over_s + static_cast<std::int64_t>(2.0 * s * detail::rint(bias));
    return out;
}

/// True when a compare policy is strict (``theta < v``): read on one exact tie.
template <typename ComparePolicy>
constexpr bool strict_compare() noexcept
{
    return !ComparePolicy::fires(1.0, 1.0);
}

} // namespace nevresim::timing
