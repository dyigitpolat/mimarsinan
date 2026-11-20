#pragma once

#include "simulator/chip/chip.hpp"
#include "loaders/input_loader.hpp"
#include "loaders/spike_train_input_loader.hpp"
#include "loaders/weights_loader.hpp"
#include "common/types.hpp"
#include "common/constants.hpp"

#include <iostream>
#include <utility>
#include <iterator>
#include <algorithm>
#include <limits>
#include <fstream>
#include <memory>

namespace nevresim
{
namespace tests
{

constexpr nevresim::core_id_t in = nevresim::k_input_buffer_id;
constexpr nevresim::core_id_t off = nevresim::k_no_connection;
constexpr nevresim::core_id_t on = nevresim::k_always_on;

template <
    typename WeightType,
    auto connections,
    auto outputs,
    std::size_t axon_count,
    std::size_t neuron_count,
    std::size_t core_count,
    std::size_t input_size,
    std::size_t output_size,
    std::size_t max_spans_per_core,
    constant<MembraneLeak<WeightType>> leak,
    typename ComputePolicy
>
consteval auto generate_test_chip()
{

    using Cfg = nevresim::ChipConfiguration<
        WeightType,
        WeightType,
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        max_spans_per_core,
        leak
    >;

    using Map = Mapping<Cfg>;

    constexpr Map mapping{connections, outputs};
    
    using Chip = Chip<Cfg, mapping, ComputePolicy>;

    constexpr Chip chip{};
    return chip;
}

void print_prediction_summary(const auto& buffer)
{
    for(int q{}; auto x : buffer)
    {
        std::cout << "[" << q++ << ": " << x << "] ";
    }
    std::cout << "\n";
}

int argmax(const auto& buffer)
{
    return std::distance(
        buffer.begin(),
        std::max_element(buffer.begin(), buffer.end()));
}

inline void report(
    int guess, int target, int idx, int& correct, int& total)
{
    std::cout 
        << idx << ") prediction: " << guess
        << "\n";

    std::cout 
        << idx << ") target: " << target
        << "\n";

    std::cout << correct << "/" << total << "\n";
}

template <typename FloatType>
constexpr bool is_almost_equal(
    FloatType value_1, FloatType value_2)
{
    FloatType absolute_difference{
        (value_1 > value_2) ?
        (value_1 - value_2) :
        (value_2 - value_1)
    };

    return absolute_difference < 
        (std::numeric_limits<FloatType>::epsilon() * 2);
}

template <typename WeightType, typename ThresholdType, typename Chip>
void load_weights(Chip& chip, auto weights_filename)
{
    std::unique_ptr<WeightsLoader<Chip::config_, WeightType, ThresholdType>>
        weights_loader_ptr{new WeightsLoader<Chip::config_, WeightType, ThresholdType>{}};

    std::ifstream weights_stream(weights_filename);
    if(weights_stream.is_open())
    {
        weights_stream >> *weights_loader_ptr;
    }
    else
    {
        std::cerr << "Failed to open file: " << weights_filename << "\n";
        exit(1);
    }

    chip.load_weights(weights_loader_ptr->chip_weights_);
}

auto load_input_n(auto input_filename_prefix, int input_id)
{
    std::unique_ptr<InputLoader> input_loader_ptr{new InputLoader{}};
    
    std::string fname = 
        std::string{input_filename_prefix} +
        std::to_string(input_id) +
        std::string{".txt"};

    std::ifstream input_stream(fname);
    if(input_stream.is_open())
    {
        input_stream >> *input_loader_ptr;
    }
    else
    {
        std::cerr << "Failed to open file: " << fname << "\n";
        exit(1);
    }

    return std::pair{input_loader_ptr->input_, input_loader_ptr->target_};
}

auto load_spike_train_input_n(auto input_filename_prefix, int input_id)
{
    std::unique_ptr<SpikeTrainInputLoader> input_loader_ptr{
        new SpikeTrainInputLoader{}};

    std::string fname =
        std::string{input_filename_prefix} +
        std::to_string(input_id) +
        std::string{".txt"};

    std::ifstream input_stream(fname);
    if (input_stream.is_open()) {
        input_stream >> *input_loader_ptr;
    } else {
        std::cerr << "Failed to open file: " << fname << "\n";
        exit(1);
    }

    return std::pair{input_loader_ptr->spikes_, input_loader_ptr->target_};
}

void print_details(
    const auto& buffer, 
    int guess, int target, int idx, 
    int& correct, int& total)
{
    std::cout << "\n\n";
    print_prediction_summary(buffer);
    report(guess, target, idx, correct, total);
}

template <typename ExecutionPolicy>
auto test_on_inputs(
    auto& chip, 
    auto input_filename_prefix, 
    int input_count,
    bool verbose = false)
{
    int correct{};
    int total{};
    for(int idx = 0; idx < input_count; ++idx)
    {
        auto [input, target] = load_input_n(input_filename_prefix, idx);
        auto buffer = chip.template execute<ExecutionPolicy>(input);
        chip.reset();

        int guess = argmax(buffer);

        total++;
        if(guess==target) correct++;

        if (verbose) print_details(buffer, guess, target, idx, correct, total);
    }

    return std::pair{correct, total};
}

void report_result(auto title, auto test)
{
    std::cout << title << ": ";
    if(test()) 
        std::cout << "PASS\n";
    else 
        std::cout << "FAIL\n";
}

} // namespace nevresim::tests
} // namespace nevresim
