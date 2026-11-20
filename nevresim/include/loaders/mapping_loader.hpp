#pragma once

#include "simulator/chip_utilities.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>

namespace nevresim
{

inline core_id_t span_core_id_from_kind(char kind, core_id_t stored_core)
{
    if (kind == 'o')
        return k_no_connection;
    if (kind == 'i')
        return k_input_buffer_id;
    if (kind == 'a')
        return k_always_on;
    return stored_core;
}

inline char span_kind_from_core_id(core_id_t core)
{
    if (core == k_no_connection)
        return 'o';
    if (core == k_input_buffer_id)
        return 'i';
    if (core == k_always_on)
        return 'a';
    return 'c';
}

template<typename Config>
Mapping<Config> load_mapping_from_spans(std::istream& in)
{
    Mapping<Config> mapping{};

    std::size_t core_count{};
    std::size_t max_spans{};
    std::size_t output_size{};
    in >> core_count >> max_spans >> output_size;
    if (!in || core_count != Config::core_count_ || max_spans != Config::max_spans_per_core_
        || output_size != Config::output_size_)
    {
        throw std::runtime_error("mapping span header mismatch with chip configuration");
    }

    for (std::size_t c = 0; c < core_count; ++c)
    {
        std::size_t span_count{};
        in >> span_count;
        if (!in || span_count > max_spans)
            throw std::runtime_error("invalid span_count in mapping file");

        mapping.core_sources_[c].span_count_ = span_count;
        for (std::size_t s = 0; s < span_count; ++s)
        {
            char kind{};
            core_id_t stored_core{};
            axon_id_t start{};
            std::size_t count{};
            in >> kind >> stored_core >> start >> count;
            if (!in)
                throw std::runtime_error("truncated span entry in mapping file");

            mapping.core_sources_[c].spans_[s] = SourceSpan{
                span_core_id_from_kind(kind, stored_core),
                start,
                count,
            };
        }
    }

    for (std::size_t o = 0; o < output_size; ++o)
    {
        core_id_t core{};
        axon_id_t neuron{};
        in >> core >> neuron;
        if (!in)
            throw std::runtime_error("truncated output source in mapping file");
        mapping.output_sources_[o] = SpikeSource{core, neuron};
    }

    return mapping;
}

template<typename Config>
Mapping<Config> load_mapping_from_spans_file(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("failed to open mapping spans file: " + path);
    return load_mapping_from_spans<Config>(in);
}

} // namespace nevresim
