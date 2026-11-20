# nevresim
## A Neuromorphic Chip Simulator for Spiking Neural Networks
[![CodeFactor](https://www.codefactor.io/repository/github/dyigitpolat/nevresim/badge)](https://www.codefactor.io/repository/github/dyigitpolat/nevresim)  

nevresim is a powerful and flexible simulation tool for spiking neural networks. With support for both real-valued and spiking execution, as well as options for deterministic or stochastic spike generation, nevresim offers a wide range of customization options to suit your simulation needs.

## Features

- Configurable chip connectivity at compile-time
- Real-valued and spiking execution support
- Deterministic and stochastic spike generation support
- Customizable firing strategy, including support for the Novena chip's reset-to-zero strategy

## Compile-time Customization Points

- Firing strategy
- Spike generation type (deterministic or stochastic)
- Chip connectivity

## Usage

- Create a chip configuration (see include/_tests/xor_test for reference)
- Build nevresim
- Provide input files and execute the simulation

## Contributing

Contributions to nevresim are welcome and encouraged. 

## License

nevresim is open-source software released under the MIT License.
