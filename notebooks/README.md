# UCC Notebooks

This directory contains Jupyter notebooks demonstrating UCC (Unitary Compiler Collection) usage and capabilities.

## Available Notebooks

### [getting_started_with_ucc.ipynb](getting_started_with_ucc.ipynb)
A comprehensive getting started guide that demonstrates:
- Basic UCC compilation
- Hardware-aware optimization for different device topologies
- Performance comparisons with Qiskit's built-in transpiler
- Custom pass creation
- Real-world example with Quantum Fourier Transform
- Visualization of optimization results

**Key Features Demonstrated:**
- Circuit depth and size optimization
- Different coupling map topologies (linear, grid, all-to-all)
- Custom gate set compilation
- Multiple output formats
- Performance benchmarking

## Running the Notebooks

1. Install UCC and dependencies:
   ```bash
   git clone https://github.com/unitaryfoundation/ucc.git
   cd ucc
   uv sync --all-extras --all-groups
   ```

2. Launch Jupyter:
   ```bash
   uv run jupyter notebook notebooks/
   ```

3. Open the desired notebook and run the cells

## Requirements

- Python 3.12+
- UCC
- Qiskit
- Matplotlib (for visualizations)
- Jupyter

## Contributing

Feel free to add more example notebooks demonstrating UCC's capabilities with different:
- Quantum algorithms
- Hardware backends
- Optimization techniques
- Custom passes

For questions or suggestions, please open an issue on the main UCC repository.
