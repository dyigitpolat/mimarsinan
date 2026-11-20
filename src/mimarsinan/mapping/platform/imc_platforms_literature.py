"""The 15 literature-sourced IMC platforms (structured-elimination paper W5 handoff, two added at TCAD CL2, one at U-C).

Pure data: every ``register_imc_platform(IMCPlatform(...))`` call below is
transcribed EXACTLY (geometry, weight_bits, provenance quote + location) from
the extraction cards of ``13_chip_geometries.json``: the first 12 from the
structured-elimination workspace (research-tree tag
``checkpoint/structured-elimination-2026-07-29``), ``hermes_64`` and
``puma_node_like`` from the CL2 cards that extend it, ``loihi2_like`` from the card U-C promoted there
(``papers/mimarsinan_tcad_2026/research_artifacts/11_wave2/explore/cl2/chip_geometries/``).
[U-CHIPKEYS] ``has_bias`` is a key that changes the computation, which a chip description states from its source:
loihi_dense_equiv_128x1024 and hermes_64 state the on-chip bias their sources describe, each quote appended to the
provenance (the extraction cards' ``has_bias`` false was a modeling default, not a quote); [U-CHIPKEYS2]
hermes_core_like's is an assumption named in its description (its source states no bias). [U-CHIPKEYS2]
``weight_bits`` is the composed weight one synapse represents: prime_mat_256x256's card gave one cell's precision.
Do not round or invent a number here — add a new extraction card to that JSON
and re-transcribe instead. ``claim_eligibility`` mirrors the curated table in
the sibling ``13_chip_geometries.md`` §"S2 / allocation-claim eligibility"
(charter gate G5, red-team VR#4/MS#2), the rule every entry follows.

Split out of ``imc_platforms.py`` (the registry logic + synthetic fixtures)
purely to keep both files under the repo's per-file LOC budget; importing
``imc_platforms`` always pulls this module in for its registration side
effects, so callers never import this module directly.
"""

from __future__ import annotations

from mimarsinan.mapping.platform.imc_platforms import IMCPlatform, register_imc_platform

register_imc_platform(
    IMCPlatform(
        name="truenorth_like",
        cores=(
            {"max_axons": 256, "max_neurons": 256, "count": 4096, "has_bias": True},
        ),
        weight_bits=1,
        provenance="merolla2014a: \"From a structural view, the basic building block is a core, a self-contained neural network with 256 input lines (axons) and 256 outputs (neurons) connected via 256 × 256 directed, programmable synaptic connections (Fig. 2D).\" (main text, page 3 (S1/'From a structural view' paragraph))",
        capabilities={"allow_scheduling": True},
        claim_eligibility="headline-eligible",
    )
)

register_imc_platform(
    IMCPlatform(
        name="neurram_like",
        cores=(
            {"max_axons": 256, "max_neurons": 256, "count": 48, "has_bias": False},
        ),
        weight_bits=4,
        provenance="wan2022a: \"Central to each core is a TNSA consisting of 256×256 RRAM cells and 256 CMOS neuron circuits that implement analogue-to-digital converters (ADCs) and activation functions.\" (p.505, 'Reconfigurable RRAM-CIM architecture' section) | wan2022a: \"we present NeuRRAM—a RRAM-based CIM chip that simultaneously delivers versatility in reconfiguring CIM cores for diverse model architectures, energy efficiency that is two-times better than previous state-of-the-art RRAM-CIM chips across various computational bit-precisions, and inference accuracy comparable to software models quantized to four-bit weights across various AI tasks\" (p.504, Abstract)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="headline-eligible",
    )
)

register_imc_platform(
    IMCPlatform(
        name="hermes_core_like",
        cores=(
            {"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": True},
        ),
        weight_bits=8,
        provenance="khaddamaljameh2022hermescorea: \"Central to its architecture is an array of 256 × 256 8T4R unit cells.\" (Section II, p.1028) | khaddamaljameh2022hermescorea: \"These units apply the scaling and offset factors and also subtract positive and negative ADC values.\" (Section III-C, the LDPU's FP16 multiply-add units; [U-CHIPKEYS2] has_bias is assumed by name, not read: the core's paper states the LDPU's scaling and offset factors and an inference-required affine scaling, never a bias, and the 64-core chip built from these cores adds the bias in the LDPU, gallo2022a's inference experiments, PDF p.5)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="quarantined",
    )
)

register_imc_platform(
    IMCPlatform(
        name="isaac_like",
        cores=(
            {"max_axons": 128, "max_neurons": 128, "count": 16128, "has_bias": False},
        ),
        weight_bits=16,
        provenance="shafiee2016isaac: \"The optimal design point has 8 128×128 arrays, 8 ADCs per IMA, and 12 IMAs per tile. We refer to this design as ISAAC-CE.\" (Section VII (Results, 'Design Space Exploration'), p.9 col2)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="curve-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="meng_subarray_144x32",
        cores=(
            {"max_axons": 144, "max_neurons": 32, "count": 4096, "has_bias": False},
        ),
        weight_bits=4,
        provenance="meng2021structured: \"We set the memory size as 144 × 32 to avoid the memory waste in shallow layers.\" (Section II-C, p.1578)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="occupancy-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="xformer_pe_like",
        cores=(
            {"max_axons": 128, "max_neurons": 128, "count": 1728, "has_bias": False},
        ),
        weight_bits=8,
        provenance="sridharan2023xformer: \"Crossbar Dimension 128x128\" (Table I, page 7)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="curve-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="admm_crossbar_128x64",
        cores=(
            {"max_axons": 128, "max_neurons": 64, "count": 2048, "has_bias": False},
        ),
        weight_bits=8,
        provenance="yuan2019an: \"We use 128×64 crossbar size on ResNet-18 and VGG-16, where ConvNet and LeNet-5 uses 32×32 crossbar size.\" (Section IV, page 4 (start of Experimental Results)) | ma2019tiny: \"we limited our design by using multiple 128×64 [25] crossbars for all DNN layers\" (Sec 4.3, p.3)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="occupancy-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="recom_64x64_256",
        cores=(
            {"max_axons": 64, "max_neurons": 64, "count": 256, "has_bias": False},
        ),
        weight_bits=8,
        provenance="ji2018recom: \"Crossbar Size 64 x 64 ... Number of PEs 16 ... Number of Crossbar 256\" (Table II, Simulation Configuration, p.239)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="headline-eligible",
    )
)

register_imc_platform(
    IMCPlatform(
        name="prime_mat_256x256",
        cores=(
            {"max_axons": 256, "max_neurons": 256, "count": 128, "has_bias": False},
        ),
        weight_bits=9,
        provenance="chi2016prime: \"There are 2 FF subarrays and 1 Buffer subarray per bank (totally 64 subarrays). In FF subarrays, for each mat, there are 256×256 ReRAM cells and eight 6-bit reconfigurable SAs; for each ReRAM cell, we assume 4-bit MLC for computation while SLC for memory; the input voltage has 8 levels (3-bit) for computation while 2 levels (1-bit) for memory.\" (Section V.A 'PRIME Configurations', p.36) | chi2016prime: \"two 4-bit cells to represent one 8-bit synaptic weight\" (Section III-D 'Overcoming the Precision Challenge', PDF p.7) | chi2016prime: \"we employ two crossbar arrays store positive and negative weights, respectively\" (Section III-A, PDF p.5; [U-CHIPKEYS2] weight_bits, the composed weight one synapse represents: an 8-bit magnitude in two 4-bit cells, its sign held by the crossbar pair, the symmetric signed grid of 9 bits; the extraction card's 4 was one cell's precision)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="curve-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="ibm_analog_ai_512x512x34",
        cores=(
            {"max_axons": 512, "max_neurons": 512, "count": 34, "has_bias": False},
        ),
        weight_bits=8,
        provenance="ambrogio2023an: \"A micrograph of the chip is shown in Fig. 1c, highlighting the 2D grid of 34 analog tiles, each of which has its own 512 × 2,048 PCM crossbar array.\" (Chip architecture section, p.769 (Nature vol.620)) | ambrogio2023an: \"Alternatively, they can have a 2-PCM-per-weight configuration, which achieves a higher density. By reading different input frames through weights W_P1 or W_P2, a single tile can map 1,024 × 512 weight layers.\" (Fig. 1 caption (panel h), p.769)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="headline-eligible",
    )
)

register_imc_platform(
    IMCPlatform(
        name="loihi_dense_equiv_128x1024",
        cores=(
            {"max_axons": 128, "max_neurons": 1024, "count": 128, "has_bias": True},
        ),
        weight_bits=8,
        provenance="davies2018loihi: \"Loihi features a manycore mesh comprising 128 neuromorphic cores, three embedded x86 processor cores, and off-chip communication interfaces that hierarchically extend the mesh in four planar directions to other chips.\" (p.86, 'Chip Overview') | davies2018loihi: \"Each neuromorphic core implements 1,024 primitive spiking neural units (compartments) grouped into sets of trees constituting neurons.\" (p.86, 'Chip Overview') | davies2018loihi: \"The total number of neurons assigned to any core may not exceed 1,024 (Ncx). The total synaptic fan-in state mapped to any core must not exceed 128 KB (Nsyn × 64b, subject to compression and list alignment considerations). The total number of core-to-core fan-out edges mapped to any given core must not exceed 4,096 (Naxout)... The total number of distribution lists, associated by axon_id, in any core must not exceed 4,096 (Naxin)... In practice, constraints 2 and 4 tend to be the most limiting.\" (p.89, numbered constraint list under 'Network Connectivity Architecture') | davies2018loihi: \"The synaptic response current is the sum of filtered input spike trains and a constant bias current\" ('Spiking Neural Unit', Eq. (1), p.83; [U-CHIPKEYS] has_bias: each compartment adds its bias on chip)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="curve-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="dynap_se_64x256x4",
        cores=(
            {"max_axons": 64, "max_neurons": 256, "count": 4, "has_bias": False},
        ),
        weight_bits=2,
        provenance="moradi2017a: \"the chip comprises four cores; each core comprises 256 neurons, and each neuron has a fan-out of 4k.\" (p.7, Sec. IV 'A multi-core neuromorphic processor prototype') | moradi2017a: \"Fig. 9: Block diagram of one of the 256 computing nodes in each core. Each node comprises 64 mixed memory words, consisting of 10-bit CAM and 2-bit SRAM cells, 64 pulse generators circuits (PG), 64 pulse extenders circuits (PE), 64 pulse decoders (DECs), 64×4 digital pulse to analog current converters, 4 DPI filters, one adaptive I&F neuron circuit, and one handshaking block (HS).\" (p.9, Fig. 9 caption)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="curve-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="hermes_64",
        cores=(
            {"max_axons": 256, "max_neurons": 256, "count": 64, "has_bias": True},
        ),
        weight_bits=4,
        provenance="gallo2022a: \"The fully-integrated chip features 64 256x256 AIMC cores interconnected via an on-chip communication network.\" (Abstract, p.1) | gallo2022a: \"Its 64 cores, each of size 1.2 mm × 1.16 mm, are placed in eight rows and eight columns.\" (Section I 'Chip architecture', p.2) | gallo2022a: \"Each core contains a PCM crossbar array capable of storing a 256 × 256 weight matrix and performing an analog MVM using the input activations provided from outside of the core.\" (Section I 'Chip architecture', p.2) | gallo2022a: \"Compared with an equivalent digital engine having 8-bit input/output precision and N-bit weight precision, the precision achieved with ODP is close to 3-bit weights. With TDP, a precision between 3-bit and 4-bit weights is achieved.\" (Section IV 'MVM accuracy', p.5) | gallo2022a: \"For the last dense layer, the bias is programmed into the affine scale registers of the LDPU so that it gets added to the MVM result.\" (Section 'Inference experiments', p.6; [U-CHIPKEYS] has_bias: the bias is added on chip)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="headline-eligible",
    )
)

register_imc_platform(
    IMCPlatform(
        name="puma_node_like",
        cores=(
            {"max_axons": 128, "max_neurons": 128, "count": 2208, "has_bias": False},
        ),
        weight_bits=16,
        provenance="ankit2019puma: \"MVMU ... # per core 2 ... dimensions 128×128 ... Core ... # per tile 8 ... Tile ... # per node 138\" (Table 3 'PUMA Hardware Characteristics', p.9) | ankit2019puma: \"We conservatively use 2 bits per device, and realize 16-bit MVM operations by combining 8 crossbars via bit-slicing [95], illustrated in Figure 2(b).\" (Section 3.2.1 'Precision Considerations', p.5) | ankit2019puma: \"Since all eight 2-bit crossbars of a 16-bit MVM operation are used simultaneously on the same input, we co-locate these 2-bit crossbars on the same core in the same MVMU\" (Section 3.2.2 'Crossbar Co-location and Input Sharing', p.5)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="curve-only",
    )
)

register_imc_platform(
    IMCPlatform(
        name="loihi2_like",
        cores=(
            {"max_axons": 128, "max_neurons": 1024, "count": 128, "has_bias": False},
        ),
        weight_bits=8,
        provenance="intel2021loihi2: \"Up to 128 KB synaptic memory\" and \"Up to 8192 neurons\" (Figure 1, 'Neuromorphic Core' callout, p.5) | intel2021loihi2: \"each Loihi 2 chip consists of microprocessor cores and up to 128 fully asynchronous neuron cores connected by a network-on-chip (NoC)\" ('Base Architecture', p.5) | intel2021loihi2: \"Max # Neuron Cores/Chip 128\" (Table 2 'Comparison of Loihi to Loihi 2', Loihi 2 column, p.7)",
        capabilities={"allow_scheduling": True},
        claim_eligibility="curve-only",
    )
)
