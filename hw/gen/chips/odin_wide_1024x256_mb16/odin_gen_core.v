// Copyright (C) 2016-2019 Université catholique de Louvain (UCLouvain), Belgium.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 2.0 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-2.0/. The software, hardware and materials
// distributed under this License are provided in the hope that it will be useful
// on an as is basis, without warranties or conditions of any kind, either
// expressed or implied; without even the implied warranty of merchantability or
// fitness for a particular purpose. See the Solderpad Hardware License for more
// detailed permissions and limitations.
//------------------------------------------------------------------------------
//
// STATEMENT OF CHANGES (Solderpad Hardware License v2.0, section 4(b)):
//   This is a Modified Work derived from ODIN, https://github.com/ChFrenkel/ODIN
//   at commit 1781931, Copyright (C) 2016-2019 UCLouvain.
//   Modified by the mimarsinan project, 2026-08-21.
//   Changes, all of them deliberate reductions or parameterizations of the
//   upstream architecture rather than new science:
//     * GEOMETRY IS PARAMETRIC. Upstream hard-wires 256 neurons, a 128-bit
//       neuron word, a 32-bit synapse word, a 13-bit synapse address and the
//       controller's sweep bounds (upstream `neuron_core.v`, `synaptic_core.v`,
//       `controller.v`). Here the axon count, the neuron count, the membrane /
//       threshold register width, its signedness and the synapse weight width
//       are module parameters, and every address width, memory depth and sweep
//       bound is derived from them.
//     * LIF ONLY, OPEN LOOP. The SDSP online-learning datapath, the Izhikevich
//       neuron, the burst-timeref mode, the calcium/leak state and the BIST
//       path are omitted entirely: a mimarsinan deployment programs
//       `SPI_OPEN_LOOP=1` with weight updates doubly disabled, so none of them
//       is ever exercised and carrying them would be untested silicon.
//     * THE CONFIGURATION INTERFACE IS A DIRECT SYNCHRONOUS WRITE PORT, not the
//       20-bit SPI slave. Upstream's SPI costs 40 SCK per transaction at
//       SCK <= CLK/4; the variants are cosimulated and (P7) driven by a host
//       kernel that already owns a synchronous bus, so the programming port is
//       one address/data pair per clock. The AER links, the event-serial sweep
//       and the soma arithmetic are unchanged in kind.
//     * THE SOMA LAW IS A PARAMETER. Upstream evaluates the threshold after
//       every synaptic event on an unsigned 8-bit register. `PER_EVENT`
//       selects between that law and the per-cycle sync-fire law (accumulate
//       the whole window, compare once at the time reference), `MSIGNED`
//       selects the two's-complement register, and `ASSERT_NO_SAT` makes a
//       rail a reported failure instead of a clamp for the law whose contract
//       says the rails are unreachable.
//     * THE SYNAPSE MEMORY IS A BLOCK RAM. Upstream leaves its arrays
//       behavioural and instructs the implementer to "replace them with SRAM
//       macros or Block RAM" (upstream doc Section 5); `syn_mem` below is that
//       substitution performed in the template -- one synchronous write port,
//       one REGISTERED read port, `ram_style = "block"` -- because a tile has
//       no asynchronous read port and an array read combinationally can only
//       land in SLICEM distributed RAM. The read address is the sweep position
//       ONE CYCLE AHEAD, so the datapath sees the same weight in the same cycle
//       as the asynchronous read did and no count moves. The neuron arrays
//       (`thr_arr`, `vmem_arr`) stay behavioural: `vmem_arr` is read AND
//       written in one cycle by the soma, which is not a tile access pattern.
//
//   GENERATED FILE -- do not edit. `mapping/export/odin_gen/` expands this
//   template from a `CoreSpec`, which is itself a projection of one declared
//   core type and the resolved `SomaLaw`; edit the template or the spec.
//
//------------------------------------------------------------------------------
//
// CROSS-LANGUAGE CONTRACT: this module executes the SAME soma law as
// `models/spiking/serial/fold.py` (PER_EVENT=1) / `models/spiking/lif_core_step.py`
// (PER_EVENT=0) and the matching nevresim integration policy. Every ordering
// decision below is the one written down there: ascending axon slots,
// occurrences of one slot adjacent (the host router delivers them so), saturate
// BEFORE the compare, reset per crossing.
//
// AER-in address: bit [AW] set = the all-neurons TIME REFERENCE; otherwise the
// low AW bits name the axon row whose weights sweep the neurons.
// AER-out address: the neuron that fired, one four-phase transaction per spike.

`timescale 1ns/1ps

module odin_gen_core #(
    parameter AXONS         = 1024,
    parameter NEURONS       = 256,
    parameter MBITS         = 16,
    parameter MSIGNED       = 0,
    parameter WBITS         = 8,
    parameter PER_EVENT     = 1,
    parameter RESET_ZERO    = 1,
    parameter CMP_INCL      = 1,
    parameter ASSERT_NO_SAT = 0
)(
    input  wire                 CLK,
    input  wire                 RST,

    // Direct synchronous configuration port (see the statement of changes).
    input  wire                 PROG_EN,
    input  wire [          1:0] PROG_SEL,
    input  wire [         31:0] PROG_ADDR,
    input  wire [         31:0] PROG_DATA,

    // Input AER link (four-phase).
    input  wire [    10:0]    AERIN_ADDR,
    input  wire                 AERIN_REQ,
    output reg                  AERIN_ACK,

    // Output AER link (four-phase).
    output reg  [  7:0]    AEROUT_ADDR,
    output reg                  AEROUT_REQ,
    input  wire                 AEROUT_ACK,

    // The generated spec, readable by the testbench so a harness that believes
    // a different geometry than the one generated fails loud instead of
    // silently truncating a port.
    output wire [         31:0] SPEC_AXONS,
    output wire [         31:0] SPEC_NEURONS,
    output wire [         31:0] SPEC_MBITS,
    output wire [         31:0] SPEC_WBITS,
    output wire [         31:0] SPEC_FLAGS,

    // Sticky: the membrane reached a rail of a register whose law says it
    // cannot. Never self-clearing -- the run's verdict is that it happened.
    output reg                  RAIL_TOUCHED
);

    localparam AW     = 10;                  // clog2(AXONS)
    localparam NW     = 8;                  // clog2(NEURONS)
    localparam CPWW   = 2; // log2 synapse cells per word
    localparam SWPR   = 64;   // 32-bit words per axon row
    localparam SDEPTH = 65536;
    localparam SAW    = 16;

    // One wide signed accumulator holds any (membrane + weight) before the
    // clamp, so the saturation is a decision and never a silent wrap.
    localparam ACCW = MBITS + WBITS + 2;
    localparam signed [ACCW-1:0] V_LO = 0;
    localparam signed [ACCW-1:0] V_HI = 65535;

    localparam S_IDLE   = 3'd0;
    localparam S_SWEEP  = 3'd1;
    localparam S_EMIT   = 3'd2;
    localparam S_EMITDN = 3'd3;
    localparam S_ACK    = 3'd4;

    localparam M_EVENT = 1'b0;
    localparam M_TREF  = 1'b1;

    (* ram_style = "block" *)
    reg  [31:0]      syn_mem  [0:SDEPTH-1];
    reg  [MBITS-1:0] thr_arr  [0:NEURONS-1];
    reg  [MBITS-1:0] vmem_arr [0:NEURONS-1];

    reg              gate;
    reg  [2:0]       state;
    reg              mode;
    reg  [AW-1:0]    axon;
    reg  [NW:0]      n;
    reg              aerin_req_sync_int, aerin_req_sync;

    integer          i;

    assign SPEC_AXONS   = AXONS;
    assign SPEC_NEURONS = NEURONS;
    assign SPEC_MBITS   = MBITS;
    assign SPEC_WBITS   = WBITS;
    assign SPEC_FLAGS   = {27'b0,
                           (ASSERT_NO_SAT != 0), (CMP_INCL != 0),
                           (RESET_ZERO != 0), (PER_EVENT != 0),
                           (MSIGNED != 0)};

    // Double-latching barrier on REQ, exactly as upstream's controller does.
    always @(posedge CLK) begin
        if (RST) begin
            aerin_req_sync_int <= 1'b0;
            aerin_req_sync     <= 1'b0;
        end else begin
            aerin_req_sync_int <= AERIN_REQ;
            aerin_req_sync     <= aerin_req_sync_int;
        end
    end

    wire             is_tref  = AERIN_ADDR[AW];
    wire [AW-1:0]    ev_axon  = AERIN_ADDR[AW-1:0];

    // Declared here, driven by the soma below: the next-state terms the memory
    // address register needs are written before the datapath that produces them.
    wire             fire_now;
    wire             last_neuron;

    // The sweep position ONE CYCLE AHEAD. Every place `axon` and `n` move in
    // the controller below is exactly one of these terms, so the block RAM's
    // address register holds the position the datapath will occupy when its
    // registered word arrives: `syn_word` is, in every cycle, the word an
    // asynchronous read of the CURRENT position would have returned.
    wire             accept    = (state == S_IDLE) && aerin_req_sync && !gate;
    wire             advance   = ((state == S_SWEEP)  && !fire_now && !last_neuron)
                              || ((state == S_EMITDN) && !AEROUT_ACK && !last_neuron);
    wire [AW-1:0]    axon_nxt  = RST ? {AW{1'b0}}
                               : ((accept && !is_tref) ? ev_axon : axon);
    wire [NW:0]      n_nxt     = (RST || accept) ? {(NW+1){1'b0}}
                               : (advance ? (n + 1'b1) : n);

    wire [31:0]      n_wide    = { {(32-(NW+1)){1'b0}}, n_nxt };
    wire [31:0]      syn_index = axon_nxt * SWPR + (n_wide >> CPWW);

    reg  [31:0]      syn_word;

    // The one write port and the one registered read port of the tile. The
    // programming port is the only writer, and it is exercised while the core
    // is gated, so no run reads a word in the cycle it is written.
    always @(posedge CLK) begin
        syn_word <= syn_mem[syn_index[SAW-1:0]];
        if (PROG_EN && (PROG_SEL == 2'd3))
            syn_mem[PROG_ADDR[SAW-1:0]] <= PROG_DATA;
    end

    wire [31:0]      cell_lsb  = n[CPWW-1:0] * WBITS;
    wire [WBITS-1:0] w_raw     = syn_word[cell_lsb +: WBITS];

    wire [MBITS-1:0] v_cur    = vmem_arr[n[NW-1:0]];
    wire [MBITS-1:0] thr_cur  = thr_arr[n[NW-1:0]];

    wire signed [ACCW-1:0] v_ext = MSIGNED
        ? $signed({ {(ACCW-MBITS){v_cur[MBITS-1]}}, v_cur })
        : $signed({ {(ACCW-MBITS){1'b0}},           v_cur });
    wire signed [ACCW-1:0] w_ext =
        $signed({ {(ACCW-WBITS){w_raw[WBITS-1]}}, w_raw });
    wire signed [ACCW-1:0] thr_ext =
        $signed({ {(ACCW-MBITS){1'b0}}, thr_cur });

    wire signed [ACCW-1:0] acc = (mode == M_EVENT) ? (v_ext + w_ext) : v_ext;
    wire signed [ACCW-1:0] sat =
        (acc < V_LO) ? V_LO : ((acc > V_HI) ? V_HI : acc);
    wire                   rail = (acc <= V_LO) || (acc >= V_HI);

    wire compares = (mode == M_TREF) || (PER_EVENT != 0);
    assign fire_now = compares
        && ((CMP_INCL != 0) ? (sat >= thr_ext) : (sat > thr_ext));

    wire signed [ACCW-1:0] reset_v =
        (RESET_ZERO != 0) ? {ACCW{1'b0}} : (sat - thr_ext);
    wire signed [ACCW-1:0] post_raw = fire_now ? reset_v : sat;
    wire signed [ACCW-1:0] post =
        (post_raw < V_LO) ? V_LO : ((post_raw > V_HI) ? V_HI : post_raw);
    wire [MBITS-1:0]       new_v = post[MBITS-1:0];

    assign last_neuron = (n == (NEURONS - 1));

    always @(posedge CLK) begin
        // The sweep position is driven ONLY from the next-state terms above --
        // the same terms the memory's address register sees, which is what
        // keeps the pipelined read exact.
        axon <= axon_nxt;
        n    <= n_nxt;
        if (RST) begin
            state        <= S_IDLE;
            mode         <= M_EVENT;
            gate         <= 1'b1;
            AERIN_ACK    <= 1'b0;
            AEROUT_REQ   <= 1'b0;
            AEROUT_ADDR  <= {NW{1'b0}};
            RAIL_TOUCHED <= 1'b0;
        end else begin
            if (PROG_EN) begin
                case (PROG_SEL)
                    2'd0: if (PROG_ADDR == 32'd0) gate <= PROG_DATA[0];
                    2'd1: thr_arr[PROG_ADDR[NW-1:0]]  <= PROG_DATA[MBITS-1:0];
                    2'd2: vmem_arr[PROG_ADDR[NW-1:0]] <= PROG_DATA[MBITS-1:0];
                    // 2'd3 (the synapse word) is written by the memory's own
                    // single write port above.
                    default: ;
                endcase
            end

            case (state)
                S_IDLE: begin
                    if (aerin_req_sync && !gate) begin
                        if (is_tref) begin
                            mode <= M_TREF;
                            // Under the event-serial law the time reference
                            // drives leak only, and leak is off in deployment:
                            // the event is consumed and changes nothing.
                            if (PER_EVENT != 0) begin
                                AERIN_ACK <= 1'b1;
                                state     <= S_ACK;
                            end else begin
                                state <= S_SWEEP;
                            end
                        end else begin
                            mode  <= M_EVENT;
                            state <= S_SWEEP;
                        end
                    end
                end

                S_SWEEP: begin
                    vmem_arr[n[NW-1:0]] <= new_v;
                    if ((ASSERT_NO_SAT != 0) && (mode == M_EVENT) && rail)
                        RAIL_TOUCHED <= 1'b1;
                    if (fire_now) begin
                        AEROUT_ADDR <= n[NW-1:0];
                        AEROUT_REQ  <= 1'b1;
                        state       <= S_EMIT;
                    end else if (last_neuron) begin
                        AERIN_ACK <= 1'b1;
                        state     <= S_ACK;
                    end
                end

                S_EMIT: begin
                    if (AEROUT_ACK) begin
                        AEROUT_REQ <= 1'b0;
                        state      <= S_EMITDN;
                    end
                end

                S_EMITDN: begin
                    if (!AEROUT_ACK) begin
                        if (last_neuron) begin
                            AERIN_ACK <= 1'b1;
                            state     <= S_ACK;
                        end else begin
                            state <= S_SWEEP;
                        end
                    end
                end

                S_ACK: begin
                    if (!aerin_req_sync) begin
                        AERIN_ACK <= 1'b0;
                        state     <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // The arrays have no reset, exactly as upstream's memories have none: the
    // programming port must fully initialise everything a run touches. This
    // block states that zero image for a SIMULATOR, which is where "no reset"
    // would otherwise read as X. A synthesizer needs no help -- a block RAM's
    // INIT costs nothing and the programming port overwrites it before any
    // run -- and yosys's Verilog frontend folds an initial loop over a deep
    // memory in quadratic time: at this chip's 65,536-word synapse depth,
    // 10 s of synthesis became 25 minutes of frontend (measured 2026-08-30).
    // `YOSYS` is defined by that frontend and by nothing else, so every
    // simulator and every vendor synthesizer still reads the block.
`ifndef YOSYS
    initial begin
        for (i = 0; i < SDEPTH;  i = i + 1) syn_mem[i]  = 32'd0;
        for (i = 0; i < NEURONS; i = i + 1) thr_arr[i]  = {MBITS{1'b0}};
        for (i = 0; i < NEURONS; i = i + 1) vmem_arr[i] = {MBITS{1'b0}};
    end
`endif

endmodule
