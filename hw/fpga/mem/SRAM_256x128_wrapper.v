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
//   This is a Modified Work derived from the `SRAM_256x128_wrapper` module of
//   "neuron_core.v" (lines 283-313) in ODIN, https://github.com/ChFrenkel/ODIN
//   at commit 1781931, Copyright (C) 2016-2019 UCLouvain.
//   Modified by the mimarsinan project, 2026-08-21.
//   Change: the upstream file marks that module's body as
//   "Simple behavioral code for simulation, to be replaced by a 256-word
//   128-bit SRAM macro or Block RAM (BRAM) memory with the same format for FPGA
//   implementations" (upstream doc/README.md Section 5). This file performs
//   exactly that substitution: the body is rewritten in the single-clock
//   read-first synchronous form Vivado/Yosys infer a Block RAM from, while the
//   module name, the port names, the port widths and the port directions are
//   left untouched so it is a drop-in for the vendored declaration. No other
//   change. The vendored tree itself is never edited.
//
// Behavioural equivalence to the replaced model (the P5 gate proves it):
//   * read latency 1 -- Q reflects the address presented one clock earlier;
//   * CS low HOLDS the previous Q (no new read is launched) rather than
//     returning zero or X;
//   * CS & WE writes D at that address on the SAME edge, and the value latched
//     into Q on that edge is the OLD content (read-first / read-before-write),
//     which is what makes the controller's read-modify-write pair work;
//   * RSTN is unused, exactly as upstream: the arrays have no reset, which is
//     why the exported program must fully initialise everything a run touches.
//
//------------------------------------------------------------------------------


module SRAM_256x128_wrapper (

    // Global inputs
    input          RSTN,                     // Reset_N (unused: BRAM has no reset)
    input          CK,                       // Clock (synchronous read/write)

    // Control and data inputs
    input          CS,                       // Chip select (active high)
    input          WE,                       // Write enable (active high)
    input  [  7:0] A,                        // Address bus
    input  [127:0] D,                        // Data input bus (write)

    // Data output
    output [127:0] Q                         // Data output bus (read)
);

    // Synthesizable, BRAM-inferable single-port synchronous memory. The
    // separate `if (CS)` / `if (CS & WE)` statements in one clocked block with
    // the read taken BEFORE the write is the canonical read-first template.
    (* ram_style = "block" *)
    reg [127:0] mem [0:255];
    reg [127:0] q_r;

    always @(posedge CK) begin
        if (CS)
            q_r <= mem[A];
        if (CS & WE)
            mem[A] <= D;
    end

    assign Q = q_r;

endmodule
