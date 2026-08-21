// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "tb_mem_overlay.v" - equivalence harness for the BRAM memory overlay.
//
// It instantiates `SRAM_256x128_wrapper` and `SRAM_8192x32_wrapper` by NAME and
// prints every read port value it observes. Whichever declaration the source
// list selects -- the vendored behavioural model inside neuron_core.v /
// synaptic_core.v, or the synthesizable overlay in hw/fpga/mem/ -- gets the
// SAME deterministic stimulus, so two runs whose transcripts are byte-identical
// are two memories with identical read latency and identical CS/WE semantics.
//
// The stimulus deliberately covers what the ODIN controller does to these
// memories: back-to-back read-then-write at one address (the read-modify-write
// pair the neuron sweep performs), CS held low across cycles (Q must HOLD, not
// clear), and writes to an address that is being read on the same edge (the
// read must see the OLD word).

`timescale 1ns/1ps

module tb_mem_overlay;

    localparam WARM_CYCLES = 512;
    localparam TEST_CYCLES = 4096;

    reg          CK;
    reg          RSTN;

    reg          n_cs, n_we;
    reg  [  7:0] n_a;
    reg  [127:0] n_d;
    wire [127:0] n_q;

    reg          s_cs, s_we;
    reg  [ 12:0] s_a;
    reg  [ 31:0] s_d;
    wire [ 31:0] s_q;

    reg  [31:0]  lfsr;
    integer      i;

    SRAM_256x128_wrapper u_neuron (
        .RSTN(RSTN), .CK(CK), .CS(n_cs), .WE(n_we),
        .A(n_a), .D(n_d), .Q(n_q));

    SRAM_8192x32_wrapper u_synapse (
        .RSTN(RSTN), .CK(CK), .CS(s_cs), .WE(s_we),
        .A(s_a), .D(s_d), .Q(s_q));

    initial CK = 1'b0;
    always #5 CK = ~CK;

    task step;
        begin
            @(posedge CK);
            #1;
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        end
    endtask

    initial begin
        RSTN = 1'b0;
        lfsr = 32'hACE1_2345;
        n_cs = 1'b0; n_we = 1'b0; n_a = 8'd0;  n_d = 128'd0;
        s_cs = 1'b0; s_we = 1'b0; s_a = 13'd0; s_d = 32'd0;
        repeat (4) @(posedge CK);
        RSTN = 1'b1;

        // Warm-up: fill the addresses the test phase touches, so a read never
        // returns an uninitialised word (these memories have NO reset, which is
        // exactly why the exported program initialises everything it touches).
        for (i = 0; i < WARM_CYCLES; i = i + 1) begin
            n_cs = 1'b1; n_we = 1'b1;
            n_a  = i[7:0];
            n_d  = {lfsr, ~lfsr, lfsr ^ 32'hA5A5_5A5A, lfsr + 32'd7};
            s_cs = 1'b1; s_we = 1'b1;
            s_a  = i[12:0];
            s_d  = lfsr ^ 32'h1234_5678;
            step;
        end

        for (i = 0; i < TEST_CYCLES; i = i + 1) begin
            // CS low on one cycle in eight: Q must HOLD its previous value.
            n_cs = (lfsr[3:1] != 3'b000);
            n_we = lfsr[0];
            n_a  = lfsr[15:8] & 8'h7F;
            n_d  = {lfsr, lfsr ^ 32'hFFFF_0000, ~lfsr, lfsr + 32'd1};
            s_cs = (lfsr[7:5] != 3'b000);
            s_we = lfsr[4];
            s_a  = {4'b0, lfsr[24:16]};
            s_d  = lfsr + 32'h0BAD_F00D;
            step;
            $display("MEM %0d %032x %08x", i, n_q, s_q);
        end

        $display("MEMDONE %0d", TEST_CYCLES);
        $finish;
    end

endmodule
