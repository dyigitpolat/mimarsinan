// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "tb_odin_core.v" - the mimarsinan cosimulation testbench for NC vendored ODIN
// cores. Original work: it INSTANTIATES the vendored core and derives no text
// from it. Verilog-2005 only, so one source runs under both simulators the
// project's suite drives (iverilog/vvp, and the faster binary+timing flow).
//
// It executes a token program produced by
// `mimarsinan.chip_simulation.odin_rtl.stimulus` -- the flat encoding of the
// exporter's sequencer schema v1:
//
//   CONFIG / GATE  -> OP_SPI_W  (config-register and memory byte writes)
//   CLEAR          -> OP_SPI_W  (masked neuron-state byte writes)
//   INJECT         -> OP_AER    (neuron-spike events, canonical drain order)
//   TREF           -> OP_AER    (all-neurons time reference, 0x007F)
//   BARRIER        -> OP_WAIT   (a plain cycle wait of the program's bound)
//   READOUT        -> the AER-out capture, tagged by OP_TAG
//
// plus two gate-only opcodes: OP_SPI_R (memory readback byte-compare) and
// OP_SHADOW (config registers have NO readback -- doc Sec.4 -- so the tb
// asserts them hierarchically instead; that is a SIMULATION-ONLY assertion and
// is never available to the runtime).
//
// CROSS-LANGUAGE CONTRACTS, transcribed from ChFrenkel/ODIN @ 1781931:
//   * one SPI transaction is 40 SCK edges, 20-bit address then 20-bit data,
//     MSB first; MOSI is sampled on posedge SCK and MISO changes on negedge
//     SCK (src/spi_slave.v:82-146). SCK must be at least 4x slower than CLK
//     (doc Sec.2.1), so this master runs SCK = CLK/4 exactly.
//   * a read byte appears on MISO during the LAST EIGHT bit slots: the shift
//     register is loaded at the negedge where spi_cnt==31 (spi_slave.v:119-120)
//     and MISO is its bit 19 (:146).
//   * the input AER link is a four-phase handshake with a double-latching
//     barrier on REQ (doc Sec.2.2); a neuron-spike event is
//     {1'b0, pre_neur[7:0], 8'h07} and an all-neurons time reference is
//     {1'b0, 8'h00, 8'h7F} (doc Sec.2.2.1).
//   * in standard mode the output AER address IS the spiking neuron's address
//     (doc Sec.2.2.2); one output event per fire, and the controller stalls on
//     AEROUT_CTRL_BUSY until the handshake completes (src/aer_out.v:143-159).

`timescale 1ns/1ps

module tb_odin_core;

    parameter NC        = 1;          // cores instantiated (the geometry key)
    parameter PROGWORDS = 1024;       // token-program length, set by the harness
    parameter N         = 256;
    parameter M         = 8;

    // Token opcodes -- the ONE encoding, mirrored in stimulus.py.
    localparam OP_END    = 32'd0;
    localparam OP_SPI_W  = 32'd1;
    localparam OP_SPI_R  = 32'd2;
    localparam OP_AER    = 32'd3;
    localparam OP_WAIT   = 32'd4;
    localparam OP_TAG    = 32'd5;
    localparam OP_SHADOW = 32'd6;

    localparam AER_TIMEOUT = 32'd2000000;

    reg                CLK;
    reg                rst;
    reg  [31:0]        cycle;
    reg  [31:0]        tag;

    // Per-core driver signals. TB-level arrays so the tasks can address a core
    // by a RUNTIME index; the generate block below connects element `c` to
    // core `c` with a constant index.
    reg                spi_sck    [0:NC-1];
    reg                spi_mosi   [0:NC-1];
    reg                aerin_req  [0:NC-1];
    reg  [2*M:0]       aerin_addr [0:NC-1];

    wire [NC-1:0]      miso_w;
    wire [NC-1:0]      aerin_ack_w;
    wire [NC-1:0]      aerout_req_w;
    wire [M*NC-1:0]    aerout_addr_w;

    // Config-register shadow bus: one 20-bit slice per core, muxed by the
    // register id the OP_SHADOW token names.
    reg  [31:0]        shadow_reg;
    wire [20*NC-1:0]   shadow_bus;

    reg  [19:0]        spi_rdata;

    reg  [31:0]        prog [0:PROGWORDS-1];
    integer            pc;

    integer            reads_done, read_fails;
    integer            shadow_done, shadow_fails;
    integer            events_seen;

    genvar c;

    //----------------------------------------------------------------------
    //  Clock and free-running cycle counter
    //----------------------------------------------------------------------

    initial CLK = 1'b0;
    always #5 CLK = ~CLK;

    always @(posedge CLK)
        if (rst) cycle <= 32'd0;
        else     cycle <= cycle + 32'd1;

    //----------------------------------------------------------------------
    //  The cores, their AER-out capture, and their shadow taps
    //----------------------------------------------------------------------

    generate
        for (c = 0; c < NC; c = c + 1) begin : cg

            reg ack_r;
            reg req_d;

            ODIN #(.N(N), .M(M)) dut (
                .CLK         (CLK),
                .RST         (rst),
                .SCK         (spi_sck[c]),
                .MOSI        (spi_mosi[c]),
                .MISO        (miso_w[c]),
                .AERIN_ADDR  (aerin_addr[c]),
                .AERIN_REQ   (aerin_req[c]),
                .AERIN_ACK   (aerin_ack_w[c]),
                .AEROUT_ADDR (aerout_addr_w[M*c+M-1:M*c]),
                .AEROUT_REQ  (aerout_req_w[c]),
                .AEROUT_ACK  (ack_r)
            );

            // Four-phase AER-out consumer: ACK follows REQ, and every rising
            // REQ is one captured output spike of the neuron it names.
            always @(posedge CLK) begin
                if (rst) begin
                    ack_r <= 1'b0;
                    req_d <= 1'b0;
                end else begin
                    ack_r <= aerout_req_w[c];
                    req_d <= aerout_req_w[c];
                    if (aerout_req_w[c] && !req_d) begin
                        $display("EV %0d %0d %0d %0d",
                                 c, aerout_addr_w[M*c+M-1:M*c], cycle, tag);
                        events_seen = events_seen + 1;
                    end
                end
            end

            // Config registers have no readback path: this hierarchical tap is
            // the SIMULATION-ONLY substitute the plan calls for (Sec.5.4).
            assign shadow_bus[20*c+19:20*c] =
                  (shadow_reg == 32'd0)  ? {19'b0, dut.spi_slave_0.SPI_GATE_ACTIVITY}
                : (shadow_reg == 32'd1)  ? {19'b0, dut.spi_slave_0.SPI_OPEN_LOOP}
                : (shadow_reg == 32'd2)  ?  dut.spi_slave_0.SPI_BURST_TIMEREF
                : (shadow_reg == 32'd3)  ? {19'b0, dut.spi_slave_0.SPI_AER_SRC_CTRL_nNEUR}
                : (shadow_reg == 32'd4)  ? {19'b0, dut.spi_slave_0.SPI_OUT_AER_MONITOR_EN}
                : (shadow_reg == 32'd5)  ? {12'b0, dut.spi_slave_0.SPI_MONITOR_NEUR_ADDR}
                : (shadow_reg == 32'd6)  ? {12'b0, dut.spi_slave_0.SPI_MONITOR_SYN_ADDR}
                : (shadow_reg == 32'd7)  ? {19'b0, dut.spi_slave_0.SPI_UPDATE_UNMAPPED_SYN}
                : (shadow_reg == 32'd8)  ? {19'b0, dut.spi_slave_0.SPI_PROPAGATE_UNMAPPED_SYN}
                : (shadow_reg == 32'd9)  ? {19'b0, dut.spi_slave_0.SPI_SDSP_ON_SYN_STIM}
                : {4'b0, dut.spi_slave_0.SPI_SYN_SIGN[
                        (shadow_reg - 32'd10) * 16 +: 16]};
        end
    endgenerate

    //----------------------------------------------------------------------
    //  SPI master: 20-bit address + 20-bit data, SCK = CLK/4, MSB first
    //----------------------------------------------------------------------

    task spi_xfer;
        input integer core;
        input [19:0]  addr;
        input [19:0]  data;
        integer       bit_index;
        reg   [39:0]  frame;
        begin
            frame     = {addr, data};
            spi_rdata = 20'd0;
            for (bit_index = 0; bit_index < 40; bit_index = bit_index + 1) begin
                spi_mosi[core] = frame[39 - bit_index];
                @(posedge CLK);
                @(posedge CLK);
                spi_sck[core] = 1'b1;
                @(posedge CLK);
                spi_rdata = {spi_rdata[18:0], miso_w[core]};
                @(posedge CLK);
                spi_sck[core] = 1'b0;
            end
            repeat (4) @(posedge CLK);
        end
    endtask

    //----------------------------------------------------------------------
    //  AER-in driver: the four-phase handshake, one event at a time so the
    //  canonical drain order the program emits is the order the core sees.
    //----------------------------------------------------------------------

    task aer_send;
        input integer core;
        input [2*M:0] addr;
        integer       guard;
        begin
            @(posedge CLK);
            aerin_addr[core] = addr;
            aerin_req[core]  = 1'b1;
            guard = 0;
            while (aerin_ack_w[core] !== 1'b1) begin
                @(posedge CLK);
                guard = guard + 1;
                if (guard > AER_TIMEOUT) begin
                    $display("FATAL aer_ack_timeout core=%0d addr=%0d", core, addr);
                    $finish;
                end
            end
            aerin_req[core] = 1'b0;
            guard = 0;
            while (aerin_ack_w[core] !== 1'b0) begin
                @(posedge CLK);
                guard = guard + 1;
                if (guard > AER_TIMEOUT) begin
                    $display("FATAL aer_req_down_timeout core=%0d", core);
                    $finish;
                end
            end
            @(posedge CLK);
        end
    endtask

    //----------------------------------------------------------------------
    //  The program executor
    //----------------------------------------------------------------------

    integer            i;
    reg   [31:0]       op;
    integer            arg_core;
    reg   [31:0]       arg_a;
    reg   [31:0]       arg_b;
    reg   [19:0]       shadow_got;
    reg                running;
    reg   [1023:0]     stim_path;

    initial begin
        for (i = 0; i < NC; i = i + 1) begin
            spi_sck[i]    = 1'b0;
            spi_mosi[i]   = 1'b0;
            aerin_req[i]  = 1'b0;
            aerin_addr[i] = 17'd0;
        end
        rst          = 1'b1;
        cycle        = 32'd0;
        tag          = 32'd0;
        shadow_reg   = 32'd0;
        spi_rdata    = 20'd0;
        reads_done   = 0;
        read_fails   = 0;
        shadow_done  = 0;
        shadow_fails = 0;
        events_seen  = 0;
        pc           = 0;
        running      = 1'b1;

        if (!$value$plusargs("stim=%s", stim_path)) begin
            $display("FATAL missing_plusarg +stim=<path>");
            $finish;
        end
        $readmemh(stim_path, prog);

        repeat (12) @(posedge CLK);
        rst = 1'b0;
        repeat (4) @(posedge CLK);

        while (running) begin
            op = prog[pc];
            pc = pc + 1;
            if (op === OP_END || op === 32'bx) begin
                running = 1'b0;
            end else if (op === OP_SPI_W) begin
                arg_core = prog[pc];     arg_a = prog[pc+1];
                arg_b    = prog[pc+2];   pc    = pc + 3;
                spi_xfer(arg_core, arg_a[19:0], arg_b[19:0]);
            end else if (op === OP_SPI_R) begin
                arg_core = prog[pc];     arg_a = prog[pc+1];
                arg_b    = prog[pc+2];   pc    = pc + 3;
                spi_xfer(arg_core, arg_a[19:0], 20'd0);
                reads_done = reads_done + 1;
                if (spi_rdata[7:0] !== arg_b[7:0]) begin
                    read_fails = read_fails + 1;
                    if (read_fails <= 20)
                        $display("RBFAIL %0d %0d %0d %0d",
                                 arg_core, arg_a, spi_rdata[7:0], arg_b[7:0]);
                end
            end else if (op === OP_AER) begin
                arg_core = prog[pc];     arg_a = prog[pc+1];
                pc       = pc + 2;
                aer_send(arg_core, arg_a[2*M:0]);
            end else if (op === OP_WAIT) begin
                arg_a = prog[pc];        pc = pc + 1;
                // The BARRIER's own start cycle: the drain bound is stated
                // against this instant, so the gate can compare it to the
                // cycle of the last output event of the same window.
                $display("BARRIER %0d %0d %0d", tag, cycle, arg_a);
                repeat (arg_a) @(posedge CLK);
            end else if (op === OP_TAG) begin
                arg_a = prog[pc];        pc = pc + 1;
                tag   = arg_a;
                $display("TAGAT %0d %0d", tag, cycle);
            end else if (op === OP_SHADOW) begin
                arg_core   = prog[pc];   arg_a = prog[pc+1];
                arg_b      = prog[pc+2]; pc    = pc + 3;
                shadow_reg = arg_a;
                #1;
                shadow_got  = shadow_bus[20*arg_core +: 20];
                shadow_done = shadow_done + 1;
                if (shadow_got !== arg_b[19:0]) begin
                    shadow_fails = shadow_fails + 1;
                    $display("SHFAIL %0d %0d %0d %0d",
                             arg_core, arg_a, shadow_got, arg_b[19:0]);
                end
            end else begin
                $display("FATAL unknown_opcode %0d at %0d", op, pc - 1);
                $finish;
            end
        end

        repeat (8) @(posedge CLK);
        $display("RBSTAT %0d %0d", reads_done, read_fails);
        $display("SHSTAT %0d %0d", shadow_done, shadow_fails);
        $display("DONE %0d %0d", cycle, events_seen);
        $finish;
    end

endmodule
