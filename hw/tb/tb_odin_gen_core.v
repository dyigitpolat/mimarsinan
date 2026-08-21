// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "tb_odin_gen_core.v" - the mimarsinan cosimulation testbench for NC GENERATED
// variant cores (`hw/gen/odin_gen_core.v.tmpl` expanded from a `CoreSpec`).
// Original work: it INSTANTIATES the generated core and derives no text from
// it. Verilog-2005 only, so one source runs under both simulators the project
// drives.
//
// It is the same instrument as `tb_odin_core.v`, with one difference that
// follows from the generator's own statement of changes: the variant core is
// programmed through a DIRECT SYNCHRONOUS write port instead of the 20-bit SPI
// slave, so CONFIG/CLEAR stages become OP_PROG tokens rather than OP_SPI_W
// transactions. The event program (OP_AER / OP_WAIT / OP_TAG) and the capture
// line protocol are unchanged, which is what lets the P5 harness read a variant
// run exactly as it reads a stock one.
//
//   OP_PROG (core, sel, addr, data)  -> one clock on the configuration port
//                                       sel 0 = config register, 1 = threshold,
//                                       2 = membrane (the per-sample CLEAR),
//                                       3 = synapse word
//   OP_AER  (core, addr)             -> one four-phase input event
//   OP_WAIT (cycles)                 -> the BARRIER drain bound
//   OP_TAG  (tag)                    -> opens the window the capture is keyed on
//
// Two gates ride the same transcript. SPECFAIL: the generated core reports its
// own geometry and law on constant outputs, and the harness's parameters must
// equal them -- a build whose ports were sized from a different spec than the
// file declares fails loud instead of silently truncating an address.
// RAILFAIL: the sticky rail flag of a core whose declared law says the rails
// are unreachable.

`timescale 1ns/1ps

module tb_odin_gen_core;

    parameter NC        = 1;          // cores instantiated
    parameter PROGWORDS = 1024;       // token-program length, set by the harness
    parameter AW        = 8;          // AER-in address width - 1 (clog2 axons)
    parameter NW        = 8;          // AER-out address width (clog2 neurons)
    parameter AXONS     = 256;
    parameter NEURONS   = 256;
    parameter MBITS     = 8;
    parameter WBITS     = 4;
    parameter FLAGS     = 0;          // {ASSERT_NO_SAT,CMP_INCL,RESET_ZERO,PER_EVENT,MSIGNED}

    localparam OP_END    = 32'd0;
    localparam OP_AER    = 32'd3;
    localparam OP_WAIT   = 32'd4;
    localparam OP_TAG    = 32'd5;
    localparam OP_PROG   = 32'd7;

    localparam AER_TIMEOUT = 32'd20000000;

    reg                CLK;
    reg                rst;
    reg  [31:0]        cycle;
    reg  [31:0]        tag;

    reg                prog_en    [0:NC-1];
    reg  [1:0]         prog_sel   [0:NC-1];
    reg  [31:0]        prog_addr  [0:NC-1];
    reg  [31:0]        prog_data  [0:NC-1];
    reg                aerin_req  [0:NC-1];
    reg  [AW:0]        aerin_addr [0:NC-1];

    wire [NC-1:0]      aerin_ack_w;
    wire [NC-1:0]      aerout_req_w;
    wire [NW*NC-1:0]   aerout_addr_w;
    wire [NC-1:0]      rail_w;
    wire [32*NC-1:0]   spec_axons_w, spec_neurons_w;
    wire [32*NC-1:0]   spec_mbits_w, spec_wbits_w, spec_flags_w;

    reg  [31:0]        prog [0:PROGWORDS-1];
    integer            pc;
    integer            events_seen;
    integer            spec_fails, rail_fails;

    genvar c;

    initial CLK = 1'b0;
    always #5 CLK = ~CLK;

    always @(posedge CLK)
        if (rst) cycle <= 32'd0;
        else     cycle <= cycle + 32'd1;

    generate
        for (c = 0; c < NC; c = c + 1) begin : cg

            reg ack_r;
            reg req_d;

            odin_gen_core dut (
                .CLK          (CLK),
                .RST          (rst),
                .PROG_EN      (prog_en[c]),
                .PROG_SEL     (prog_sel[c]),
                .PROG_ADDR    (prog_addr[c]),
                .PROG_DATA    (prog_data[c]),
                .AERIN_ADDR   (aerin_addr[c]),
                .AERIN_REQ    (aerin_req[c]),
                .AERIN_ACK    (aerin_ack_w[c]),
                .AEROUT_ADDR  (aerout_addr_w[NW*c+NW-1:NW*c]),
                .AEROUT_REQ   (aerout_req_w[c]),
                .AEROUT_ACK   (ack_r),
                .SPEC_AXONS   (spec_axons_w[32*c+31:32*c]),
                .SPEC_NEURONS (spec_neurons_w[32*c+31:32*c]),
                .SPEC_MBITS   (spec_mbits_w[32*c+31:32*c]),
                .SPEC_WBITS   (spec_wbits_w[32*c+31:32*c]),
                .SPEC_FLAGS   (spec_flags_w[32*c+31:32*c]),
                .RAIL_TOUCHED (rail_w[c])
            );

            always @(posedge CLK) begin
                if (rst) begin
                    ack_r <= 1'b0;
                    req_d <= 1'b0;
                end else begin
                    ack_r <= aerout_req_w[c];
                    req_d <= aerout_req_w[c];
                    if (aerout_req_w[c] && !req_d) begin
                        $display("EV %0d %0d %0d %0d",
                                 c, aerout_addr_w[NW*c+NW-1:NW*c], cycle, tag);
                        events_seen = events_seen + 1;
                    end
                end
            end
        end
    endgenerate

    task prog_write;
        input integer core;
        input [31:0]  sel;
        input [31:0]  addr;
        input [31:0]  data;
        begin
            @(posedge CLK);
            prog_sel[core]  = sel[1:0];
            prog_addr[core] = addr;
            prog_data[core] = data;
            prog_en[core]   = 1'b1;
            @(posedge CLK);
            prog_en[core]   = 1'b0;
        end
    endtask

    task aer_send;
        input integer core;
        input [AW:0]  addr;
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

    task check_spec;
        input integer core;
        begin
            if (spec_axons_w[32*core +: 32] !== AXONS
             || spec_neurons_w[32*core +: 32] !== NEURONS
             || spec_mbits_w[32*core +: 32] !== MBITS
             || spec_wbits_w[32*core +: 32] !== WBITS
             || spec_flags_w[32*core +: 32] !== FLAGS) begin
                spec_fails = spec_fails + 1;
                $display("SPECFAIL %0d %0d %0d %0d %0d %0d",
                         core,
                         spec_axons_w[32*core +: 32],
                         spec_neurons_w[32*core +: 32],
                         spec_mbits_w[32*core +: 32],
                         spec_wbits_w[32*core +: 32],
                         spec_flags_w[32*core +: 32]);
            end
        end
    endtask

    integer            i;
    reg   [31:0]       op;
    integer            arg_core;
    reg   [31:0]       arg_a;
    reg   [31:0]       arg_b;
    reg   [31:0]       arg_c;
    reg                running;
    reg   [1023:0]     stim_path;

    initial begin
        for (i = 0; i < NC; i = i + 1) begin
            prog_en[i]    = 1'b0;
            prog_sel[i]   = 2'd0;
            prog_addr[i]  = 32'd0;
            prog_data[i]  = 32'd0;
            aerin_req[i]  = 1'b0;
            aerin_addr[i] = {(AW+1){1'b0}};
        end
        rst         = 1'b1;
        cycle       = 32'd0;
        tag         = 32'd0;
        events_seen = 0;
        spec_fails  = 0;
        rail_fails  = 0;
        pc          = 0;
        running     = 1'b1;

        if (!$value$plusargs("stim=%s", stim_path)) begin
            $display("FATAL missing_plusarg +stim=<path>");
            $finish;
        end
        $readmemh(stim_path, prog);

        repeat (8) @(posedge CLK);
        rst = 1'b0;
        repeat (4) @(posedge CLK);

        for (i = 0; i < NC; i = i + 1) check_spec(i);
        if (spec_fails != 0) begin
            $display("FATAL spec_mismatch %0d", spec_fails);
            $finish;
        end

        while (running) begin
            op = prog[pc];
            pc = pc + 1;
            if (op === OP_END || op === 32'bx) begin
                running = 1'b0;
            end else if (op === OP_PROG) begin
                arg_core = prog[pc];     arg_a = prog[pc+1];
                arg_b    = prog[pc+2];   arg_c = prog[pc+3];
                pc       = pc + 4;
                prog_write(arg_core, arg_a, arg_b, arg_c);
            end else if (op === OP_AER) begin
                arg_core = prog[pc];     arg_a = prog[pc+1];
                pc       = pc + 2;
                aer_send(arg_core, arg_a[AW:0]);
            end else if (op === OP_WAIT) begin
                arg_a = prog[pc];        pc = pc + 1;
                $display("BARRIER %0d %0d %0d", tag, cycle, arg_a);
                repeat (arg_a) @(posedge CLK);
            end else if (op === OP_TAG) begin
                arg_a = prog[pc];        pc = pc + 1;
                tag   = arg_a;
                $display("TAGAT %0d %0d", tag, cycle);
            end else begin
                $display("FATAL unknown_opcode %0d at %0d", op, pc - 1);
                $finish;
            end
        end

        repeat (8) @(posedge CLK);
        for (i = 0; i < NC; i = i + 1)
            if (rail_w[i] !== 1'b0) begin
                rail_fails = rail_fails + 1;
                $display("RAILFAIL %0d", i);
            end
        $display("SPECSTAT %0d %0d", NC, spec_fails);
        $display("RAILSTAT %0d %0d", NC, rail_fails);
        $display("DONE %0d %0d", cycle, events_seen);
        $finish;
    end

endmodule
