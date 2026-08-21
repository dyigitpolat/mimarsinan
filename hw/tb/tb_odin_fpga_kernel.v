// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "tb_odin_fpga_kernel.v" - the KERNEL-LEVEL smoke testbench: it preloads one
// token program into the fabric kernel's program RAM, pulses ap_start, waits
// for ap_done, and prints the capture RAM in the SAME line protocol
// `hw/tb/tb_odin_core.v` prints ("EV core neuron cycle tag", "DONE cycles
// events"). The host-side gate runs the same program through both testbenches
// and compares the counts: that comparison is what proves the on-fabric
// sequencer implements the same semantics as the host-side tb driver.
//
// Verilog-2005 only, so one source runs under both simulators the suite drives.

`timescale 1ns/1ps

module tb_odin_fpga_kernel;

    parameter NC        = 1;
    parameter PROGWORDS = 4096;
    parameter CAPWORDS  = 4096;
    parameter N         = 256;
    parameter M         = 8;

    localparam [31:0] OP_END = 32'd0;

    reg         CLK;
    reg         rst;
    reg         ap_start;
    wire        ap_done, ap_idle, ap_ready, err;

    reg         prog_we;
    reg  [31:0] prog_waddr, prog_wdata;
    reg  [31:0] cap_raddr;
    wire [31:0] cap_rdata;

    reg  [31:0] prog [0:PROGWORDS-1];
    reg  [1023:0] stim_path;
    integer     i, tokens, events, guard;
    reg  [31:0] header_events, header_cycles;
    reg  [31:0] rec_tag, rec_cycle, rec_core, rec_neuron;

    initial CLK = 1'b0;
    always #5 CLK = ~CLK;

    odin_fpga_kernel #(
        .NC(NC), .N(N), .M(M),
        .PROG_WORDS(PROGWORDS), .CAP_WORDS(CAPWORDS)
    ) dut (
        .clk (CLK), .rst (rst),
        .ap_start (ap_start), .ap_done (ap_done), .ap_idle (ap_idle),
        .ap_ready (ap_ready), .err (err),
        .prog_we (prog_we), .prog_waddr (prog_waddr), .prog_wdata (prog_wdata),
        .cap_raddr (cap_raddr), .cap_rdata (cap_rdata)
    );

    task read_word;
        input  [31:0] addr;
        output [31:0] value;
        begin
            cap_raddr = addr;
            @(posedge CLK);
            @(posedge CLK);
            value = cap_rdata;
        end
    endtask

    initial begin
        rst        = 1'b1;
        ap_start   = 1'b0;
        prog_we    = 1'b0;
        prog_waddr = 32'd0;
        prog_wdata = 32'd0;
        cap_raddr  = 32'd0;
        events     = 0;

        if (!$value$plusargs("stim=%s", stim_path)) begin
            $display("FATAL missing_plusarg +stim=<path>");
            $finish;
        end
        for (i = 0; i < PROGWORDS; i = i + 1) prog[i] = OP_END;
        $readmemh(stim_path, prog);

        repeat (8) @(posedge CLK);

        // The AXI loader's job, done directly: the program RAM is filled
        // before ap_start, exactly as the DMA fills it on the board.
        tokens = 0;
        for (i = 0; i < PROGWORDS; i = i + 1) begin
            @(posedge CLK);
            prog_we    = 1'b1;
            prog_waddr = i;
            prog_wdata = prog[i];
        end
        @(posedge CLK);
        prog_we = 1'b0;

        repeat (8) @(posedge CLK);
        rst = 1'b0;
        repeat (16) @(posedge CLK);

        ap_start = 1'b1;
        @(posedge CLK);
        while (ap_idle === 1'b1) @(posedge CLK);
        ap_start = 1'b0;

        guard = 0;
        while (ap_done !== 1'b1) begin
            @(posedge CLK);
            guard = guard + 1;
            if (guard > 200000000) begin
                $display("FATAL kernel_never_finished");
                $finish;
            end
        end
        repeat (4) @(posedge CLK);

        read_word(32'd0, header_events);
        read_word(32'd1, header_cycles);
        for (i = 0; i < header_events; i = i + 1) begin
            read_word(32'd2 + 4*i,     rec_tag);
            read_word(32'd2 + 4*i + 1, rec_cycle);
            read_word(32'd2 + 4*i + 2, rec_core);
            read_word(32'd2 + 4*i + 3, rec_neuron);
            $display("EV %0d %0d %0d %0d", rec_core, rec_neuron, rec_cycle, rec_tag);
            events = events + 1;
        end
        if (err === 1'b1) begin
            $display("FATAL kernel_refused_the_program");
            $finish;
        end
        $display("RBSTAT 0 0");
        $display("SHSTAT 0 0");
        $display("DONE %0d %0d", header_cycles, events);
        $finish;
    end

endmodule
