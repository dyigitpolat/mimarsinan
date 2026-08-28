// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "tb_odin_fpga_kernel.v" - the KERNEL-LEVEL smoke testbench: it STREAMS one
// token program into the fabric kernel's elastic FIFO the way the wrapper's
// read engine does -- one word per cycle while the FIFO has room, starting
// after ap_start, with the sequencer already running -- waits for ap_done, and
// prints the capture RAM in the SAME line protocol `hw/tb/tb_odin_core.v`
// prints ("EV core neuron cycle tag", "DONE cycles events"). The host-side gate
// runs the same program through both testbenches and compares the counts: that
// comparison is what proves the on-fabric sequencer implements the same
// semantics as the host-side tb driver.
//
// Verilog-2005 only, so one source runs under both simulators the suite drives.

`timescale 1ns/1ps

module tb_odin_fpga_kernel;

    parameter NC        = 1;
    parameter PROGWORDS = 4096;
    parameter CAPWORDS  = 4096;
    parameter FIFOWORDS = 1024;
    parameter N         = 256;
    parameter M         = 8;

    localparam [31:0] OP_END = 32'd0;

    reg         CLK;
    reg         rst;
    reg         ap_start;
    wire        ap_done, ap_idle, ap_ready, err;

    reg         str_valid;
    reg  [31:0] str_data;
    wire [31:0] str_space;
    reg  [31:0] cap_raddr;
    wire [31:0] cap_rdata;

    reg         streaming;
    integer     push_idx;

    reg  [31:0] prog [0:PROGWORDS-1];
    reg  [1023:0] stim_path;
    integer     i, events, guard;
    reg  [31:0] header_events, header_cycles;
    reg  [31:0] rec_tag, rec_cycle, rec_core, rec_neuron;

    initial CLK = 1'b0;
    always #5 CLK = ~CLK;

    odin_fpga_kernel #(
        .NC(NC), .N(N), .M(M),
        .FIFO_WORDS(FIFOWORDS), .CAP_WORDS(CAPWORDS)
    ) dut (
        .clk (CLK), .rst (rst),
        .ap_start (ap_start), .ap_done (ap_done), .ap_idle (ap_idle),
        .ap_ready (ap_ready), .err (err),
        .str_valid (str_valid), .str_data (str_data), .str_space (str_space),
        .cap_raddr (cap_raddr), .cap_rdata (cap_rdata)
    );

    // The read engine's job, done directly, under the SAME credit discipline:
    // `str_space` is the FIFO's free count as of this edge, so it does not yet
    // know about the push already in flight on `str_valid`. Reserving that one
    // too is what the wrapper does a whole burst at a time.
    wire [31:0] credit = str_space - {31'd0, str_valid};

    always @(posedge CLK) begin
        if (rst) begin
            str_valid <= 1'b0;
            push_idx  <= 0;
        end else if (streaming && (push_idx < PROGWORDS) && (credit != 32'd0)) begin
            str_valid <= 1'b1;
            str_data  <= prog[push_idx];
            push_idx  <= push_idx + 1;
        end else begin
            str_valid <= 1'b0;
        end
    end

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
        streaming  = 1'b0;
        str_data   = 32'd0;
        cap_raddr  = 32'd0;
        events     = 0;

        if (!$value$plusargs("stim=%s", stim_path)) begin
            $display("FATAL missing_plusarg +stim=<path>");
            $finish;
        end
        for (i = 0; i < PROGWORDS; i = i + 1) prog[i] = OP_END;
        $readmemh(stim_path, prog);

        repeat (8) @(posedge CLK);
        rst = 1'b0;
        repeat (16) @(posedge CLK);

        // The FIFO is flushed at ap_start, so streaming is armed only once the
        // sequencer has actually accepted the run.
        ap_start = 1'b1;
        @(posedge CLK);
        while (ap_idle === 1'b1) @(posedge CLK);
        ap_start  = 1'b0;
        streaming = 1'b1;

        guard = 0;
        while (ap_done !== 1'b1) begin
            @(posedge CLK);
            guard = guard + 1;
            if (guard > 200000000) begin
                $display("FATAL kernel_never_finished");
                $finish;
            end
        end
        streaming = 1'b0;
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
