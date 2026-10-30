// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "tb_odin_fpga_kernel_axil.v" - the CONTROL-PORT conformance testbench: an
// AXI4-Lite master that uses the freedoms the protocol gives a master, against
// `odin_fpga_kernel_top`'s s_axi_control, and nothing else (the kernel is never
// started, so the AXI4 master side stays idle).
//
// WHY THIS EXISTS. On the card the control port sits behind the shell's AXI
// interconnect, which converts from the PCIe clock to the kernel clock with an
// independent clock-crossing FIFO PER CHANNEL. The write address and the write
// data of one register write therefore reach the kernel independently: W may
// arrive before AW, AW of the next write may arrive before W of this one, and B
// may be back-pressured. All of that is legal AXI4-Lite. A slave that pairs a W
// with whichever address it saw last writes a kernel argument into the wrong
// register, and the run then streams the wrong buffer or the wrong length.
//
// What the master does, per round of writes to every argument register (in the
// order a host writes them) plus GIE, each round with fresh pseudo-random data:
//   +mode=0  W leads: each W is presented before its AW, AW after a gap;
//   +mode=1  AW leads: the address process runs up to two writes ahead of data;
//   +mode=2  independent pseudo-random gaps on AW, W and B;
//   +mode=3  AW and W together (what the wrapper testbench's master does).
// B is back-pressured pseudo-randomly in every mode, and the read channel is
// exercised with back-to-back ARs under R back-pressure.
//
// Checks, printed as one AXIL_PASS / AXIL_FAIL line for the host gate:
//   * after every round, every argument register holds exactly the value last
//     written to ITS address (read through the hierarchy: the wrapper has no
//     read-back for them);
//   * exactly one B per write and one R per read, each with the right data.
//
// Verilog-2005 only.

`timescale 1ns/1ps

module tb_odin_fpga_kernel_axil;

    parameter NC        = 1;
    parameter PROGWORDS = 4096;   // unused; the harness passes it to every tb
    parameter CAPWORDS  = 4096;
    parameter FIFOWORDS = 1024;
    parameter N         = 256;
    parameter M         = 8;
    parameter ROUNDS    = 24;

    localparam NREG   = 10;
    localparam NWRITE = NREG * ROUNDS;

    reg CLK;
    reg rst_n;
    initial CLK = 1'b0;
    always #5 CLK = ~CLK;

    reg         c_awvalid, c_wvalid, c_arvalid, c_rready, c_bready;
    reg  [11:0] c_awaddr, c_araddr;
    reg  [31:0] c_wdata;
    reg  [3:0]  c_wstrb;
    wire        c_awready, c_wready, c_arready, c_rvalid, c_bvalid;
    wire [31:0] c_rdata;
    wire [1:0]  c_rresp, c_bresp;
    wire        irq;

    wire        m_arvalid, m_rready, m_awvalid, m_wvalid, m_wlast, m_bready;
    wire [63:0] m_araddr, m_awaddr;
    wire [7:0]  m_arlen, m_awlen;
    wire [2:0]  m_arsize, m_awsize;
    wire [1:0]  m_arburst, m_awburst;
    wire [31:0] m_wdata;
    wire [3:0]  m_wstrb;

    odin_fpga_kernel_top #(
        .NC(NC), .N(N), .M(M),
        .FIFO_WORDS(FIFOWORDS), .CAP_WORDS(CAPWORDS)
    ) dut (
        .ap_clk (CLK), .ap_rst_n (rst_n),
        .s_axi_control_awvalid (c_awvalid), .s_axi_control_awready (c_awready),
        .s_axi_control_awaddr  (c_awaddr),
        .s_axi_control_wvalid  (c_wvalid),  .s_axi_control_wready  (c_wready),
        .s_axi_control_wdata   (c_wdata),   .s_axi_control_wstrb   (c_wstrb),
        .s_axi_control_arvalid (c_arvalid), .s_axi_control_arready (c_arready),
        .s_axi_control_araddr  (c_araddr),
        .s_axi_control_rvalid  (c_rvalid),  .s_axi_control_rready  (c_rready),
        .s_axi_control_rdata   (c_rdata),   .s_axi_control_rresp   (c_rresp),
        .s_axi_control_bvalid  (c_bvalid),  .s_axi_control_bready  (c_bready),
        .s_axi_control_bresp   (c_bresp),   .interrupt             (irq),
        .m_axi_gmem_arvalid (m_arvalid), .m_axi_gmem_arready (1'b0),
        .m_axi_gmem_araddr  (m_araddr),  .m_axi_gmem_arlen   (m_arlen),
        .m_axi_gmem_arsize  (m_arsize),  .m_axi_gmem_arburst (m_arburst),
        .m_axi_gmem_rvalid  (1'b0),      .m_axi_gmem_rready  (m_rready),
        .m_axi_gmem_rdata   (32'd0),     .m_axi_gmem_rresp   (2'b00),
        .m_axi_gmem_rlast   (1'b0),
        .m_axi_gmem_awvalid (m_awvalid), .m_axi_gmem_awready (1'b0),
        .m_axi_gmem_awaddr  (m_awaddr),  .m_axi_gmem_awlen   (m_awlen),
        .m_axi_gmem_awsize  (m_awsize),  .m_axi_gmem_awburst (m_awburst),
        .m_axi_gmem_wvalid  (m_wvalid),  .m_axi_gmem_wready  (1'b0),
        .m_axi_gmem_wdata   (m_wdata),   .m_axi_gmem_wstrb   (m_wstrb),
        .m_axi_gmem_wlast   (m_wlast),
        .m_axi_gmem_bvalid  (1'b0),      .m_axi_gmem_bresp   (2'b00),
        .m_axi_gmem_bready  (m_bready)
    );

    // The register file a host writes, in the order it writes it (GIE last).
    function [11:0] reg_addr;
        input integer k;
        begin
            case (k)
                0: reg_addr = 12'h010; 1: reg_addr = 12'h014;
                2: reg_addr = 12'h01C; 3: reg_addr = 12'h020;
                4: reg_addr = 12'h028; 5: reg_addr = 12'h02C;
                6: reg_addr = 12'h034; 7: reg_addr = 12'h03C;
                8: reg_addr = 12'h044; default: reg_addr = 12'h004;
            endcase
        end
    endfunction

    // What the wrapper holds at each address, through the hierarchy.
    function [31:0] held;
        input integer k;
        begin
            case (k)
                0: held = dut.prog_addr_r[31:0];  1: held = dut.prog_addr_r[63:32];
                2: held = dut.stim_addr_r[31:0];  3: held = dut.stim_addr_r[63:32];
                4: held = dut.cap_addr_r[31:0];   5: held = dut.cap_addr_r[63:32];
                6: held = dut.prog_words_r;       7: held = dut.stim_words_r;
                8: held = dut.cap_events_r;       default: held = {31'd0, dut.gie_r};
            endcase
        end
    endfunction

    reg  [31:0] seed, mode;
    reg  [31:0] lfsr;
    reg  [31:0] wdata_of [0:NWRITE-1];
    integer     aw_sent, w_sent, b_seen, failures, i, k, round, guard;
    integer     r_seen, ar_sent;
    reg  [31:0] r_data [0:7];

    // One pseudo-random number per call (a 32-bit xorshift; seed never 0).
    function [31:0] next;
        input [31:0] x;
        reg   [31:0] y;
        begin
            y = x ^ (x << 13); y = y ^ (y >> 17); y = y ^ (y << 5);
            next = y;
        end
    endfunction

    // Gaps are drawn from per-process generators so the processes stay independent.
    reg [31:0] g_aw, g_w, g_b;

    // ---- B: back-pressured pseudo-randomly, every response counted ----------
    always @(posedge CLK) begin
        if (!rst_n) begin
            c_bready <= 1'b0; b_seen <= 0; g_b <= 32'h9E3779B9 ^ seed;
        end else begin
            g_b <= next(g_b);
            if (c_bvalid && c_bready) begin
                b_seen <= b_seen + 1;
                if (c_bresp != 2'b00) begin
                    $display("AXIL_FAIL bresp %0d on write %0d", c_bresp, b_seen);
                    failures = failures + 1;
                end
            end
            c_bready <= (mode == 3) ? 1'b1 : g_b[2];
        end
    end

    // ---- AW: one address per write, after a mode-dependent gap ---------------
    task aw_one;
        input integer n;
        integer gap, j;
        begin
            g_aw = next(g_aw);
            case (mode)
                0: gap = 2 + {30'd0, g_aw[1:0]};     // after its W (below)
                1: gap = 0;                              // as early as the slave allows
                2: gap = {29'd0, g_aw[2:0]};
                default: gap = 0;
            endcase
            // In "AW leads", never more than two writes ahead of the data.
            while ((mode == 1) && (n > w_sent + 2)) @(posedge CLK);
            // In "W leads", the address waits for its own data to be presented.
            while ((mode == 0) && (n >= w_sent) && !(c_wvalid && n == w_sent)) @(posedge CLK);
            for (j = 0; j < gap; j = j + 1) @(posedge CLK);
            #1;
            c_awaddr = reg_addr(n % NREG); c_awvalid = 1'b1;
            @(posedge CLK);
            while (!c_awready) @(posedge CLK);
            #1; c_awvalid = 1'b0;
            aw_sent = n + 1;
        end
    endtask

    task w_one;
        input integer n;
        integer gap, j;
        begin
            g_w = next(g_w);
            case (mode)
                0: gap = 0;
                1: gap = 1 + {30'd0, g_w[1:0]};
                2: gap = {29'd0, g_w[2:0]};
                default: gap = 0;
            endcase
            // AW and W together: wait for the address process to present this write.
            while ((mode == 3) && !(c_awvalid && n == aw_sent)) @(posedge CLK);
            for (j = 0; j < gap; j = j + 1) @(posedge CLK);
            #1;
            c_wdata = wdata_of[n]; c_wstrb = 4'hF; c_wvalid = 1'b1;
            @(posedge CLK);
            while (!c_wready) @(posedge CLK);
            #1; c_wvalid = 1'b0;
            w_sent = n + 1;
        end
    endtask

    // Both processes stop at the end of a round until the checker opens the
    // next one, so a round's registers are read while nothing is in flight.
    integer n_aw, n_w, limit;
    initial begin
        aw_sent = 0;
        wait (limit > 0);
        for (n_aw = 0; n_aw < NWRITE; n_aw = n_aw + 1) begin
            while (n_aw >= limit) @(posedge CLK);
            aw_one(n_aw);
        end
    end
    initial begin
        w_sent = 0;
        wait (limit > 0);
        for (n_w = 0; n_w < NWRITE; n_w = n_w + 1) begin
            while (n_w >= limit) @(posedge CLK);
            w_one(n_w);
        end
    end

    // ---- the reads: back-to-back ARs, R back-pressured ------------------------
    task read_burst;
        input integer count;
        integer j, got;
        begin
            got = 0;
            r_seen = 0; ar_sent = 0;
            fork
                begin : issue
                    for (j = 0; j < count; j = j + 1) begin
                        @(posedge CLK); #1;
                        c_araddr = (j % 2 == 0) ? 12'h054 : 12'h04C; c_arvalid = 1'b1;
                        @(posedge CLK);
                        while (!c_arready) @(posedge CLK);
                        #1; c_arvalid = 1'b0;
                        ar_sent = ar_sent + 1;
                    end
                end
                begin : take
                    guard = 0;
                    while (got < count && guard < 2000) begin
                        @(posedge CLK); #1;
                        c_rready = (guard % 3 == 2);
                        @(posedge CLK);
                        if (c_rvalid && c_rready) begin
                            r_data[got] = c_rdata; got = got + 1;
                        end
                        #1; c_rready = 1'b0;
                        guard = guard + 1;
                    end
                end
            join
            r_seen = got;
        end
    endtask

    initial begin
        rst_n = 1'b0;
        c_awvalid = 1'b0; c_wvalid = 1'b0; c_arvalid = 1'b0; c_rready = 1'b0;
        c_awaddr = 12'd0; c_araddr = 12'd0; c_wdata = 32'd0; c_wstrb = 4'h0;
        failures = 0; limit = 0;
        if (!$value$plusargs("seed=%d", seed)) seed = 32'd1;
        if (!$value$plusargs("mode=%d", mode)) mode = 32'd0;
        if (seed == 32'd0) seed = 32'd1;
        g_aw = 32'h85EBCA6B ^ seed; g_w = 32'hC2B2AE35 ^ seed;
        lfsr = seed;
        for (i = 0; i < NWRITE; i = i + 1) begin
            lfsr = next(lfsr);
            // GIE holds one bit; every other register holds the full word.
            wdata_of[i] = ((i % NREG) == NREG - 1) ? {31'd0, lfsr[7]} : lfsr;
        end
        repeat (8) @(posedge CLK);
        rst_n = 1'b1;
        repeat (8) @(posedge CLK);
        for (round = 0; round < ROUNDS; round = round + 1) begin
            limit = NREG * (round + 1);
            guard = 0;
            while (b_seen < NREG * (round + 1) && guard < 200000) begin
                @(posedge CLK); guard = guard + 1;
            end
            if (b_seen < NREG * (round + 1)) begin
                $display("AXIL_FAIL round %0d: %0d of %0d write responses arrived (mode %0d seed %0d)",
                         round, b_seen, NREG * (round + 1), mode, seed);
                $display("AXIL_DONE");
                $finish;
            end
            // Let this round's last write settle before reading the registers.
            repeat (2) @(posedge CLK);
            for (k = 0; k < NREG; k = k + 1) begin
                if (held(k) !== wdata_of[NREG * round + k]) begin
                    $display("AXIL_FAIL round %0d register 0x%03h holds 0x%08h, wrote 0x%08h (mode %0d seed %0d)",
                             round, reg_addr(k), held(k), wdata_of[NREG * round + k], mode, seed);
                    failures = failures + 1;
                end
            end
        end
        repeat (50) @(posedge CLK);
        if (b_seen != NWRITE) begin
            $display("AXIL_FAIL %0d write responses for %0d writes", b_seen, NWRITE);
            failures = failures + 1;
        end
        read_burst(6);
        if (r_seen != 6) begin
            $display("AXIL_FAIL %0d read responses for 6 reads", r_seen);
            failures = failures + 1;
        end
        for (i = 0; i < r_seen; i = i + 1) begin
            if ((i % 2 == 0) && (r_data[i] !== (CAPWORDS - 2) / 4)) begin
                $display("AXIL_FAIL read %0d of 0x054 returned 0x%08h", i, r_data[i]);
                failures = failures + 1;
            end
            if ((i % 2 == 1) && (r_data[i] !== 32'd0)) begin
                $display("AXIL_FAIL read %0d of 0x04C returned 0x%08h on an idle kernel", i, r_data[i]);
                failures = failures + 1;
            end
        end
        if (failures == 0)
            $display("AXIL_PASS writes %0d reads %0d mode %0d seed %0d", b_seen, r_seen, mode, seed);
        $display("AXIL_DONE");
        $finish;
    end

endmodule
