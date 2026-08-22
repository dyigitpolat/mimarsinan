// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "tb_odin_fpga_kernel_axi.v" - the WRAPPER-LEVEL smoke testbench: it drives
// `odin_fpga_kernel_top` exactly as the XDMA shell does, through its two AXI
// interfaces and nothing else. Nothing is preloaded into the fabric.
//
//   * a behavioural AXI4 SLAVE (`mem`) stands in for the board's HBM: it serves
//     INCR read bursts and accepts INCR write bursts with valid/ready stalls, so
//     a master that mishandled a handshake would hang or lose a beat here;
//   * the token program is placed in that memory SPLIT IN TWO -- the first
//     `SPLIT` tokens plus an END terminator as the PROGRAM buffer, the rest plus
//     an END as the STIMULUS buffer -- which is the split the host makes;
//   * the AXI4-Lite master writes the six kernel arguments, reads the two
//     read-only capacity registers, pulses ap_start, polls ap_done, and reads
//     the status register;
//   * the capture is read back OUT OF THE AXI MEMORY, never out of the fabric.
//
// It prints the same line protocol as `hw/tb/tb_odin_core.v` ("EV core neuron
// cycle tag", "DONE cycles events") plus one KSTAT line carrying the register
// reads, so the host gate compares the counts and reads the refusals.
//
// Verilog-2005 only, so one source runs under both simulators the suite drives.

`timescale 1ns/1ps

module tb_odin_fpga_kernel_axi;

    parameter NC        = 1;
    parameter PROGWORDS = 4096;
    parameter CAPWORDS  = 4096;
    parameter N         = 256;
    parameter M         = 8;
    // Where the token stream is cut into the program and stimulus buffers.
    parameter SPLIT     = 1024;
    // The capture capacity the HOST declares; the fabric clamps it to its own.
    parameter HOSTCAP   = 32'h000FFFFF;

    localparam [31:0] OP_END = 32'd0;

    localparam ADDR_AP_CTRL  = 12'h000;
    localparam ADDR_PROG_LO  = 12'h010;
    localparam ADDR_PROG_HI  = 12'h014;
    localparam ADDR_STIM_LO  = 12'h01C;
    localparam ADDR_STIM_HI  = 12'h020;
    localparam ADDR_CAP_LO   = 12'h028;
    localparam ADDR_CAP_HI   = 12'h02C;
    localparam ADDR_PROG_N   = 12'h034;
    localparam ADDR_STIM_N   = 12'h03C;
    localparam ADDR_CAP_N    = 12'h044;
    localparam ADDR_STATUS   = 12'h04C;
    localparam ADDR_CAP_CAP  = 12'h054;
    localparam ADDR_PROG_CAP = 12'h05C;

    localparam [31:0] AP_DONE_BIT = 32'd2;

    // PROGWORDS is a power of two >= 4096, so every base below is 4 KiB
    // aligned -- which is what lets the master's bursts never cross a page.
    localparam [63:0] PROG_BASE = 64'h0000_1000;
    localparam [63:0] STIM_BASE = PROG_BASE + PROGWORDS * 4;
    localparam [63:0] CAP_BASE  = STIM_BASE + PROGWORDS * 4;
    localparam        MEMWORDS  = (CAP_BASE / 4) + CAPWORDS + 16;

    reg CLK;
    reg rst_n;

    initial CLK = 1'b0;
    always #5 CLK = ~CLK;

    //----------------------------------------------------------------------
    //  Wires
    //----------------------------------------------------------------------

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

    reg         s_arready, s_rvalid, s_rlast, s_awready, s_wready, s_bvalid;
    reg  [31:0] s_rdata;

    odin_fpga_kernel_top #(
        .NC(NC), .N(N), .M(M),
        .PROG_WORDS(PROGWORDS), .CAP_WORDS(CAPWORDS)
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
        .m_axi_gmem_arvalid (m_arvalid), .m_axi_gmem_arready (s_arready),
        .m_axi_gmem_araddr  (m_araddr),  .m_axi_gmem_arlen   (m_arlen),
        .m_axi_gmem_arsize  (m_arsize),  .m_axi_gmem_arburst (m_arburst),
        .m_axi_gmem_rvalid  (s_rvalid),  .m_axi_gmem_rready  (m_rready),
        .m_axi_gmem_rdata   (s_rdata),   .m_axi_gmem_rresp   (2'b00),
        .m_axi_gmem_rlast   (s_rlast),
        .m_axi_gmem_awvalid (m_awvalid), .m_axi_gmem_awready (s_awready),
        .m_axi_gmem_awaddr  (m_awaddr),  .m_axi_gmem_awlen   (m_awlen),
        .m_axi_gmem_awsize  (m_awsize),  .m_axi_gmem_awburst (m_awburst),
        .m_axi_gmem_wvalid  (m_wvalid),  .m_axi_gmem_wready  (s_wready),
        .m_axi_gmem_wdata   (m_wdata),   .m_axi_gmem_wstrb   (m_wstrb),
        .m_axi_gmem_wlast   (m_wlast),
        .m_axi_gmem_bvalid  (s_bvalid),  .m_axi_gmem_bresp   (2'b00),
        .m_axi_gmem_bready  (m_bready)
    );

    //----------------------------------------------------------------------
    //  The AXI4 memory model: one array, INCR bursts, stalled handshakes
    //----------------------------------------------------------------------

    reg [31:0] mem [0:MEMWORDS-1];

    reg [63:0] ar_addr_q, aw_addr_q;
    reg [8:0]  ar_left_q, aw_left_q;
    reg        ar_busy, aw_busy;
    reg [15:0] lfsr;

    function [31:0] widx;
        input [63:0] byte_addr;
        widx = byte_addr[33:2];
    endfunction

    // Every address the master emits must land inside the model, or the beats
    // would read/write X and the counts would be a coincidence.
    always @(posedge CLK) begin
        if (rst_n && s_arready && m_arvalid && (widx(m_araddr) >= MEMWORDS)) begin
            $display("FATAL axi_read_outside_model %0d", widx(m_araddr));
            $finish;
        end
        if (rst_n && s_awready && m_awvalid && (widx(m_awaddr) >= MEMWORDS)) begin
            $display("FATAL axi_write_outside_model %0d", widx(m_awaddr));
            $finish;
        end
    end

    integer cycles_elapsed;
    always @(posedge CLK) begin
        if (!rst_n) begin
            cycles_elapsed <= 0;
        end else begin
            cycles_elapsed <= cycles_elapsed + 1;
            if (cycles_elapsed > 200000000) begin
                $display("FATAL simulation_watchdog_expired");
                $finish;
            end
        end
    end

    always @(posedge CLK) begin
        if (!rst_n) begin
            s_arready <= 1'b0; s_rvalid <= 1'b0; s_rlast <= 1'b0;
            s_awready <= 1'b0; s_wready <= 1'b0; s_bvalid <= 1'b0;
            ar_busy   <= 1'b0; aw_busy  <= 1'b0;
            ar_left_q <= 9'd0; aw_left_q <= 9'd0;
            lfsr      <= 16'hACE1;
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

            // ---- read address ------------------------------------------
            if (!ar_busy && !s_arready) begin
                s_arready <= lfsr[0];
            end else if (s_arready && m_arvalid) begin
                if (m_arburst != 2'b01 || m_arsize != 3'd2) begin
                    $display("FATAL axi_read_burst_not_incr32 %0d %0d",
                             m_arburst, m_arsize);
                    $finish;
                end
                ar_addr_q <= m_araddr;
                ar_left_q <= {1'b0, m_arlen} + 9'd1;
                ar_busy   <= 1'b1;
                s_arready <= 1'b0;
                s_rdata   <= mem[widx(m_araddr)];
                s_rlast   <= (m_arlen == 8'd0);
                s_rvalid  <= 1'b1;
            end else if (s_arready && !m_arvalid) begin
                s_arready <= lfsr[1];
            end

            // ---- read data ---------------------------------------------
            if (ar_busy) begin
                if (s_rvalid && m_rready) begin
                    if (ar_left_q == 9'd1) begin
                        s_rvalid <= 1'b0;
                        s_rlast  <= 1'b0;
                        ar_busy  <= 1'b0;
                    end else begin
                        ar_addr_q <= ar_addr_q + 64'd4;
                        ar_left_q <= ar_left_q - 9'd1;
                        s_rdata   <= mem[widx(ar_addr_q + 64'd4)];
                        s_rlast   <= (ar_left_q == 9'd2);
                        s_rvalid  <= lfsr[2];
                    end
                end else if (!s_rvalid) begin
                    s_rdata  <= mem[widx(ar_addr_q)];
                    s_rlast  <= (ar_left_q == 9'd1);
                    s_rvalid <= 1'b1;
                end
            end

            // ---- write address -----------------------------------------
            if (!aw_busy && !s_awready) begin
                s_awready <= lfsr[3];
            end else if (s_awready && m_awvalid) begin
                if (m_awburst != 2'b01 || m_awsize != 3'd2) begin
                    $display("FATAL axi_write_burst_not_incr32 %0d %0d",
                             m_awburst, m_awsize);
                    $finish;
                end
                aw_addr_q <= m_awaddr;
                aw_left_q <= {1'b0, m_awlen} + 9'd1;
                aw_busy   <= 1'b1;
                s_awready <= 1'b0;
                s_wready  <= 1'b1;
            end else if (s_awready && !m_awvalid) begin
                s_awready <= lfsr[4];
            end

            // ---- write data --------------------------------------------
            if (aw_busy && s_wready && m_wvalid) begin
                if (m_wstrb != 4'hF) begin
                    $display("FATAL axi_partial_write_strobe %0h", m_wstrb);
                    $finish;
                end
                mem[widx(aw_addr_q)] <= m_wdata;
                if (m_wlast != (aw_left_q == 9'd1)) begin
                    $display("FATAL axi_wlast_misplaced %0d %0d",
                             m_wlast, aw_left_q);
                    $finish;
                end
                aw_addr_q <= aw_addr_q + 64'd4;
                aw_left_q <= aw_left_q - 9'd1;
                if (m_wlast) begin
                    s_wready <= 1'b0;
                    aw_busy  <= 1'b0;
                    s_bvalid <= 1'b1;
                end else begin
                    s_wready <= lfsr[5];
                end
            end else if (aw_busy && !s_wready) begin
                s_wready <= lfsr[6];
            end

            if (s_bvalid && m_bready) s_bvalid <= 1'b0;
        end
    end

    //----------------------------------------------------------------------
    //  The AXI4-Lite master
    //----------------------------------------------------------------------

    task axil_write;
        input [11:0] addr;
        input [31:0] data;
        integer awdone, wdone, bdone, guard;
        begin
            @(posedge CLK); #1;
            c_awaddr = addr; c_awvalid = 1'b1;
            c_wdata  = data; c_wvalid  = 1'b1; c_wstrb = 4'hF;
            c_bready = 1'b1;
            awdone = 0; wdone = 0; bdone = 0; guard = 0;
            while (!(awdone && wdone && bdone)) begin
                @(posedge CLK);
                if (!awdone && c_awvalid && c_awready) awdone = 1;
                if (!wdone  && c_wvalid  && c_wready)  wdone  = 1;
                if (!bdone  && c_bready  && c_bvalid)  bdone  = 1;
                #1;
                if (awdone) c_awvalid = 1'b0;
                if (wdone)  c_wvalid  = 1'b0;
                if (bdone)  c_bready  = 1'b0;
                guard = guard + 1;
                if (guard > 1000) begin
                    $display("FATAL axil_write_stalled %03h", addr);
                    $finish;
                end
            end
        end
    endtask

    task axil_read;
        input  [11:0] addr;
        output [31:0] data;
        integer ardone, rdone, guard;
        begin
            @(posedge CLK); #1;
            c_araddr = addr; c_arvalid = 1'b1; c_rready = 1'b1;
            ardone = 0; rdone = 0; guard = 0;
            while (!rdone) begin
                @(posedge CLK);
                if (!ardone && c_arvalid && c_arready) ardone = 1;
                if (ardone && c_rvalid && c_rready) begin
                    data  = c_rdata;
                    rdone = 1;
                end
                #1;
                if (ardone) c_arvalid = 1'b0;
                if (rdone)  c_rready  = 1'b0;
                guard = guard + 1;
                if (guard > 1000) begin
                    $display("FATAL axil_read_stalled %03h", addr);
                    $finish;
                end
            end
        end
    endtask

    //----------------------------------------------------------------------
    //  The run
    //----------------------------------------------------------------------

    reg  [31:0] prog [0:PROGWORDS-1];
    reg  [1023:0] stim_path;
    integer     i, events, guard;
    reg  [31:0] ctrl, status, cap_capacity, prog_capacity;
    reg  [31:0] prog_words, stim_words, host_cap;
    reg  [31:0] header_events, header_cycles, shown;
    reg  [31:0] rec_tag, rec_cycle, rec_core, rec_neuron;

    initial begin
        rst_n     = 1'b0;
        c_awvalid = 1'b0; c_wvalid = 1'b0; c_arvalid = 1'b0;
        c_rready  = 1'b0; c_bready = 1'b0;
        c_awaddr  = 12'd0; c_araddr = 12'd0; c_wdata = 32'd0; c_wstrb = 4'h0;
        events    = 0;

        if (!$value$plusargs("stim=%s", stim_path)) begin
            $display("FATAL missing_plusarg +stim=<path>");
            $finish;
        end
        for (i = 0; i < PROGWORDS; i = i + 1) prog[i] = OP_END;
        $readmemh(stim_path, prog);
        for (i = 0; i < MEMWORDS; i = i + 1) mem[i] = 32'd0;

        // The host's split: [0, SPLIT) + END is the program buffer, the rest
        // + END is the stimulus buffer. The kernel lands the stimulus ON the
        // program's terminator, so the fabric sees one continuous stream.
        for (i = 0; i < SPLIT; i = i + 1)
            mem[(PROG_BASE / 4) + i] = prog[i];
        mem[(PROG_BASE / 4) + SPLIT] = OP_END;
        prog_words = SPLIT + 1;

        for (i = SPLIT; i < PROGWORDS - 1; i = i + 1)
            mem[(STIM_BASE / 4) + (i - SPLIT)] = prog[i];
        mem[(STIM_BASE / 4) + (PROGWORDS - 1 - SPLIT)] = OP_END;
        stim_words = PROGWORDS - SPLIT;

        repeat (8) @(posedge CLK);
        rst_n = 1'b1;
        repeat (16) @(posedge CLK);

        axil_read(ADDR_CAP_CAP,  cap_capacity);
        axil_read(ADDR_PROG_CAP, prog_capacity);
        host_cap = (HOSTCAP < cap_capacity) ? HOSTCAP : cap_capacity;

        axil_write(ADDR_PROG_LO, PROG_BASE[31:0]);
        axil_write(ADDR_PROG_HI, PROG_BASE[63:32]);
        axil_write(ADDR_STIM_LO, STIM_BASE[31:0]);
        axil_write(ADDR_STIM_HI, STIM_BASE[63:32]);
        axil_write(ADDR_CAP_LO,  CAP_BASE[31:0]);
        axil_write(ADDR_CAP_HI,  CAP_BASE[63:32]);
        axil_write(ADDR_PROG_N,  prog_words);
        axil_write(ADDR_STIM_N,  stim_words);
        axil_write(ADDR_CAP_N,   HOSTCAP);

        axil_write(ADDR_AP_CTRL, 32'd1);

        guard = 0;
        ctrl  = 32'd0;
        while ((ctrl & AP_DONE_BIT) == 32'd0) begin
            axil_read(ADDR_AP_CTRL, ctrl);
            guard = guard + 1;
            if (guard > 20000000) begin
                $display("FATAL kernel_never_finished");
                $finish;
            end
        end

        axil_read(ADDR_STATUS, status);

        header_events = mem[CAP_BASE / 4];
        header_cycles = mem[(CAP_BASE / 4) + 1];
        shown = (header_events < host_cap) ? header_events : host_cap;
        for (i = 0; i < shown; i = i + 1) begin
            rec_tag    = mem[(CAP_BASE / 4) + 2 + 4*i];
            rec_cycle  = mem[(CAP_BASE / 4) + 2 + 4*i + 1];
            rec_core   = mem[(CAP_BASE / 4) + 2 + 4*i + 2];
            rec_neuron = mem[(CAP_BASE / 4) + 2 + 4*i + 3];
            $display("EV %0d %0d %0d %0d", rec_core, rec_neuron, rec_cycle, rec_tag);
            events = events + 1;
        end

        // err, events the fabric SAW, fabric capacity, program capacity, the
        // host's declared capacity, and how many records reached memory.
        $display("KSTAT %0d %0d %0d %0d %0d %0d",
                 status[31], status[30:0], cap_capacity, prog_capacity,
                 host_cap, events);
        $display("KSPLIT %0d %0d %0d", prog_words, stim_words, PROGWORDS);
        $display("RBSTAT 0 0");
        $display("SHSTAT 0 0");
        $display("DONE %0d %0d", header_cycles, events);
        $finish;
    end

endmodule
