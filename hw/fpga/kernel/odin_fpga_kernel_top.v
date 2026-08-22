// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "odin_fpga_kernel_top.v" - the Vitis RTL-kernel wrapper around
// `odin_fpga_kernel`: an AXI4-Lite control slave carrying ap_ctrl and the
// six kernel arguments, and an AXI4 master that MOVES THE BYTES -- it
// burst-reads the program and the stimulus out of host memory into the
// fabric's program RAM, runs the sequencer, and burst-writes the capture back.
//
// The register map is the Vitis RTL-kernel convention and is the SECOND copy
// of one table whose host-side home is
// `mimarsinan.chip_simulation.odin_fpga.kernel_registers` (ARG_* and ADDR_*):
//
//   0x00 ap_ctrl   bit0 ap_start (W1S), bit1 ap_done (RC), bit2 ap_idle,
//                  bit3 ap_ready, bit7 auto_restart
//   0x04 GIER      0x08 IP_IER      0x0C IP_ISR   (declared, unused)
//   0x10/0x14 program buffer address (64-bit)
//   0x1C/0x20 stimulus buffer address
//   0x28/0x2C capture buffer address
//   0x34 program words   0x3C stimulus words   0x44 capture events
//   0x4C status: {err, events_seen[30:0]} -- read after ap_done
//   0x54 capture capacity, in EVENTS, of the fabric's capture RAM  (read-only)
//   0x5C program capacity, in WORDS, of the fabric's program RAM   (read-only)
//
// CROSS-LANGUAGE CONTRACT (the payload split). The host hands two END-terminated
// token streams. The DMA loads the program at word 0 and the stimulus AT THE
// PROGRAM'S LAST WORD, overwriting its END terminator, so the fabric executes
// one continuous stream -- exactly the stream `payload_bytes(program.ops +
// stimulus.ops)` would have produced. `kernel_registers.stimulus_base_word` is the
// host-side copy of that rule.
//
// SCOPE: this engine is proven against a behavioural AXI4 memory model in
// simulation (`hw/tb/tb_odin_fpga_kernel_axi.v`); silicon is P7b/B0.

`timescale 1ns/1ps

module odin_fpga_kernel_top #(
    parameter NC          = 1,
    parameter N           = 256,
    parameter M           = 8,
    // Programming ONE stock core over SPI is ~147.6k tokens (8192 synapse
    // words plus the neuron memory and the config registers, four words per
    // transaction), so the program RAM is sized per core with headroom; the
    // host reads the real depth back at 0x5C and REFUSES a run that overruns
    // it rather than wrapping the address.
    parameter PROG_WORDS  = NC * 262144,
    parameter CAP_WORDS   = 65536,
    parameter C_S_AXI_ADDR_WIDTH = 12,
    parameter C_M_AXI_ADDR_WIDTH = 64,
    parameter C_M_AXI_DATA_WIDTH = 32
) (
    input  wire        ap_clk,
    input  wire        ap_rst_n,

    // ---- AXI4-Lite control ------------------------------------------------
    input  wire                                s_axi_control_awvalid,
    output wire                                s_axi_control_awready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       s_axi_control_awaddr,
    input  wire                                s_axi_control_wvalid,
    output wire                                s_axi_control_wready,
    input  wire [31:0]                         s_axi_control_wdata,
    input  wire [3:0]                          s_axi_control_wstrb,
    input  wire                                s_axi_control_arvalid,
    output wire                                s_axi_control_arready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       s_axi_control_araddr,
    output wire                                s_axi_control_rvalid,
    input  wire                                s_axi_control_rready,
    output wire [31:0]                         s_axi_control_rdata,
    output wire [1:0]                          s_axi_control_rresp,
    output wire                                s_axi_control_bvalid,
    input  wire                                s_axi_control_bready,
    output wire [1:0]                          s_axi_control_bresp,
    output wire                                interrupt,

    // ---- AXI4 master to global memory (the shell's HBM/DDR) ---------------
    output wire                                m_axi_gmem_arvalid,
    input  wire                                m_axi_gmem_arready,
    output wire [C_M_AXI_ADDR_WIDTH-1:0]       m_axi_gmem_araddr,
    output wire [7:0]                          m_axi_gmem_arlen,
    output wire [2:0]                          m_axi_gmem_arsize,
    output wire [1:0]                          m_axi_gmem_arburst,
    input  wire                                m_axi_gmem_rvalid,
    output wire                                m_axi_gmem_rready,
    input  wire [C_M_AXI_DATA_WIDTH-1:0]       m_axi_gmem_rdata,
    input  wire [1:0]                          m_axi_gmem_rresp,
    input  wire                                m_axi_gmem_rlast,
    output wire                                m_axi_gmem_awvalid,
    input  wire                                m_axi_gmem_awready,
    output wire [C_M_AXI_ADDR_WIDTH-1:0]       m_axi_gmem_awaddr,
    output wire [7:0]                          m_axi_gmem_awlen,
    output wire [2:0]                          m_axi_gmem_awsize,
    output wire [1:0]                          m_axi_gmem_awburst,
    output wire                                m_axi_gmem_wvalid,
    input  wire                                m_axi_gmem_wready,
    output wire [C_M_AXI_DATA_WIDTH-1:0]       m_axi_gmem_wdata,
    output wire [C_M_AXI_DATA_WIDTH/8-1:0]     m_axi_gmem_wstrb,
    output wire                                m_axi_gmem_wlast,
    input  wire                                m_axi_gmem_bvalid,
    input  wire [1:0]                          m_axi_gmem_bresp,
    output wire                                m_axi_gmem_bready
);

    localparam ADDR_AP_CTRL  = 8'h00;
    localparam ADDR_GIE      = 8'h04;
    localparam ADDR_PROG_LO  = 8'h10;
    localparam ADDR_PROG_HI  = 8'h14;
    localparam ADDR_STIM_LO  = 8'h1C;
    localparam ADDR_STIM_HI  = 8'h20;
    localparam ADDR_CAP_LO   = 8'h28;
    localparam ADDR_CAP_HI   = 8'h2C;
    localparam ADDR_PROG_N   = 8'h34;
    localparam ADDR_STIM_N   = 8'h3C;
    localparam ADDR_CAP_N    = 8'h44;
    localparam ADDR_STATUS   = 8'h4C;
    localparam ADDR_CAP_CAP  = 8'h54;
    localparam ADDR_PROG_CAP = 8'h5C;

    // The datapath is 32-bit-beat only (to_bound/wdata/prog_wdata all assume
    // 4-byte beats); any other width must fail at elaboration, in simulation
    // and in synthesis alike, not silently narrow.
    generate
        if (C_M_AXI_DATA_WIDTH != 32) begin : g_unsupported_axi_width
            unsupported_C_M_AXI_DATA_WIDTH_use_32 guard_inst();
        end
    endgenerate

    // The capture layout the host decodes: two header words, four words per
    // record (odin_fpga_kernel.v CAP_HEADER, kernel_registers.CAPTURE_*_WORDS).
    localparam [31:0] CAP_HEADER  = 32'd2;
    localparam [31:0] CAP_STRIDE  = 32'd4;
    localparam [31:0] CAP_EVENTS  = (CAP_WORDS - 2) / 4;
    localparam [31:0] PROG_LIMIT  = PROG_WORDS;

    // One outstanding burst, INCR, one bus word per beat. 16 beats keeps every
    // burst inside a 4 KiB page once the boundary clamp below is applied.
    localparam [31:0] MAX_BEATS = 32'd16;
    localparam        BYTES_PER_BEAT = C_M_AXI_DATA_WIDTH / 8;
    localparam [2:0]  AXSIZE = (BYTES_PER_BEAT == 8) ? 3'd3 : 3'd2;
    localparam [1:0]  AXBURST_INCR = 2'b01;
    localparam [1:0]  AXRESP_OKAY = 2'b00;

    localparam D_IDLE    = 4'd0;
    localparam D_RD_AR   = 4'd1;
    localparam D_RD_R    = 4'd2;
    localparam D_RUN     = 4'd3;
    localparam D_PEEK    = 4'd4;
    localparam D_CAP_SET = 4'd5;
    localparam D_CAP_AW  = 4'd6;
    localparam D_CAP_FE  = 4'd7;
    localparam D_CAP_W   = 4'd8;
    localparam D_CAP_B   = 4'd9;
    localparam D_DONE    = 4'd10;

    wire rst = ~ap_rst_n;

    reg        ap_start_r;
    reg [63:0] prog_addr_r, stim_addr_r, cap_addr_r;
    reg [31:0] prog_words_r, stim_words_r, cap_events_r;
    reg        gie_r;

    reg        awready_r, wready_r, bvalid_r, arready_r, rvalid_r;
    reg [31:0] rdata_r;
    reg [C_S_AXI_ADDR_WIDTH-1:0] waddr_r;

    reg  [3:0]  dma_state;
    reg         phase_r;          // 0 = program payload, 1 = stimulus payload
    reg  [63:0] axi_addr_r;
    reg  [31:0] words_left_r, ram_idx_r, events_seen_r;
    reg  [8:0]  beats_left_r;
    reg         arvalid_r, awvalid_r;
    reg         krn_start_r, run_started_r;
    reg         dma_err_r, ap_done_r, ap_idle_r;
    reg  [1:0]  peek_wait_r;

    wire        ap_done_k, ap_idle_k, ap_ready_k, err_k;
    wire [31:0] cap_rdata_w;

    //----------------------------------------------------------------------
    //  Sizing: where the stimulus lands, whether it fits, how long a burst is
    //----------------------------------------------------------------------

    // The stimulus overwrites the program's END terminator (see the contract
    // note at the top): one stream, not two.
    wire [31:0] stim_base_w  = (prog_words_r == 32'd0)
                             ? 32'd0 : (prog_words_r - 32'd1);
    wire [31:0] total_words_w = stim_base_w + stim_words_r;
    wire        prog_over_w  = (prog_words_r > PROG_LIMIT)
                             || (total_words_w > PROG_LIMIT);

    wire [31:0] eff_events_w = (cap_events_r < CAP_EVENTS)
                             ? cap_events_r : CAP_EVENTS;
    wire [31:0] rec_events_w = (events_seen_r < eff_events_w)
                             ? events_seen_r : eff_events_w;
    wire [31:0] cap_words_w  = CAP_HEADER + CAP_STRIDE * rec_events_w;

    wire [31:0] chunk_w    = (words_left_r > MAX_BEATS) ? MAX_BEATS : words_left_r;
    wire [31:0] to_bound_w = 32'd1024 - {22'd0, axi_addr_r[11:2]};
    wire [31:0] beats_w    = (chunk_w > to_bound_w) ? to_bound_w : chunk_w;
    wire [7:0]  axlen_w    = beats_w[7:0] - 8'd1;

    wire        rd_beat_w  = m_axi_gmem_rvalid & m_axi_gmem_rready;
    wire        wr_beat_w  = m_axi_gmem_wvalid & m_axi_gmem_wready;

    //----------------------------------------------------------------------
    //  AXI4-Lite control: one 32-bit register file, ap_start self-clearing
    //----------------------------------------------------------------------

    wire dma_accept_w  = (dma_state == D_IDLE) & ap_start_r;
    wire ctrl_read_w   = s_axi_control_arvalid & arready_r;
    wire ap_done_ack_w = ctrl_read_w & (s_axi_control_araddr[7:0] == ADDR_AP_CTRL);
    wire status_err_w  = dma_err_r | err_k;

    assign s_axi_control_awready = awready_r;
    assign s_axi_control_wready  = wready_r;
    assign s_axi_control_bvalid  = bvalid_r;
    assign s_axi_control_bresp   = 2'b00;
    assign s_axi_control_arready = arready_r;
    assign s_axi_control_rvalid  = rvalid_r;
    assign s_axi_control_rdata   = rdata_r;
    assign s_axi_control_rresp   = 2'b00;
    assign interrupt             = gie_r & ap_done_r;

    always @(posedge ap_clk) begin
        if (rst) begin
            ap_start_r <= 1'b0; gie_r <= 1'b0;
            prog_addr_r <= 64'd0; stim_addr_r <= 64'd0; cap_addr_r <= 64'd0;
            prog_words_r <= 32'd0; stim_words_r <= 32'd0; cap_events_r <= 32'd0;
            awready_r <= 1'b0; wready_r <= 1'b0; bvalid_r <= 1'b0;
            arready_r <= 1'b0; rvalid_r <= 1'b0; rdata_r <= 32'd0;
            waddr_r <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            awready_r <= s_axi_control_awvalid & ~awready_r;
            if (s_axi_control_awvalid & ~awready_r) waddr_r <= s_axi_control_awaddr;
            wready_r  <= s_axi_control_wvalid & ~wready_r;
            if (s_axi_control_wvalid & wready_r) begin
                case (waddr_r[7:0])
                    ADDR_AP_CTRL: if (s_axi_control_wdata[0]) ap_start_r <= 1'b1;
                    ADDR_GIE:     gie_r        <= s_axi_control_wdata[0];
                    ADDR_PROG_LO: prog_addr_r[31:0]  <= s_axi_control_wdata;
                    ADDR_PROG_HI: prog_addr_r[63:32] <= s_axi_control_wdata;
                    ADDR_STIM_LO: stim_addr_r[31:0]  <= s_axi_control_wdata;
                    ADDR_STIM_HI: stim_addr_r[63:32] <= s_axi_control_wdata;
                    ADDR_CAP_LO:  cap_addr_r[31:0]   <= s_axi_control_wdata;
                    ADDR_CAP_HI:  cap_addr_r[63:32]  <= s_axi_control_wdata;
                    ADDR_PROG_N:  prog_words_r  <= s_axi_control_wdata;
                    ADDR_STIM_N:  stim_words_r  <= s_axi_control_wdata;
                    ADDR_CAP_N:   cap_events_r  <= s_axi_control_wdata;
                    default: ;
                endcase
                bvalid_r <= 1'b1;
            end else if (s_axi_control_bready & bvalid_r) begin
                bvalid_r <= 1'b0;
            end
            if (dma_accept_w) ap_start_r <= 1'b0;

            arready_r <= s_axi_control_arvalid & ~arready_r;
            if (ctrl_read_w) begin
                case (s_axi_control_araddr[7:0])
                    ADDR_AP_CTRL: rdata_r <= {28'd0, dma_accept_w, ap_idle_r,
                                              ap_done_r, ap_start_r};
                    ADDR_STATUS:  rdata_r <= {status_err_w, events_seen_r[30:0]};
                    ADDR_CAP_CAP: rdata_r <= CAP_EVENTS;
                    ADDR_PROG_CAP:rdata_r <= PROG_LIMIT;
                    default:      rdata_r <= 32'd0;
                endcase
                rvalid_r <= 1'b1;
            end else if (s_axi_control_rready & rvalid_r) begin
                rvalid_r <= 1'b0;
            end
        end
    end

    //----------------------------------------------------------------------
    //  The DMA engine: one AXI4 master, one outstanding INCR burst at a time.
    //  Reads fill the program RAM through the kernel's write port; the drain
    //  streams the capture RAM back out through the same master.
    //----------------------------------------------------------------------

    assign m_axi_gmem_arvalid = arvalid_r;
    assign m_axi_gmem_araddr  = axi_addr_r;
    assign m_axi_gmem_arlen   = axlen_w;
    assign m_axi_gmem_arsize  = AXSIZE;
    assign m_axi_gmem_arburst = AXBURST_INCR;
    assign m_axi_gmem_rready  = (dma_state == D_RD_R);

    assign m_axi_gmem_awvalid = awvalid_r;
    assign m_axi_gmem_awaddr  = axi_addr_r;
    assign m_axi_gmem_awlen   = axlen_w;
    assign m_axi_gmem_awsize  = AXSIZE;
    assign m_axi_gmem_awburst = AXBURST_INCR;
    assign m_axi_gmem_wvalid  = (dma_state == D_CAP_W);
    assign m_axi_gmem_wdata   = cap_rdata_w;
    assign m_axi_gmem_wstrb   = {(C_M_AXI_DATA_WIDTH/8){1'b1}};
    assign m_axi_gmem_wlast   = (beats_left_r == 9'd1);
    assign m_axi_gmem_bready  = (dma_state == D_CAP_B);

    always @(posedge ap_clk) begin
        if (rst) begin
            dma_state <= D_IDLE; phase_r <= 1'b0;
            axi_addr_r <= 64'd0; words_left_r <= 32'd0; ram_idx_r <= 32'd0;
            beats_left_r <= 9'd0; events_seen_r <= 32'd0; peek_wait_r <= 2'd0;
            arvalid_r <= 1'b0; awvalid_r <= 1'b0;
            krn_start_r <= 1'b0; run_started_r <= 1'b0;
            dma_err_r <= 1'b0; ap_done_r <= 1'b0; ap_idle_r <= 1'b1;
        end else begin
            if (ap_done_ack_w) ap_done_r <= 1'b0;
            case (dma_state)
                D_IDLE: begin
                    ap_idle_r <= 1'b1;
                    if (ap_start_r) begin
                        ap_idle_r     <= 1'b0;
                        dma_err_r     <= 1'b0;
                        ap_done_r     <= 1'b0;
                        events_seen_r <= 32'd0;
                        if (prog_over_w) begin
                            // The declared payload does not fit the fabric's
                            // program RAM: REFUSE rather than wrap the address
                            // and execute a program nobody assembled.
                            dma_err_r <= 1'b1;
                            dma_state <= D_DONE;
                        end else begin
                            phase_r      <= 1'b0;
                            axi_addr_r   <= prog_addr_r;
                            words_left_r <= prog_words_r;
                            ram_idx_r    <= 32'd0;
                            dma_state    <= D_RD_AR;
                        end
                    end
                end
                D_RD_AR: begin
                    if (words_left_r == 32'd0) begin
                        if (phase_r == 1'b0) begin
                            phase_r      <= 1'b1;
                            axi_addr_r   <= stim_addr_r;
                            words_left_r <= stim_words_r;
                            ram_idx_r    <= stim_base_w;
                        end else begin
                            dma_state <= D_RUN;
                        end
                    end else if (!arvalid_r) begin
                        arvalid_r    <= 1'b1;
                        beats_left_r <= beats_w[8:0];
                    end else if (m_axi_gmem_arready) begin
                        arvalid_r <= 1'b0;
                        dma_state <= D_RD_R;
                    end
                end
                D_RD_R: begin
                    if (rd_beat_w) begin
                        ram_idx_r    <= ram_idx_r + 32'd1;
                        axi_addr_r   <= axi_addr_r + BYTES_PER_BEAT;
                        words_left_r <= words_left_r - 32'd1;
                        beats_left_r <= beats_left_r - 9'd1;
                        if (m_axi_gmem_rresp != AXRESP_OKAY) dma_err_r <= 1'b1;
                        if (m_axi_gmem_rlast) begin
                            if (beats_left_r != 9'd1) dma_err_r <= 1'b1;
                            dma_state <= D_RD_AR;
                        end
                    end
                end
                D_RUN: begin
                    if (!krn_start_r && !run_started_r) begin
                        krn_start_r <= 1'b1;
                    end else if (krn_start_r && !ap_idle_k) begin
                        krn_start_r   <= 1'b0;
                        run_started_r <= 1'b1;
                    end else if (run_started_r && ap_done_k) begin
                        run_started_r <= 1'b0;
                        ram_idx_r     <= 32'd0;
                        peek_wait_r   <= 2'd3;
                        dma_state     <= D_PEEK;
                    end
                end
                D_PEEK: begin
                    // The capture RAM read port is registered; word 0 is the
                    // fabric's own event count and it decides the drain length.
                    if (peek_wait_r != 2'd0) begin
                        peek_wait_r <= peek_wait_r - 2'd1;
                    end else begin
                        events_seen_r <= cap_rdata_w;
                        dma_state     <= D_CAP_SET;
                    end
                end
                D_CAP_SET: begin
                    axi_addr_r   <= cap_addr_r;
                    words_left_r <= cap_words_w;
                    ram_idx_r    <= 32'd0;
                    dma_state    <= D_CAP_AW;
                end
                D_CAP_AW: begin
                    if (words_left_r == 32'd0) begin
                        dma_state <= D_DONE;
                    end else if (!awvalid_r) begin
                        awvalid_r    <= 1'b1;
                        beats_left_r <= beats_w[8:0];
                    end else if (m_axi_gmem_awready) begin
                        awvalid_r <= 1'b0;
                        dma_state <= D_CAP_FE;
                    end
                end
                D_CAP_FE: dma_state <= D_CAP_W;
                D_CAP_W: begin
                    if (wr_beat_w) begin
                        ram_idx_r    <= ram_idx_r + 32'd1;
                        axi_addr_r   <= axi_addr_r + BYTES_PER_BEAT;
                        words_left_r <= words_left_r - 32'd1;
                        beats_left_r <= beats_left_r - 9'd1;
                        dma_state    <= (beats_left_r == 9'd1) ? D_CAP_B : D_CAP_FE;
                    end
                end
                D_CAP_B: begin
                    if (m_axi_gmem_bvalid) begin
                        if (m_axi_gmem_bresp != AXRESP_OKAY) dma_err_r <= 1'b1;
                        dma_state <= D_CAP_AW;
                    end
                end
                default: begin
                    ap_done_r <= 1'b1;
                    ap_idle_r <= 1'b1;
                    dma_state <= D_IDLE;
                end
            endcase
        end
    end

    odin_fpga_kernel #(
        .NC(NC), .N(N), .M(M),
        .PROG_WORDS(PROG_WORDS), .CAP_WORDS(CAP_WORDS)
    ) kernel_i (
        .clk        (ap_clk),
        .rst        (rst),
        .ap_start   (krn_start_r),
        .ap_done    (ap_done_k),
        .ap_idle    (ap_idle_k),
        .ap_ready   (ap_ready_k),
        .err        (err_k),
        .prog_we    (rd_beat_w & (dma_state == D_RD_R)),
        .prog_waddr (ram_idx_r),
        .prog_wdata (m_axi_gmem_rdata[31:0]),
        .cap_raddr  (ram_idx_r),
        .cap_rdata  (cap_rdata_w)
    );

    // ap_ready is a wrapper-level pulse (dma_accept_w); the kernel's own is
    // consumed by the D_RUN handshake and named here so a sweep does not hide
    // that it is part of the contract.
    wire _unused_kernel_ready = ap_ready_k;

endmodule
