// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "odin_fpga_kernel_top.v" - the Vitis RTL-kernel wrapper around
// `odin_fpga_kernel`: an AXI4-Lite control slave carrying ap_ctrl and the
// six kernel arguments, and an AXI4 master that moves the program and the
// stimulus in and the capture out of host memory on the U55C XDMA shell.
//
// The register map is the Vitis RTL-kernel convention and is the SECOND copy
// of one table whose host-side home is
// `mimarsinan.chip_simulation.odin_fpga.xrt_transport` (ARG_* and CTRL_OFFSET):
//
//   0x00 ap_ctrl   bit0 ap_start (W1S), bit1 ap_done (RC), bit2 ap_idle,
//                  bit3 ap_ready, bit7 auto_restart
//   0x04 GIER      0x08 IP_IER      0x0C IP_ISR   (declared, unused)
//   0x10/0x14 program buffer address (64-bit)
//   0x1C/0x20 stimulus buffer address
//   0x28/0x2C capture buffer address
//   0x34 program words   0x3C stimulus words   0x44 capture events
//   0x4C status: {err, events_seen[30:0]} -- read after ap_done
//
// SCOPE: the DMA engine here is the P7b bring-up target; what P7a proves is
// that the fabric SEQUENCER implements the host program's semantics
// (`hw/tb/tb_odin_fpga_kernel.v`) and that the whole tree ELABORATES.

`timescale 1ns/1ps

module odin_fpga_kernel_top #(
    parameter NC          = 1,
    parameter N           = 256,
    parameter M           = 8,
    parameter PROG_WORDS  = 4096,
    parameter CAP_WORDS   = 4096,
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
    input  wire                                m_axi_gmem_rvalid,
    output wire                                m_axi_gmem_rready,
    input  wire [C_M_AXI_DATA_WIDTH-1:0]       m_axi_gmem_rdata,
    input  wire                                m_axi_gmem_rlast,
    output wire                                m_axi_gmem_awvalid,
    input  wire                                m_axi_gmem_awready,
    output wire [C_M_AXI_ADDR_WIDTH-1:0]       m_axi_gmem_awaddr,
    output wire [7:0]                          m_axi_gmem_awlen,
    output wire                                m_axi_gmem_wvalid,
    input  wire                                m_axi_gmem_wready,
    output wire [C_M_AXI_DATA_WIDTH-1:0]       m_axi_gmem_wdata,
    output wire [C_M_AXI_DATA_WIDTH/8-1:0]     m_axi_gmem_wstrb,
    output wire                                m_axi_gmem_wlast,
    input  wire                                m_axi_gmem_bvalid,
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

    wire rst = ~ap_rst_n;

    reg        ap_start_r;
    reg [63:0] prog_addr_r, stim_addr_r, cap_addr_r;
    reg [31:0] prog_words_r, stim_words_r, cap_events_r;
    reg        gie_r;

    reg        awready_r, wready_r, bvalid_r, arready_r, rvalid_r;
    reg [31:0] rdata_r;
    reg [C_S_AXI_ADDR_WIDTH-1:0] waddr_r;

    wire        ap_done_w, ap_idle_w, ap_ready_w, err_w;
    wire [31:0] cap_rdata_w;

    assign s_axi_control_awready = awready_r;
    assign s_axi_control_wready  = wready_r;
    assign s_axi_control_bvalid  = bvalid_r;
    assign s_axi_control_bresp   = 2'b00;
    assign s_axi_control_arready  = arready_r;
    assign s_axi_control_rvalid   = rvalid_r;
    assign s_axi_control_rdata    = rdata_r;
    assign s_axi_control_rresp    = 2'b00;
    assign interrupt              = gie_r & ap_done_w;

    //----------------------------------------------------------------------
    //  AXI4-Lite control: one 32-bit register file, ap_start self-clearing
    //----------------------------------------------------------------------

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
            if (ap_ready_w) ap_start_r <= 1'b0;

            arready_r <= s_axi_control_arvalid & ~arready_r;
            if (s_axi_control_arvalid & arready_r) begin
                case (s_axi_control_araddr[7:0])
                    ADDR_AP_CTRL: rdata_r <= {28'd0, ap_ready_w, ap_idle_w,
                                              ap_done_w, ap_start_r};
                    ADDR_STATUS:  rdata_r <= {err_w, cap_rdata_w[30:0]};
                    default:      rdata_r <= 32'd0;
                endcase
                rvalid_r <= 1'b1;
            end else if (s_axi_control_rready & rvalid_r) begin
                rvalid_r <= 1'b0;
            end
        end
    end

    //----------------------------------------------------------------------
    //  The DMA engine (P7b bring-up target): the declared AXI4 master that
    //  streams the program/stimulus in and drains the capture out. The
    //  handshake wires are tied to their idle values here so the wrapper
    //  ELABORATES and its port list is the real one the packaging step needs.
    //----------------------------------------------------------------------

    assign m_axi_gmem_arvalid = 1'b0;
    assign m_axi_gmem_araddr  = prog_addr_r;
    assign m_axi_gmem_arlen   = 8'd0;
    assign m_axi_gmem_rready  = 1'b1;
    assign m_axi_gmem_awvalid = 1'b0;
    assign m_axi_gmem_awaddr  = cap_addr_r;
    assign m_axi_gmem_awlen   = 8'd0;
    assign m_axi_gmem_wvalid  = 1'b0;
    assign m_axi_gmem_wdata   = cap_rdata_w;
    assign m_axi_gmem_wstrb   = {(C_M_AXI_DATA_WIDTH/8){1'b1}};
    assign m_axi_gmem_wlast   = 1'b0;
    assign m_axi_gmem_bready  = 1'b1;

    wire        prog_we_w    = m_axi_gmem_rvalid & m_axi_gmem_rready;
    reg  [31:0] prog_waddr_r;

    always @(posedge ap_clk) begin
        if (rst || ap_start_r) prog_waddr_r <= 32'd0;
        else if (prog_we_w)    prog_waddr_r <= prog_waddr_r + 32'd1;
    end

    odin_fpga_kernel #(
        .NC(NC), .N(N), .M(M),
        .PROG_WORDS(PROG_WORDS), .CAP_WORDS(CAP_WORDS)
    ) kernel_i (
        .clk        (ap_clk),
        .rst        (rst),
        .ap_start   (ap_start_r),
        .ap_done    (ap_done_w),
        .ap_idle    (ap_idle_w),
        .ap_ready   (ap_ready_w),
        .err        (err_w),
        .prog_we    (prog_we_w),
        .prog_waddr (prog_waddr_r),
        .prog_wdata (m_axi_gmem_rdata[31:0]),
        .cap_raddr  (32'd0),
        .cap_rdata  (cap_rdata_w)
    );

    // Declared and read by the status register / DMA engine; named here so an
    // unused-signal sweep does not hide that they are part of the contract.
    wire [31:0] _unused_counts = stim_words_r ^ prog_words_r ^ cap_events_r
                                 ^ stim_addr_r[31:0];

endmodule
