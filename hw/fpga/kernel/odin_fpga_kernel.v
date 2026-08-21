// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "odin_fpga_kernel.v" - P vendored ODIN cores, one SPI master and one AER
// bridge each, and an ON-FABRIC SEQUENCER that executes the SAME token program
// the cosimulation testbench executes. Original work: it instantiates the
// vendored core and derives no text from it. Verilog-2005 only.
//
// CROSS-LANGUAGE CONTRACT - the opcode table below is the third copy of ONE
// encoding whose home is `mimarsinan.chip_simulation.odin_rtl.stimulus`; the
// other copies are `hw/tb/tb_odin_core.v` (the host-driven testbench) and the
// host-side payload builder. The kernel-level smoke gate exists precisely to
// prove that the FABRIC sequencer implements the same semantics as the tb
// driver: the same program, against the same core, must produce the same
// per-neuron counts.
//
// SHADOW and PROG are testbench/variant opcodes with no fabric implementation
// (config registers have no readback path on real silicon, doc Sec.4), so the
// sequencer REFUSES them by raising `err` instead of skipping them silently.
// An SPI READ is executed on the wire but its byte-compare stays a HOST gate:
// the kernel has no expected image to compare against.
//
// Capture layout, mirrored in `chip_simulation/odin_fpga/xrt_transport.py`:
//   word 0            : events SEEN (may exceed the record capacity -- the
//                       host then refuses a TRUNCATED capture rather than
//                       reading the missing events as silent neurons)
//   word 1            : the free-running cycle count when the program ended
//   words 2 + 4*i ... : {tag, cycle, core, neuron} per captured event

`timescale 1ns/1ps

module odin_fpga_kernel #(
    parameter NC         = 1,      // vendored cores instantiated
    parameter N          = 256,    // neurons per core (stock geometry)
    parameter M          = 8,      // neuron-address width (stock geometry)
    parameter PROG_WORDS = 4096,   // program RAM depth, in 32-bit words
    parameter CAP_WORDS  = 4096    // capture RAM depth, in 32-bit words
) (
    input  wire        clk,
    input  wire        rst,

    // ap_ctrl (the Vitis RTL-kernel convention; the s_axilite control block in
    // `odin_fpga_kernel_top.v` drives ap_start and reads ap_done/ap_idle).
    input  wire        ap_start,
    output reg         ap_done,
    output reg         ap_idle,
    output reg         ap_ready,
    output reg         err,

    // Program RAM write port: the AXI loader fills it from the host's program
    // buffer before ap_start; the smoke testbench preloads it directly.
    input  wire        prog_we,
    input  wire [31:0] prog_waddr,
    input  wire [31:0] prog_wdata,

    // Capture RAM read port: drained to the host's capture buffer after ap_done.
    input  wire [31:0] cap_raddr,
    output reg  [31:0] cap_rdata
);

    localparam [31:0] OP_END    = 32'd0;
    localparam [31:0] OP_SPI_W  = 32'd1;
    localparam [31:0] OP_SPI_R  = 32'd2;
    localparam [31:0] OP_AER    = 32'd3;
    localparam [31:0] OP_WAIT   = 32'd4;
    localparam [31:0] OP_TAG    = 32'd5;

    localparam [31:0] CAP_HEADER = 32'd2;
    localparam PROG_AW  = $clog2(PROG_WORDS);
    localparam CAP_AW   = $clog2(CAP_WORDS);
    localparam AER_BITS = 2 * M + 1;

    reg [31:0] prog_ram [0:PROG_WORDS-1];
    reg [31:0] cap_ram  [0:CAP_WORDS-1];

    reg [31:0] pc, op, arg0, arg1, arg2;
    reg [31:0] tag, cycle, events_seen, cap_ptr, wait_left;

    reg        spi_start, aer_start, issued, saw_busy, is_read;
    reg [19:0] spi_addr, spi_data;
    reg [31:0] active_core;
    reg [AER_BITS-1:0] aer_word;

    wire [NC-1:0]    spi_busy_w, aer_busy_w, aer_timeout_w, miso_w, aerout_req_w;
    wire [M*NC-1:0]  aerout_addr_w;
    wire [20*NC-1:0] spi_rdata_w;
    reg  [NC-1:0]    aerout_ack_r;

    integer i;
    genvar c;

    //----------------------------------------------------------------------
    //  Program RAM, capture read port, free-running cycle counter
    //----------------------------------------------------------------------

    always @(posedge clk)
        if (prog_we) prog_ram[prog_waddr[PROG_AW-1:0]] <= prog_wdata;

    always @(posedge clk) cap_rdata <= cap_ram[cap_raddr[CAP_AW-1:0]];

    always @(posedge clk)
        if (rst) cycle <= 32'd0;
        else     cycle <= cycle + 32'd1;

    //----------------------------------------------------------------------
    //  The cores, their SPI masters and their AER-in bridges
    //----------------------------------------------------------------------

    generate
        for (c = 0; c < NC; c = c + 1) begin : core_gen
            wire sel = (active_core == c);
            wire sck_w, mosi_w, aerin_req_w, aerin_ack_w;
            wire [AER_BITS-1:0] aerin_addr_w;

            odin_spi_master spi_i (
                .clk (clk), .rst (rst), .start (spi_start && sel),
                .addr (spi_addr), .data (spi_data), .miso (miso_w[c]),
                .sck (sck_w), .mosi (mosi_w), .busy (spi_busy_w[c]),
                .rdata (spi_rdata_w[20*c+19:20*c])
            );

            odin_aer_bridge #(.ADDR_BITS(AER_BITS)) aer_i (
                .clk (clk), .rst (rst), .start (aer_start && sel),
                .addr (aer_word), .aer_ack (aerin_ack_w),
                .aer_req (aerin_req_w), .aer_addr (aerin_addr_w),
                .busy (aer_busy_w[c]), .timeout (aer_timeout_w[c])
            );

            ODIN #(.N(N), .M(M)) dut (
                .CLK         (clk),
                .RST         (rst),
                .SCK         (sck_w),
                .MOSI        (mosi_w),
                .MISO        (miso_w[c]),
                .AERIN_ADDR  (aerin_addr_w),
                .AERIN_REQ   (aerin_req_w),
                .AERIN_ACK   (aerin_ack_w),
                .AEROUT_ADDR (aerout_addr_w[M*c+M-1:M*c]),
                .AEROUT_REQ  (aerout_req_w[c]),
                .AEROUT_ACK  (aerout_ack_r[c])
            );
        end
    endgenerate

    //----------------------------------------------------------------------
    //  Capture engine: one four-word record per AER-out event. ACK is held
    //  off until the record is stored, so a core STALLS rather than dropping
    //  a spike -- a lost event would read as a silent neuron, which is the
    //  one failure this design must not have.
    //----------------------------------------------------------------------

    reg [1:0]  cap_state;
    reg [31:0] cap_core;
    reg        cap_active, cap_space;

    wire [31:0] cap_word =
          (cap_state == 2'd0) ? tag
        : (cap_state == 2'd1) ? cycle
        : (cap_state == 2'd2) ? cap_core
        : {{(32-M){1'b0}}, aerout_addr_w[M*cap_core[7:0] +: M]};

    always @(posedge clk) begin : capture
        reg found;
        if (rst) begin
            cap_state    <= 2'd0;
            cap_core     <= 32'd0;
            cap_active   <= 1'b0;
            cap_space    <= 1'b0;
            cap_ptr      <= CAP_HEADER;
            events_seen  <= 32'd0;
            aerout_ack_r <= {NC{1'b0}};
        end else if (!cap_active) begin
            found = 1'b0;
            for (i = 0; i < NC; i = i + 1) begin
                if (!found && aerout_req_w[i] && !aerout_ack_r[i]) begin
                    found      = 1'b1;
                    cap_core   <= i[31:0];
                    cap_active <= 1'b1;
                    cap_state  <= 2'd0;
                    cap_space  <= (cap_ptr + 32'd4) <= CAP_WORDS;
                end
                if (aerout_ack_r[i] && !aerout_req_w[i]) aerout_ack_r[i] <= 1'b0;
            end
        end else begin
            if (cap_space) begin
                cap_ram[cap_ptr[CAP_AW-1:0]] <= cap_word;
                cap_ptr <= cap_ptr + 32'd1;
            end
            cap_state <= cap_state + 2'd1;
            if (cap_state == 2'd3) begin
                // Counted even when the RAM is full: the host reads
                // `written > capacity` and REFUSES the run.
                events_seen                 <= events_seen + 32'd1;
                aerout_ack_r[cap_core[7:0]] <= 1'b1;
                cap_active                  <= 1'b0;
            end
        end
    end

    //----------------------------------------------------------------------
    //  The sequencer: the same token program, executed on the fabric
    //----------------------------------------------------------------------

    localparam S_IDLE  = 3'd0;
    localparam S_FETCH = 3'd1;
    localparam S_DEC   = 3'd2;
    localparam S_ARGS  = 3'd3;
    localparam S_SPI   = 3'd4;
    localparam S_AER   = 3'd5;
    localparam S_WAIT  = 3'd6;
    localparam S_DONE  = 3'd7;

    reg [2:0] state;
    reg [1:0] argc, argi;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; pc <= 32'd0; tag <= 32'd0; wait_left <= 32'd0;
            ap_done <= 1'b0; ap_idle <= 1'b1; ap_ready <= 1'b0; err <= 1'b0;
            spi_start <= 1'b0; aer_start <= 1'b0; active_core <= 32'd0;
            argc <= 2'd0; argi <= 2'd0; is_read <= 1'b0;
            issued <= 1'b0; saw_busy <= 1'b0;
        end else begin
            spi_start <= 1'b0;
            aer_start <= 1'b0;
            ap_ready  <= 1'b0;
            case (state)
                S_IDLE: begin
                    ap_idle <= 1'b1;
                    if (ap_start) begin
                        pc <= 32'd0; ap_done <= 1'b0; ap_idle <= 1'b0;
                        err <= 1'b0; state <= S_FETCH;
                    end
                end
                S_FETCH: begin
                    op    <= prog_ram[pc[PROG_AW-1:0]];
                    pc    <= pc + 32'd1;
                    state <= S_DEC;
                end
                S_DEC: begin
                    argi   <= 2'd0;
                    issued <= 1'b0;
                    case (op)
                        OP_END:   state <= S_DONE;
                        OP_SPI_W: begin argc <= 2'd3; is_read <= 1'b0; state <= S_ARGS; end
                        OP_SPI_R: begin argc <= 2'd3; is_read <= 1'b1; state <= S_ARGS; end
                        OP_AER:   begin argc <= 2'd2; state <= S_ARGS; end
                        OP_WAIT:  begin argc <= 2'd1; state <= S_ARGS; end
                        OP_TAG:   begin argc <= 2'd1; state <= S_ARGS; end
                        default: begin
                            // SHADOW / PROG / anything unknown: the fabric has
                            // no implementation, so it REFUSES rather than
                            // running a different program than the host built.
                            err   <= 1'b1;
                            state <= S_DONE;
                        end
                    endcase
                end
                S_ARGS: begin
                    case (argi)
                        2'd0: begin
                            arg0 <= prog_ram[pc[PROG_AW-1:0]];
                            if (op == OP_TAG) tag <= prog_ram[pc[PROG_AW-1:0]];
                        end
                        2'd1:    arg1 <= prog_ram[pc[PROG_AW-1:0]];
                        default: arg2 <= prog_ram[pc[PROG_AW-1:0]];
                    endcase
                    pc <= pc + 32'd1;
                    if (argi + 2'd1 == argc) begin
                        case (op)
                            OP_SPI_W, OP_SPI_R: state <= S_SPI;
                            OP_AER:             state <= S_AER;
                            OP_WAIT:            state <= S_WAIT;
                            default:            state <= S_FETCH;
                        endcase
                    end else begin
                        argi <= argi + 2'd1;
                    end
                end
                S_SPI: begin
                    if (!issued) begin
                        active_core <= arg0;
                        spi_addr    <= arg1[19:0];
                        spi_data    <= is_read ? 20'd0 : arg2[19:0];
                        spi_start   <= 1'b1;
                        issued      <= 1'b1;
                        saw_busy    <= 1'b0;
                    end else if (!saw_busy) begin
                        if (spi_busy_w[active_core[7:0]]) saw_busy <= 1'b1;
                    end else if (!spi_busy_w[active_core[7:0]]) begin
                        state <= S_FETCH;
                    end
                end
                S_AER: begin
                    if (!issued) begin
                        active_core <= arg0;
                        aer_word    <= arg1[AER_BITS-1:0];
                        aer_start   <= 1'b1;
                        issued      <= 1'b1;
                        saw_busy    <= 1'b0;
                    end else if (!saw_busy) begin
                        if (aer_busy_w[active_core[7:0]]) saw_busy <= 1'b1;
                    end else if (!aer_busy_w[active_core[7:0]]) begin
                        state <= S_FETCH;
                    end
                end
                S_WAIT: begin
                    if (!issued) begin
                        wait_left <= arg0;
                        issued    <= 1'b1;
                        if (arg0 == 32'd0) state <= S_FETCH;
                    end else if (wait_left <= 32'd1) begin
                        wait_left <= 32'd0;
                        state     <= S_FETCH;
                    end else begin
                        wait_left <= wait_left - 32'd1;
                    end
                end
                default: begin
                    cap_ram[0] <= events_seen;
                    cap_ram[1] <= cycle;
                    if (|aer_timeout_w) err <= 1'b1;
                    ap_done  <= 1'b1;
                    ap_ready <= 1'b1;
                    ap_idle  <= 1'b1;
                    if (!ap_start) state <= S_IDLE;
                end
            endcase
        end
    end

    // The SPI read data is on the wire for a board-side probe; the byte
    // compare against the exporter's image is the HOST's gate (§7 row 17).
    wire [19:0] spi_rdata_probe = spi_rdata_w[20*active_core[7:0] +: 20];

endmodule
