// Copyright (c) 2026 Yigit Polat. MIT licence (see the repository LICENSE).
//
// "odin_fpga_kernel.v" - P vendored ODIN cores, one SPI master and one AER
// bridge each, and an ON-FABRIC SEQUENCER that executes the SAME token program
// the cosimulation testbench executes. Original work: it instantiates the
// vendored core and derives no text from it. Verilog-2005 only.
//
// THE FPGA IS THE CHIP; IT DOES NOT PROGRAM ITSELF. There is no program RAM.
// The op stream arrives LIVE from the runtime host at segment/pass boundaries
// and passes through the shallow elastic FIFO below; the sequencer consumes the
// head of that FIFO and has no program counter and no addressing. Programming
// bandwidth is therefore a DEPLOYMENT METRIC of the host link, not a fabric
// storage budget, and the stream is bounded only by the host's 32-bit word
// counts.
//
// DETERMINISM IS THE CONTRACT (the atol=0 certificates). The vendored core
// free-runs and WAIT timing is semantics, so a starved FIFO must not become a
// different program. CORE-ENABLE GATING is how that holds: when the sequencer
// wants a word the FIFO cannot yet give it, `core_en` drops and the WHOLE core
// domain -- the vendored cores through a glitch-free clock gate, the SPI
// masters, the AER bridges, the sequencer, the capture engine and the cycle
// counter -- stands still for that cycle. Core time is ENABLED cycles, so
// execution is bit-exact under ANY stream arrival pattern.
// `hw/tb/tb_odin_fpga_kernel_axi.v` injects seeded starvation and the
// stall-invariance gate compares the results at atol=0.
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
// Capture layout, mirrored in `chip_simulation/odin_fpga/kernel_registers.py`:
//   word 0            : events SEEN. It is counted even when the RAM is full,
//                       so it may EXCEED the record capacity; the host reads
//                       that capacity out of the wrapper's read-only 0x54
//                       register and refuses at or over it rather than reading
//                       the missing events as silent neurons.
//   word 1            : the ENABLED-cycle count when the program ended
//   words 2 + 4*i ... : {tag, cycle, core, neuron} per captured event

`timescale 1ns/1ps

module odin_fpga_kernel #(
    parameter NC         = 1,      // vendored cores instantiated
    parameter N          = 256,    // neurons per core (stock geometry)
    parameter M          = 8,      // neuron-address width (stock geometry)
    // The elastic buffer between the host link and the sequencer, in 32-bit
    // words, a power of two. It only has to cover the read engine's burst
    // granularity and the link's jitter: the SPI programming wall is ~40 SCK
    // per transaction, so a stream that keeps ahead at all keeps ahead by
    // orders of magnitude. 1,024 words is one RAMB36E2.
    parameter FIFO_WORDS = 1024,
    parameter CAP_WORDS  = 16384   // capture RAM depth, in 32-bit words
) (
    input  wire        clk,
    input  wire        rst,

    // ap_ctrl (the Vitis RTL-kernel convention; the s_axilite control block in
    // `odin_fpga_kernel_top.v` drives ap_start and reads ap_done/ap_idle).
    input  wire        ap_start,
    output reg         ap_done,
    output reg         ap_idle,
    output reg         ap_ready,
    output wire        err,

    // The op stream, pushed by the wrapper's AXI read engine. `str_space` is
    // the producer's CREDIT in words: the engine reserves a burst's worth
    // before it issues the address, so a push never meets a full FIFO.
    input  wire        str_valid,
    input  wire [31:0] str_data,
    output wire [31:0] str_space,

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
    localparam FIFO_AW  = $clog2(FIFO_WORDS);
    localparam CAP_AW   = $clog2(CAP_WORDS);
    localparam AER_BITS = 2 * M + 1;
    localparam [FIFO_AW:0] FIFO_DEPTH = FIFO_WORDS;

    // The FIFO's pointers wrap on their own width, so a depth that is not a
    // power of two would silently address a hole; fail at elaboration instead.
    generate
        if ((1 << FIFO_AW) != FIFO_WORDS) begin : g_fifo_depth
            unsupported_FIFO_WORDS_use_a_power_of_two guard_inst();
        end
    endgenerate

    localparam S_IDLE  = 3'd0;
    localparam S_FETCH = 3'd1;
    localparam S_ARG   = 3'd2;
    localparam S_SPI   = 3'd3;
    localparam S_AER   = 3'd4;
    localparam S_WAIT  = 3'd5;
    localparam S_DONE  = 3'd6;

    reg [2:0]  state;
    reg [1:0]  argc, argi;
    reg        err_r;
    reg [31:0] op, arg0, arg1, arg2;
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

    // Each ap_start re-arms the capture, flushes the FIFO and rezeroes the
    // cycle counter: a session that runs many passes through one loaded kernel
    // must report THIS run's events and THIS run's enabled-cycle time, not the
    // sum since the xclbin was loaded.
    wire run_start = (state == S_IDLE) && ap_start;

    //----------------------------------------------------------------------
    //  The elastic op-stream FIFO: a plain hand-rolled synchronous FIFO whose
    //  storage is ONE block RAM (one read point at `rd_ptr`, one write point at
    //  `wr_ptr`) and whose head sits in `ram_q`, so the consumer is handed the
    //  WORD and never an address. No vendor macro.
    //
    //  The PRODUCER side is free-running -- it is host/DMA time. The CONSUMER
    //  side is core time. `lift` is deliberately NOT gated: refilling the head
    //  register while the core domain is frozen is exactly how a stall ends.
    //----------------------------------------------------------------------

    (* ram_style = "block" *) reg [31:0] fifo_ram [0:FIFO_WORDS-1];
    (* ram_style = "block" *) reg [31:0] cap_ram  [0:CAP_WORDS-1];

    reg [FIFO_AW-1:0] wr_ptr, rd_ptr;
    reg [FIFO_AW:0]   fifo_fill;
    reg [31:0]        ram_q;
    reg               ram_q_v;
    reg               fifo_ovf;

    wire        pop_valid  = ram_q_v;
    wire [31:0] str_head   = ram_q;
    wire        fifo_ne    = (fifo_fill != {(FIFO_AW+1){1'b0}});
    wire        fifo_room  = (fifo_fill != FIFO_DEPTH);
    wire        push       = str_valid;
    wire        pop_fire;                    // the sequencer's fetch states
    wire        lift       = fifo_ne && (!ram_q_v || pop_fire);

    assign str_space = FIFO_DEPTH - fifo_fill;

    // A push and a lift never name the same address: `lift` needs a resident
    // word, and the word a push is placing is not resident until the next edge.
    always @(posedge clk) begin
        if (push) fifo_ram[wr_ptr] <= str_data;
        if (lift) ram_q <= fifo_ram[rd_ptr];
    end

    always @(posedge clk) begin
        if (rst || run_start) begin
            wr_ptr    <= {FIFO_AW{1'b0}};
            rd_ptr    <= {FIFO_AW{1'b0}};
            fifo_fill <= {(FIFO_AW+1){1'b0}};
            ram_q_v   <= 1'b0;
            fifo_ovf  <= 1'b0;
        end else begin
            if (push) wr_ptr <= wr_ptr + 1'b1;
            if (lift) rd_ptr <= rd_ptr + 1'b1;
            case ({push, lift})
                2'b10:   fifo_fill <= fifo_fill + 1'b1;
                2'b01:   fifo_fill <= fifo_fill - 1'b1;
                default: ;
            endcase
            if (lift)          ram_q_v <= 1'b1;
            else if (pop_fire) ram_q_v <= 1'b0;
            // The producer's credit is the contract. A push into a full FIFO
            // would drop a word of the host's program, which would run a
            // program nobody assembled, so it REFUSES the run instead.
            if (push && !fifo_room) fifo_ovf <= 1'b1;
        end
    end

    //----------------------------------------------------------------------
    //  Core-enable gating: core time is ENABLED cycles
    //----------------------------------------------------------------------

    wire seq_fetching = (state == S_FETCH) || (state == S_ARG);
    // Held high through reset so the cores are clocked into their reset state
    // while `state` is still X.
    wire core_en      = rst || !(seq_fetching && !pop_valid);

    assign pop_fire = seq_fetching && pop_valid;

    // The glitch-free clock gate: the enable is captured on the FALLING edge,
    // so it is stable across the whole high phase it gates. This is the shape
    // Vivado maps onto a BUFGCE, and it is the only way to freeze a vendored
    // core whose RTL stays byte-untouched -- ODIN declares no clock enable.
    reg core_en_q;
    always @(negedge clk) core_en_q <= core_en;
    wire core_clk = clk & (core_en_q | rst);

    always @(posedge clk)
        if (rst || run_start) cycle <= 32'd0;
        else if (core_en)     cycle <= cycle + 32'd1;

    //----------------------------------------------------------------------
    //  The cores, their SPI masters and their AER-in bridges
    //----------------------------------------------------------------------

    generate
        for (c = 0; c < NC; c = c + 1) begin : core_gen
            wire sel = (active_core == c);
            wire sck_w, mosi_w, aerin_req_w, aerin_ack_w;
            wire [AER_BITS-1:0] aerin_addr_w;

            odin_spi_master spi_i (
                .clk (core_clk), .rst (rst), .start (spi_start && sel),
                .addr (spi_addr), .data (spi_data), .miso (miso_w[c]),
                .sck (sck_w), .mosi (mosi_w), .busy (spi_busy_w[c]),
                .rdata (spi_rdata_w[20*c+19:20*c])
            );

            odin_aer_bridge #(.ADDR_BITS(AER_BITS)) aer_i (
                .clk (core_clk), .rst (rst), .start (aer_start && sel),
                .addr (aer_word), .aer_ack (aerin_ack_w),
                .aer_req (aerin_req_w), .aer_addr (aerin_addr_w),
                .busy (aer_busy_w[c]), .timeout (aer_timeout_w[c])
            );

            ODIN #(.N(N), .M(M)) dut (
                .CLK         (core_clk),
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
    reg [1:0]  hdr_written;

    // Raised for good when the sequencer reaches its terminal state; the two
    // header words are written once each, at DRAIN time, when the streaming
    // writes have stopped and the port is free.
    reg  run_ended;

    wire [31:0] cap_word =
          (cap_state == 2'd0) ? tag
        : (cap_state == 2'd1) ? cycle
        : (cap_state == 2'd2) ? cap_core
        : {{(32-M){1'b0}}, aerout_addr_w[M*cap_core[7:0] +: M]};

    wire        hdr_we    = run_ended && !cap_active && (hdr_written != 2'd2);
    wire        cap_we    = (cap_active && cap_space) || hdr_we;
    wire [31:0] cap_waddr = cap_active ? cap_ptr : {31'd0, hdr_written[0]};
    wire [31:0] cap_wdata = cap_active ? cap_word
                          : (hdr_written[0] ? cycle : events_seen);

    // The capture RAM's ONLY port pair, split by the domain each side lives in:
    // the record/header WRITE is core time and stands still with the core; the
    // drain READ is the wrapper's DMA time and must answer whenever asked.
    always @(posedge clk)
        cap_rdata <= cap_ram[cap_raddr[CAP_AW-1:0]];

    always @(posedge clk)
        if (core_en && cap_we) cap_ram[cap_waddr[CAP_AW-1:0]] <= cap_wdata;

    always @(posedge clk) begin : capture
        reg found;
        if (rst || run_start) begin
            cap_state    <= 2'd0;
            cap_core     <= 32'd0;
            cap_active   <= 1'b0;
            cap_space    <= 1'b0;
            cap_ptr      <= CAP_HEADER;
            events_seen  <= 32'd0;
            hdr_written  <= 2'd0;
            aerout_ack_r <= {NC{1'b0}};
        end else if (core_en) begin
            if (!cap_active) begin
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
                if (hdr_we) hdr_written <= hdr_written + 2'd1;
            end else begin
                if (cap_space) cap_ptr <= cap_ptr + 32'd1;
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
    end

    //----------------------------------------------------------------------
    //  The sequencer: the same token program, executed on the fabric
    //----------------------------------------------------------------------

    // ONE word per fetch state, taken from the head of the FIFO. Inside the
    // `core_en` guard a fetch state ALWAYS has its word -- that is what
    // `core_en` MEANS -- so the sequencer carries no starvation case at all,
    // and its enabled-cycle schedule cannot depend on when the words arrived.
    assign err = err_r | fifo_ovf;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; tag <= 32'd0; wait_left <= 32'd0;
            ap_done <= 1'b0; ap_idle <= 1'b1; ap_ready <= 1'b0; err_r <= 1'b0;
            spi_start <= 1'b0; aer_start <= 1'b0; active_core <= 32'd0;
            argc <= 2'd0; argi <= 2'd0; is_read <= 1'b0;
            issued <= 1'b0; saw_busy <= 1'b0; run_ended <= 1'b0;
        end else if (core_en) begin
            spi_start <= 1'b0;
            aer_start <= 1'b0;
            ap_ready  <= 1'b0;
            case (state)
                S_IDLE: begin
                    ap_idle <= 1'b1;
                    if (ap_start) begin
                        ap_done <= 1'b0; ap_idle <= 1'b0;
                        err_r <= 1'b0; run_ended <= 1'b0; state <= S_FETCH;
                    end
                end
                S_FETCH: begin
                    op     <= str_head;
                    argi   <= 2'd0;
                    issued <= 1'b0;
                    case (str_head)
                        OP_END:   state <= S_DONE;
                        OP_SPI_W: begin argc <= 2'd3; is_read <= 1'b0; state <= S_ARG; end
                        OP_SPI_R: begin argc <= 2'd3; is_read <= 1'b1; state <= S_ARG; end
                        OP_AER:   begin argc <= 2'd2; state <= S_ARG; end
                        OP_WAIT:  begin argc <= 2'd1; state <= S_ARG; end
                        OP_TAG:   begin argc <= 2'd1; state <= S_ARG; end
                        default: begin
                            // SHADOW / PROG / anything unknown: the fabric has
                            // no implementation, so it REFUSES rather than
                            // running a different program than the host built.
                            err_r <= 1'b1;
                            state <= S_DONE;
                        end
                    endcase
                end
                S_ARG: begin
                    case (argi)
                        2'd0: begin
                            arg0 <= str_head;
                            if (op == OP_TAG) tag <= str_head;
                        end
                        2'd1:    arg1 <= str_head;
                        default: arg2 <= str_head;
                    endcase
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
                    // The header goes out through the capture RAM's single
                    // write port: `run_ended` hands the two words to the
                    // arbiter above, which writes them while the port is idle.
                    run_ended <= 1'b1;
                    if (|aer_timeout_w) err_r <= 1'b1;
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
